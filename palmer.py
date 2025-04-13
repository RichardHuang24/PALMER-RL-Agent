import os
import random
import time
from collections import deque
from dataclasses import dataclass

import gymnasium as gym
import memory_gym  # noqa
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from einops import rearrange
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
#from pom_env import PoMEnv  # noqa
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""

    # Algorithm specific arguments
    env_id: str = "MortarMayhem-Grid-v0"
    """the id of the environment"""
    total_timesteps: int = 200000000
    """total timesteps of the experiments"""
    init_lr: float = 2.75e-4
    """the initial learning rate of the optimizer"""
    final_lr: float = 1.0e-5
    """the final learning rate of the optimizer after linearly annealing"""
    num_envs: int = 32
    """the number of parallel game environments"""
    num_steps: int = 512
    """the number of steps to run in each environment per policy rollout"""
    anneal_steps: int = 32 * 512 * 10000
    """the number of steps to linearly anneal the learning rate and entropy coefficient from initial to final"""
    gamma: float = 0.995
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 8
    """the number of mini-batches"""
    update_epochs: int = 3
    """the K epochs to update the policy"""
    norm_adv: bool = False
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    init_ent_coef: float = 0.0001
    """initial coefficient of the entropy bonus"""
    final_ent_coef: float = 0.000001
    """final coefficient of the entropy bonus after linearly annealing"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.25
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # Transformer-XL specific arguments
    trxl_num_layers: int = 3
    """the number of transformer layers"""
    trxl_num_heads: int = 4
    """the number of heads used in multi-head attention"""
    trxl_dim: int = 384
    """the dimension of the transformer"""
    trxl_memory_length: int = 119
    """the length of TrXL's sliding memory window"""
    trxl_positional_encoding: str = "absolute"
    """the positional encoding type of the transformer, choices: "", "absolute", "learned" """
    reconstruction_coef: float = 0.0
    """the coefficient of the observation reconstruction loss, if set to 0.0 the reconstruction loss is not used"""

    # To be filled on runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, run_name, render_mode="debug_rgb_array"):
    if "MiniGrid" in env_id:
        if render_mode == "debug_rgb_array":
            render_mode = "rgb_array"

    def thunk():
        if "MiniGrid" in env_id:
            env = gym.make(env_id, agent_view_size=3, tile_size=28, render_mode=render_mode)
            env = ImgObsWrapper(RGBImgPartialObsWrapper(env, tile_size=28))
            env = gym.wrappers.TimeLimit(env, 96)
        else:
            env = gym.make(env_id, render_mode=render_mode)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        return gym.wrappers.RecordEpisodeStatistics(env)

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    # torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def batched_index_select(input, dim, index):
    for ii in range(1, len(input.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)


class PositionalEncoding(nn.Module):
    def __init__(self, dim, min_timescale=2.0, max_timescale=1e4):
        super().__init__()
        freqs = torch.arange(0, dim, min_timescale)
        inv_freqs = max_timescale ** (-freqs / dim)
        self.register_buffer("inv_freqs", inv_freqs)

    def forward(self, seq_len):
        seq = torch.arange(seq_len - 1, -1, -1.0)
        sinusoidal_inp = rearrange(seq, "n -> n ()") * rearrange(self.inv_freqs, "d -> () d")
        pos_emb = torch.cat((sinusoidal_inp.sin(), sinusoidal_inp.cos()), dim=-1)
        return pos_emb


class MultiHeadAttention(nn.Module):
    """Multi-head Self-Attention implementation per the formula Attention(Q,K,V) = softmax(QK^T/sqrt(d_k))V"""

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "Embedding dimension needs to be divisible by the number of heads"

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Scaling factor
        self.scale = self.head_dim ** -0.5

    def forward(self, values, keys, query, mask):
        batch_size = query.shape[0]
        
        # Linear projections and reshape
        q = self.q_proj(query).reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(keys).reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(values).reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention: QK^T/sqrt(d_k)
        attention = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            attention = attention.masked_fill(mask.unsqueeze(1).unsqueeze(1) == 0, float("-1e20"))
        
        # Apply softmax to get attention weights (these are the gate values)
        attention_weights = torch.softmax(attention, dim=-1)
        
        # Apply attention weights to values
        context = torch.matmul(attention_weights, v)
        context = context.transpose(1, 2).reshape(batch_size, -1, self.embed_dim)
        
        # Final projection
        output = self.out_proj(context)
        
        return output, attention_weights


class TransformerLayer(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.attention = MultiHeadAttention(dim, num_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Feed-Forward Neural Network (FFN)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        
    def forward(self, value, key, query, mask):
        # Step 1: Self-attention with pre-normalization
        query_norm = self.norm1(query)
        attn_output, attention_weights = self.attention(value, key, query_norm, mask)
        
        # Step 2: First residual connection
        x = query + attn_output
        
        # Step 3: Feed-Forward Network with pre-normalization
        x_norm = self.norm2(x)
        ffn_output = self.ffn(x_norm)
        
        # Step 4: Second residual connection
        output = x + ffn_output
        
        return output, attention_weights


class TransformerModule(nn.Module):
    """Long-term memory module implemented with Transformer layers"""
    def __init__(self, num_layers, dim, num_heads, max_episode_steps, positional_encoding):
        super().__init__()
        self.max_episode_steps = max_episode_steps
        self.positional_encoding = positional_encoding
        
        # Positional encoding
        if positional_encoding == "absolute":
            self.pos_embedding = PositionalEncoding(dim)
        elif positional_encoding == "learned":
            self.pos_embedding = nn.Parameter(torch.randn(max_episode_steps, dim))
        
        # Transformer layers
        self.layers = nn.ModuleList([TransformerLayer(dim, num_heads) for _ in range(num_layers)])
        
    def forward(self, x, memories, mask, memory_indices):
        # Apply positional encoding to memories
        if self.positional_encoding == "absolute":
            pos_embedding = self.pos_embedding(self.max_episode_steps)[memory_indices]
            memories = memories + pos_embedding.unsqueeze(2)
        elif self.positional_encoding == "learned":
            memories = memories + self.pos_embedding[memory_indices].unsqueeze(2)
        
        # Process through transformer layers
        transformer_memories = []
        for i, layer in enumerate(self.layers):
            transformer_memories.append(x.detach())
            
            # Handle shape for sequence input vs single vector
            if x.dim() == 1:
                x = x.unsqueeze(0)
                
            # Pass through transformer layer
            layer_output, _ = layer(
                memories[:, :, i], memories[:, :, i], x.unsqueeze(1), mask
            )
            x = layer_output.squeeze(1)
            
            # Ensure proper shape for single examples
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
                
        return x, torch.stack(transformer_memories, dim=1)


class LSTMModule(nn.Module):
    """Short-term memory module implemented with LSTM"""
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Optional projection to match transformer output dimension
        self.proj = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x, hidden=None):
        # Reshape x if it's a single vector
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
            
        # Get batch size from input
        batch_size = x.size(0)
        
        # Initialize hidden state if None
        if hidden is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device)
            hidden = (h0, c0)
        # If batch size doesn't match, reinitialize hidden state
        elif hidden[0].size(1) != batch_size:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device)
            hidden = (h0, c0)
            
        # Run LSTM
        output, (h_n, c_n) = self.lstm(x, hidden)
        
        # Get the output from the last timestep
        last_output = output[:, -1]
        
        # Project back to input dimension if needed
        projected_output = self.proj(last_output)
        
        return projected_output, (h_n, c_n)


class AdaptiveGatingFusion(nn.Module):
    """Adaptive Gating Fusion (AGF) to combine transformer and LSTM outputs"""
    def __init__(self, input_dim, min_gate_value=0.2):
        super().__init__()
        self.gate_network = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.Sigmoid()
        )
        self.min_gate_value = min_gate_value
        
    def forward(self, transformer_output, lstm_output):
        # Concatenate outputs for gate calculation
        combined = torch.cat([transformer_output, lstm_output], dim=-1)
        
        # Calculate gating weight g_t
        gate = self.gate_network(combined)
        
        # Apply minimum threshold to ensure transformer pathway remains active
        gate = torch.clamp(gate, min=self.min_gate_value)
        
        # Apply gating fusion: z_t = g_t ⊙ h_t^T + (1-g_t) ⊙ h_t^L
        fused = gate * transformer_output + (1 - gate) * lstm_output
        
        return fused, gate  # Return gate values for monitoring


class PALMER(nn.Module):
    """Parallel Attention LME with Residual Memory architecture"""
    def __init__(self, args, max_episode_steps):
        super().__init__()
        self.dim = args.trxl_dim
        
        # 1. Transformer module for long-term memory
        self.transformer = TransformerModule(
            args.trxl_num_layers,
            args.trxl_dim,
            args.trxl_num_heads,
            max_episode_steps,
            args.trxl_positional_encoding
        )
        
        # 2. LSTM module for short-term memory
        self.lstm = LSTMModule(
            input_dim=args.trxl_dim,
            hidden_dim=args.trxl_dim
        )
        
        # 3. Adaptive Gating Fusion
        self.fusion = AdaptiveGatingFusion(args.trxl_dim)
        
        # Initial hidden states for LSTM
        self.register_buffer("lstm_h0", torch.zeros(1, 1, args.trxl_dim))
        self.register_buffer("lstm_c0", torch.zeros(1, 1, args.trxl_dim))
        
        # Add layer normalization for both pathways
        self.transformer_norm = nn.LayerNorm(args.trxl_dim)
        self.lstm_norm = nn.LayerNorm(args.trxl_dim)
        
    def forward(self, x, memories, mask, memory_indices, lstm_hidden=None):
        # Process through transformer (long-term memory)
        transformer_output, transformer_memories = self.transformer(x, memories, mask, memory_indices)
        transformer_output = self.transformer_norm(transformer_output)
        
        # Process through LSTM (short-term memory)
        if lstm_hidden is None:
            batch_size = x.size(0) if x.dim() > 1 else 1
            h0 = self.lstm_h0.repeat(1, batch_size, 1)
            c0 = self.lstm_c0.repeat(1, batch_size, 1)
            lstm_hidden = (h0, c0)
            
        lstm_output, lstm_hidden = self.lstm(x, lstm_hidden)
        lstm_output = self.lstm_norm(lstm_output)
        
        # Fuse normalized outputs
        fused_output, gate_values = self.fusion(transformer_output, lstm_output)
        self.last_gate_values = gate_values  # Store for monitoring and regularization
        
        # Return four values to match the expected unpacking in get_action_and_value
        return fused_output, transformer_memories, lstm_hidden, gate_values


class Agent(nn.Module):
    def __init__(self, args, observation_space, action_space_shape, max_episode_steps):
        super().__init__()
        self.obs_shape = observation_space.shape
        self.max_episode_steps = max_episode_steps
        self.lstm_hidden = None

        # Observation encoding
        if len(self.obs_shape) > 1:
            self.encoder = nn.Sequential(
                layer_init(nn.Conv2d(3, 32, 8, stride=4)),
                nn.ReLU(),
                layer_init(nn.Conv2d(32, 64, 4, stride=2)),
                nn.ReLU(),
                layer_init(nn.Conv2d(64, 64, 3, stride=1)),
                nn.ReLU(),
                nn.Flatten(),
                layer_init(nn.Linear(64 * 7 * 7, args.trxl_dim)),
                nn.ReLU(),
            )
        else:
            self.encoder = layer_init(nn.Linear(observation_space.shape[0], args.trxl_dim))

        # PALMER architecture
        self.palmer = PALMER(args, max_episode_steps)

        # Post-processing after memory fusion
        self.hidden_post_fusion = nn.Sequential(
            layer_init(nn.Linear(args.trxl_dim, args.trxl_dim)),
            nn.ReLU(),
        )

        # Actor and critic heads
        self.actor_branches = nn.ModuleList(
            [
                layer_init(nn.Linear(args.trxl_dim, out_features=num_actions), np.sqrt(0.01))
                for num_actions in action_space_shape
            ]
        )
        self.critic = layer_init(nn.Linear(args.trxl_dim, 1), 1)

        # Optional reconstruction module
        if args.reconstruction_coef > 0.0:
            self.transposed_cnn = nn.Sequential(
                layer_init(nn.Linear(args.trxl_dim, 64 * 7 * 7)),
                nn.ReLU(),
                nn.Unflatten(1, (64, 7, 7)),
                layer_init(nn.ConvTranspose2d(64, 64, 3, stride=1)),
                nn.ReLU(),
                layer_init(nn.ConvTranspose2d(64, 32, 4, stride=2)),
                nn.ReLU(),
                layer_init(nn.ConvTranspose2d(32, 3, 8, stride=4)),
                nn.Sigmoid(),
            )

    def reset_lstm_state(self, batch_size=1):
        """Reset LSTM hidden state between episodes or when batch size changes"""
        self.lstm_hidden = None

    def reset_lstm_state_for_env(self, env_id=None):
        """Reset LSTM hidden state for a specific environment or all environments"""
        if env_id is None:
            self.lstm_hidden = None
        elif self.lstm_hidden is not None:
            # Only reset the specified environment's LSTM state
            # For hidden state (h)
            self.lstm_hidden[0][:, env_id, :] = 0
            # For cell state (c)
            self.lstm_hidden[1][:, env_id, :] = 0

    def get_value(self, x, memory, memory_mask, memory_indices):
        if len(self.obs_shape) > 1:
            x = self.encoder(x.permute((0, 3, 1, 2)) / 255.0)
        else:
            x = self.encoder(x)
        
        # Process through PALMER architecture
        x, _, self.lstm_hidden, _ = self.palmer(x, memory, memory_mask, memory_indices, self.lstm_hidden)
        
        x = self.hidden_post_fusion(x)
        return self.critic(x).flatten()

    def get_action_and_value(self, x, memory, memory_mask, memory_indices, action=None):
        if len(self.obs_shape) > 1:
            x = self.encoder(x.permute((0, 3, 1, 2)) / 255.0)
        else:
            x = self.encoder(x)
        
        # Process through PALMER architecture
        x, transformer_memories, self.lstm_hidden, gate_values = self.palmer(x, memory, memory_mask, memory_indices, self.lstm_hidden)
        
        # Store gate values for logging
        self.attention_weights = [gate_values]  # Store as a list for consistency with other parts of the code
        
        # Rest of the method remains unchanged
        x = self.hidden_post_fusion(x)
        self.x = x  # Store for reconstruction if needed
        
        # Actor (policy) branches
        probs = [Categorical(logits=branch(x)) for branch in self.actor_branches]
        if action is None:
            action = torch.stack([dist.sample() for dist in probs], dim=1)
            
        # Compute log probabilities and entropies
        log_probs = []
        for i, dist in enumerate(probs):
            log_probs.append(dist.log_prob(action[:, i]))
        entropies = torch.stack([dist.entropy() for dist in probs], dim=1).sum(1).reshape(-1)
        
        # Return transformer_memories as the fifth value (not gate_values)
        return action, torch.stack(log_probs, dim=1), entropies, self.critic(x).flatten(), transformer_memories

    def reconstruct_observation(self):
        x = self.transposed_cnn(self.x)
        return x.permute((0, 2, 3, 1))

    def get_mean_gate_values(self):
        """Calculate mean gate (attention) values across all layers and heads"""
        if not hasattr(self, 'attention_weights') or not self.attention_weights:
            return torch.tensor(0.0, device=self.device)
            
        # Calculate mean attention weight for each layer
        mean_gates = []
        for layer_weights in self.attention_weights:
            if layer_weights is not None:
                # Average across batch, heads, and sequence dimensions
                mean_gates.append(layer_weights.mean())
                
        if not mean_gates:
            return torch.tensor(0.0, device=self.device)
            
        # Return mean across all layers
        return torch.stack(mean_gates).mean()


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Determine the device to be used for training and set the default tensor type
    if args.cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(device)
    else:
        device = torch.device("cpu")

    # Environment setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    observation_space = envs.single_observation_space
    action_space_shape = (
        (envs.single_action_space.n,)
        if isinstance(envs.single_action_space, gym.spaces.Discrete)
        else tuple(envs.single_action_space.nvec)
    )
    env_ids = range(args.num_envs)
    env_current_episode_step = torch.zeros((args.num_envs,), dtype=torch.long)
    # Determine maximum episode steps
    max_episode_steps = envs.envs[0].spec.max_episode_steps
    if not max_episode_steps:
        envs.envs[0].reset()  # Memory Gym envs need to be reset before accessing max_episode_steps
        max_episode_steps = envs.envs[0].max_episode_steps
    if max_episode_steps <= 0:
        max_episode_steps = 1024  # Memory Gym envs have max_episode_steps set to -1
    # Set transformer memory length to max episode steps if greater than max episode steps
    args.trxl_memory_length = min(args.trxl_memory_length, max_episode_steps)

    agent = Agent(args, observation_space, action_space_shape, max_episode_steps).to(device)
    optimizer = optim.AdamW(agent.parameters(), lr=args.init_lr)
    bce_loss = nn.BCELoss()  # Binary cross entropy loss for observation reconstruction

    # ALGO Logic: Storage setup
    rewards = torch.zeros((args.num_steps, args.num_envs))
    actions = torch.zeros((args.num_steps, args.num_envs, len(action_space_shape)), dtype=torch.long)
    dones = torch.zeros((args.num_steps, args.num_envs))
    obs = torch.zeros((args.num_steps, args.num_envs) + observation_space.shape)
    log_probs = torch.zeros((args.num_steps, args.num_envs, len(action_space_shape)))
    values = torch.zeros((args.num_steps, args.num_envs))
    # The length of stored-memories is equal to the number of sampled episodes during training data sampling
    # (num_episodes, max_episode_length, num_layers, embed_dim)
    stored_memories = []
    # Memory mask used during attention
    stored_memory_masks = torch.zeros((args.num_steps, args.num_envs, args.trxl_memory_length), dtype=torch.bool)
    # Index to select the correct episode memory from stored_memories
    stored_memory_index = torch.zeros((args.num_steps, args.num_envs), dtype=torch.long)
    # Indices to slice the episode memories into windows
    stored_memory_indices = torch.zeros((args.num_steps, args.num_envs, args.trxl_memory_length), dtype=torch.long)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    episode_infos = deque(maxlen=100)  # Store episode results for monitoring statistics
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs)
    # Setup placeholders for each environments's current episodic memory
    next_memory = torch.zeros((args.num_envs, max_episode_steps, args.trxl_num_layers, args.trxl_dim), dtype=torch.float32)
    # Generate episodic memory mask used in attention
    memory_mask = torch.tril(torch.ones((args.trxl_memory_length, args.trxl_memory_length)), diagonal=-1)
    """ e.g. memory mask tensor looks like this if memory_length = 6
    0, 0, 0, 0, 0, 0
    1, 0, 0, 0, 0, 0
    1, 1, 0, 0, 0, 0
    1, 1, 1, 0, 0, 0
    1, 1, 1, 1, 0, 0
    1, 1, 1, 1, 1, 0
    """
    # Setup memory window indices to support a sliding window over the episodic memory
    repetitions = torch.repeat_interleave(
        torch.arange(0, args.trxl_memory_length).unsqueeze(0), args.trxl_memory_length - 1, dim=0
    ).long()
    memory_indices = torch.stack(
        [torch.arange(i, i + args.trxl_memory_length) for i in range(max_episode_steps - args.trxl_memory_length + 1)]
    ).long()
    memory_indices = torch.cat((repetitions, memory_indices))
    """ e.g. the memory window indices tensor looks like this if memory_length = 4 and max_episode_length = 7:
    0, 1, 2, 3
    0, 1, 2, 3
    0, 1, 2, 3
    0, 1, 2, 3
    1, 2, 3, 4
    2, 3, 4, 5
    3, 4, 5, 6
    """

    for iteration in range(1, args.num_iterations + 1):
        sampled_episode_infos = []

        # Annealing the learning rate and entropy coefficient if instructed to do so
        do_anneal = args.anneal_steps > 0 and global_step < args.anneal_steps
        frac = 1 - global_step / args.anneal_steps if do_anneal else 0
        lr = (args.init_lr - args.final_lr) * frac + args.final_lr
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        ent_coef = (args.init_ent_coef - args.final_ent_coef) * frac + args.final_ent_coef

        # Init episodic memory buffer using each environments' current episodic memory
        stored_memories = [next_memory[e] for e in range(args.num_envs)]
        for e in range(args.num_envs):
            stored_memory_index[:, e] = e

        for step in range(args.num_steps):
            global_step += args.num_envs

            # ALGO LOGIC: action logic
            with torch.no_grad():
                obs[step] = next_obs
                dones[step] = next_done
                stored_memory_masks[step] = memory_mask[torch.clip(env_current_episode_step, 0, args.trxl_memory_length - 1)]
                stored_memory_indices[step] = memory_indices[env_current_episode_step]
                # Retrieve the memory window from the entire episodic memory
                memory_window = batched_index_select(next_memory, 1, stored_memory_indices[step])
                action, logprob, entropy, value, transformer_memories = agent.get_action_and_value(
                    next_obs, memory_window, stored_memory_masks[step], stored_memory_indices[step]
                )
                
                # Store the transformer memories in the next_memory
                for env_idx, env_step in enumerate(env_current_episode_step):
                    next_memory[env_idx, env_step] = transformer_memories[env_idx]
                
                # Store the action, log_prob, and value in the buffer
                actions[step], log_probs[step], values[step] = action, logprob, value

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            # Reset and process episodic memory if done
            for id, done in enumerate(next_done):
                if done:
                    # Reset the environment's current timestep
                    env_current_episode_step[id] = 0
                    # Break the reference to the environment's episodic memory
                    mem_index = stored_memory_index[step, id]
                    stored_memories[mem_index] = stored_memories[mem_index].clone()
                    # Reset episodic memory
                    next_memory[id] = torch.zeros(
                        (max_episode_steps, args.trxl_num_layers, args.trxl_dim), dtype=torch.float32
                    )
                    # Explicitly reset LSTM hidden state per environment when episode ends
                    agent.reset_lstm_state_for_env(id)
                    if step < args.num_steps - 1:
                        # Store memory inside the buffer
                        stored_memories.append(next_memory[id])
                        # Store the reference of to the current episodic memory inside the buffer
                        stored_memory_index[step + 1 :, id] = len(stored_memories) - 1
                else:
                    # Increment environment timestep if not done
                    env_current_episode_step[id] += 1

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        sampled_episode_infos.append(info["episode"])

        # Bootstrap value if not done
        with torch.no_grad():
            start = torch.clip(env_current_episode_step - args.trxl_memory_length, 0)
            end = torch.clip(env_current_episode_step, args.trxl_memory_length)
            indices = torch.stack([torch.arange(start[b], end[b]) for b in range(args.num_envs)]).long()
            memory_window = batched_index_select(next_memory, 1, indices)  # Retrieve the memory window from the entire episode
            next_value = agent.get_value(
                next_obs,
                memory_window,
                memory_mask[torch.clip(env_current_episode_step, 0, args.trxl_memory_length - 1)],
                stored_memory_indices[-1],
            )
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # Flatten the batch
        b_obs = obs.reshape(-1, *obs.shape[2:])
        b_logprobs = log_probs.reshape(-1, *log_probs.shape[2:])
        b_actions = actions.reshape(-1, *actions.shape[2:])
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_memory_index = stored_memory_index.reshape(-1)
        b_memory_indices = stored_memory_indices.reshape(-1, *stored_memory_indices.shape[2:])
        b_memory_mask = stored_memory_masks.reshape(-1, *stored_memory_masks.shape[2:])
        stored_memories = torch.stack(stored_memories, dim=0)

        # Remove unnecessary padding from TrXL memory, if applicable
        actual_max_episode_steps = (stored_memory_indices * stored_memory_masks).max().item() + 1
        if actual_max_episode_steps < args.trxl_memory_length:
            b_memory_indices = b_memory_indices[:, :actual_max_episode_steps]
            b_memory_mask = b_memory_mask[:, :actual_max_episode_steps]
            stored_memories = stored_memories[:, :actual_max_episode_steps]

        # Optimizing the policy and value network
        clipfracs = []
        for epoch in range(args.update_epochs):
            # Reset LSTM hidden states at the beginning of each epoch
            agent.reset_lstm_state()
            
            b_inds = torch.randperm(args.batch_size)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                
                # Reset LSTM hidden states for each minibatch with correct batch size
                agent.reset_lstm_state()
                
                mb_memories = stored_memories[b_memory_index[mb_inds]]
                mb_memory_windows = batched_index_select(mb_memories, 1, b_memory_indices[mb_inds])

                _, newlogprob, entropy, newvalue, transformer_memories = agent.get_action_and_value(
                    b_obs[mb_inds], mb_memory_windows, b_memory_mask[mb_inds], b_memory_indices[mb_inds], b_actions[mb_inds]
                )

                # Policy loss
                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                mb_advantages = mb_advantages.unsqueeze(1).repeat(
                    1, len(action_space_shape)
                )  # Repeat is necessary for multi-discrete action spaces
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = torch.exp(logratio)
                pgloss1 = -mb_advantages * ratio
                pgloss2 = -mb_advantages * torch.clamp(ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef)
                pg_loss = torch.max(pgloss1, pgloss2).mean()

                # Value loss
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                if args.clip_vloss:
                    v_loss_clipped = b_values[mb_inds] + (newvalue - b_values[mb_inds]).clamp(
                        min=-args.clip_coef, max=args.clip_coef
                    )
                    v_loss = torch.max(v_loss_unclipped, (v_loss_clipped - b_returns[mb_inds]) ** 2).mean()
                else:
                    v_loss = v_loss_unclipped.mean()

                # Entropy loss
                entropy_loss = entropy.mean()

                # Combined losses
                loss = pg_loss - ent_coef * entropy_loss + v_loss * args.vf_coef

                # Add reconstruction loss if used
                r_loss = torch.tensor(0.0)
                if args.reconstruction_coef > 0.0:
                    r_loss = bce_loss(agent.reconstruct_observation(), b_obs[mb_inds] / 255.0)
                    loss += args.reconstruction_coef * r_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=args.max_grad_norm)
                optimizer.step()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Log and monitor training statistics
        episode_infos.extend(sampled_episode_infos)
        episode_result = {}
        if len(episode_infos) > 0:
            for key in episode_infos[0].keys():
                episode_result[key + "_mean"] = np.mean([info[key] for info in episode_infos])
            
            # Calculate metrics for evals folder
            mean_reward = episode_result.get("r_mean", 0.0)
            mean_length = episode_result.get("l_mean", 0.0)
            # Check if success info is available in episode_infos
            if "success" in episode_infos[0]:
                mean_success_rate = np.mean([info["success"] for info in episode_infos])
            else:
                # Try to infer success from episode rewards if possible
                # Assuming success is 1 if reward is above a certain threshold, customize as needed
                mean_success_rate = 0.0
                if "r" in episode_infos[0]:
                    mean_success_rate = np.mean([float(info["r"] > 0) for info in episode_infos])

        print(
            "{:9} SPS={:4} return={:.2f} length={:.1f} pi_loss={:.3f} v_loss={:.3f} entropy={:.3f} r_loss={:.3f} value={:.3f} adv={:.3f}".format(
                iteration,
                int(global_step / (time.time() - start_time)),
                episode_result["r_mean"],
                episode_result["l_mean"],
                pg_loss.item(),
                v_loss.item(),
                entropy_loss.item(),
                r_loss.item(),
                torch.mean(values),
                torch.mean(advantages),
            )
        )

        if episode_result:
            for key in episode_result:
                writer.add_scalar("episode/" + key, episode_result[key], global_step)
            
            # Add metrics to 'evals' folder with proper definitions
            writer.add_scalar("evals/reward", mean_reward, global_step)
            writer.add_scalar("evals/episode_length", mean_length, global_step)
            writer.add_scalar("evals/success_rate", mean_success_rate, global_step)
            
        writer.add_scalar("episode/value_mean", torch.mean(values), global_step)
        writer.add_scalar("episode/advantage_mean", torch.mean(advantages), global_step)
        writer.add_scalar("charts/learning_rate", lr, global_step)
        writer.add_scalar("charts/entropy_coefficient", ent_coef, global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/loss", loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/reconstruction_loss", r_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        # Add the gate value logging to the tensorboard section
        mean_gate_value = agent.get_mean_gate_values()
        writer.add_scalar("gates/mean_attention", mean_gate_value.item(), global_step)

        # For more detailed gate analysis, you can also log per-layer gate values
        if hasattr(agent, 'attention_weights') and agent.attention_weights:
            for i, layer_weights in enumerate(agent.attention_weights):
                if layer_weights is not None:
                    writer.add_scalar(f"gates/layer_{i}_attention", layer_weights.mean().item(), global_step)

        # Add more detailed gate value monitoring
        if hasattr(agent.palmer, 'last_gate_values'):
            writer.add_scalar("gates/min_gate_value", agent.palmer.last_gate_values.min().item(), global_step)
            writer.add_scalar("gates/max_gate_value", agent.palmer.last_gate_values.max().item(), global_step)
            writer.add_scalar("gates/std_gate_value", agent.palmer.last_gate_values.std().item(), global_step)
            writer.add_scalar("gates/agf_value", agent.palmer.last_gate_values.mean().item(), global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        model_data = {
            "model_weights": agent.state_dict(),
            "args": vars(args),
        }
        torch.save(model_data, model_path)
        print(f"model saved to {model_path}")

    writer.close()
    envs.close()