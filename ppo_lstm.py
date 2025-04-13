import os
import random
import time
import math
from collections import deque
from dataclasses import dataclass

import gymnasium as gym
import memory_gym  # noqa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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

    # LSTM specific arguments (replacing Transformer-XL args)
    lstm_num_layers: int = 3
    """the number of LSTM layers"""
    lstm_hidden_dim: int = 384
    """the dimension of LSTM hidden state"""
    lstm_sequence_length: int = 128
    """the sequence length to maintain for LSTM"""
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


class LSTMNetwork(nn.Module):
    """LSTM-based architecture for RL agents"""
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Initialize LSTM weights using orthogonal initialization
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param, gain=1.0)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)
                
    def forward(self, x, hidden_states=None):
        # Ensure input shape is [batch_size, seq_len=1, input_dim]
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(1)  # Add batch and seq dimensions
        elif x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
            
        # Get current batch size
        batch_size = x.size(0)
        
        # Check if hidden_states batch dimension matches input batch size
        if hidden_states is not None and hidden_states[0].size(1) != batch_size:
            # Initialize new hidden states with correct batch size
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device)
            hidden_states = (h0, c0)
            
        # Run LSTM forward pass
        if hidden_states is not None:
            self.lstm.flatten_parameters()
            lstm_out, hidden = self.lstm(x, hidden_states)
        else:
            self.lstm.flatten_parameters()
            lstm_out, hidden = self.lstm(x)
        
        # Extract output (last timestep)
        output = lstm_out[:, -1]
        
        return output, hidden
    
    def init_hidden(self, batch_size, device):
        # Initialize hidden states (h0, c0) with zeros
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return (h0, c0)


class Agent(nn.Module):
    def __init__(self, args, observation_space, action_space_shape, device):
        super().__init__()
        self.obs_shape = observation_space.shape
        self.device = device
        
        # Initialize hidden states to None
        self.hidden_states = None
        
        # Observation encoder
        if len(self.obs_shape) > 1:
            self.encoder = nn.Sequential(
                layer_init(nn.Conv2d(3, 32, 8, stride=4)),
                nn.ReLU(),
                layer_init(nn.Conv2d(32, 64, 4, stride=2)),
                nn.ReLU(),
                layer_init(nn.Conv2d(64, 64, 3, stride=1)),
                nn.ReLU(),
                nn.Flatten(),
                layer_init(nn.Linear(64 * 7 * 7, args.lstm_hidden_dim)),
                nn.ReLU(),
            )
        else:
            self.encoder = layer_init(nn.Linear(observation_space.shape[0], args.lstm_hidden_dim))
        
        # LSTM network
        self.lstm = LSTMNetwork(
            input_dim=args.lstm_hidden_dim,
            hidden_dim=args.lstm_hidden_dim,
            num_layers=args.lstm_num_layers
        )
        
        # Post-processing layer
        self.hidden_post_lstm = nn.Sequential(
            layer_init(nn.Linear(args.lstm_hidden_dim, args.lstm_hidden_dim)),
            nn.ReLU(),
        )
        
        # Actor and critic heads
        self.actor_branches = nn.ModuleList(
            [
                layer_init(nn.Linear(args.lstm_hidden_dim, out_features=num_actions), np.sqrt(0.01))
                for num_actions in action_space_shape
            ]
        )
        self.critic = layer_init(nn.Linear(args.lstm_hidden_dim, 1), 1)
        
        # Optional reconstruction module
        if args.reconstruction_coef > 0.0:
            self.transposed_cnn = nn.Sequential(
                layer_init(nn.Linear(args.lstm_hidden_dim, 64 * 7 * 7)),
                nn.ReLU(),
                nn.Unflatten(1, (64, 7, 7)),
                layer_init(nn.ConvTranspose2d(64, 64, 3, stride=1)),
                nn.ReLU(),
                layer_init(nn.ConvTranspose2d(64, 32, 4, stride=2)),
                nn.ReLU(),
                layer_init(nn.ConvTranspose2d(32, 3, 8, stride=4)),
                nn.Sigmoid(),
            )
    
    def reset_hidden(self, batch_size=1):
        """Reset LSTM hidden states between episodes or when batch size changes"""
        self.hidden_states = self.lstm.init_hidden(batch_size, self.device)
    
    def get_value(self, x):
        # Encode observation
        if len(self.obs_shape) > 1:
            x = self.encoder(x.permute((0, 3, 1, 2)) / 255.0)
        else:
            x = self.encoder(x)
        
        # Initialize hidden states if needed
        batch_size = x.size(0) if x.dim() > 1 else 1
        if self.hidden_states is None:
            self.hidden_states = self.lstm.init_hidden(batch_size, self.device)
        
        # Process through LSTM (without updating hidden states for value estimation)
        with torch.no_grad():
            x, _ = self.lstm(x, self.hidden_states)
        
        x = self.hidden_post_lstm(x)
        
        return self.critic(x).flatten()

    def get_action_and_value(self, x, action=None):
        # Encode observation
        if len(self.obs_shape) > 1:
            x = self.encoder(x.permute((0, 3, 1, 2)) / 255.0)
        else:
            x = self.encoder(x)
        
        # Initialize hidden states if needed
        batch_size = x.size(0) if x.dim() > 1 else 1
        if self.hidden_states is None:
            self.hidden_states = self.lstm.init_hidden(batch_size, self.device)
        
        # Process through LSTM
        x, self.hidden_states = self.lstm(x, self.hidden_states)
        x = self.hidden_post_lstm(x)
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
        
        return action, torch.stack(log_probs, dim=1), entropies, self.critic(x).flatten()

    def reconstruct_observation(self):
        if not hasattr(self, 'transposed_cnn'):
            raise AttributeError("Reconstruction module not initialized (reconstruction_coef=0.0)")
        x = self.transposed_cnn(self.x)
        return x.permute((0, 2, 3, 1))


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

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete) or isinstance(
        envs.single_action_space, gym.spaces.MultiDiscrete
    ), "only discrete action space is supported"

    if isinstance(envs.single_action_space, gym.spaces.Discrete):
        action_space_shape = [envs.single_action_space.n]
    else:  # MultiDiscrete
        action_space_shape = envs.single_action_space.nvec

    max_episode_steps = envs.single_observation_space.shape[0] if len(envs.single_observation_space.shape) == 1 else 96

    # agent setup
    agent = Agent(args, envs.single_observation_space, action_space_shape, device).to(device)
    optimizer = optim.Adam(
        list(filter(lambda p: p.requires_grad, agent.parameters())), lr=args.init_lr, eps=1e-5
    )
    
    # Binary cross-entropy loss for reconstruction
    bce_loss = nn.BCELoss()

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + (len(action_space_shape),)).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs) + (len(action_space_shape),)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    
    # Initialize agent's hidden states
    agent.reset_hidden(args.num_envs)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so
        if args.anneal_steps > 0:
            frac = 1.0 - (global_step - 1.0) / args.anneal_steps
            if frac <= 0:
                frac = 0
            lr = frac * (args.init_lr - args.final_lr) + args.final_lr
            ent_coef = frac * (args.init_ent_coef - args.final_ent_coef) + args.final_ent_coef
            optimizer.param_groups[0]["lr"] = lr
        else:
            lr = args.init_lr
            ent_coef = args.init_ent_coef

        episode_infos = []
        sampled_episode_infos = []

        # Collect rollout
        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, entropy, value = agent.get_action_and_value(next_obs)
                values[step] = value
                
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = torch.Tensor(np.logical_or(terminations, truncations)).to(device)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.Tensor(next_obs).to(device)

            # Reset LSTM hidden states for terminated episodes
            if next_done.any():
                # Create a mask of environments where episodes terminated
                term_envs = next_done.nonzero(as_tuple=True)[0]
                
                # Process terminated episodes' info
                if "final_info" in infos:
                    for idx in term_envs:
                        if idx < len(infos["final_info"]) and infos["final_info"][idx] is not None:
                            if "episode" in infos["final_info"][idx]:
                                sampled_episode_infos.append(infos["final_info"][idx]["episode"])
                
                # Only reset hidden states for terminated environments
                # This is crucial for proper credit assignment across episodes
                for idx in term_envs:
                    # Get current hidden states
                    h, c = agent.hidden_states
                    
                    # Zero out the hidden states for terminated envs
                    h[:, idx, :] = torch.zeros_like(h[:, idx, :])
                    c[:, idx, :] = torch.zeros_like(c[:, idx, :])
                    
                    # Update hidden states
                    agent.hidden_states = (h, c)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs)
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

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1, len(action_space_shape))
        b_actions = actions.reshape((-1, len(action_space_shape)))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        clipfracs = []
        
        for epoch in range(args.update_epochs):
            # Reset hidden states at the beginning of each optimization epoch
            agent.reset_hidden(args.minibatch_size)
            
            b_inds = torch.randperm(args.batch_size)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )

                # Detach hidden states to break computational graph between minibatches
                if agent.hidden_states is not None:
                    h, c = agent.hidden_states
                    agent.hidden_states = (h.detach(), c.detach())

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
                r_loss = torch.tensor(0.0, device=device)
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
                mean_success_rate = 0.0
                if "r" in episode_infos[0]:
                    mean_success_rate = np.mean([float(info["r"] > 0) for info in episode_infos])

        # Safe printing with default values if keys don't exist
        print(
            "{:9} SPS={:4} return={:.2f} length={:.1f} pi_loss={:.3f} v_loss={:.3f} entropy={:.3f} r_loss={:.3f} value={:.3f} adv={:.3f}".format(
                iteration,
                int(global_step / (time.time() - start_time)),
                episode_result.get("r_mean", 0.0),  # Use .get() with default value
                episode_result.get("l_mean", 0.0),  # Use .get() with default value
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