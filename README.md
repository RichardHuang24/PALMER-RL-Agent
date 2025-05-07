# PALMER: Parallel Attentive LSTM-Transformer Architecture for RL

This repository contains the implementation and report for **PALMER**, a novel memory-augmented reinforcement learning architecture that combines LSTM and Transformer modules to handle partial observability in sequential decision-making tasks.

> 📄 Project done for the COMP0258 module: Open-Endedness and General Intelligence, UCL MSc Computational Statistics and Machine Learning.

---

## What We Did

We designed, implemented, and evaluated **PALMER**, a hybrid memory model for deep reinforcement learning. PALMER is developed to address the limitations of existing memory-based RL agents (like PPO-LSTM and PPO-TrXL) in tasks requiring both short-term and long-term memory.

We integrated PALMER into the [CleanRL](https://github.com/vwxyzjn/cleanrl) framework and evaluated it on four memory-intensive environments from the MiniGrid suite:
- `MiniGrid-MemoryS11-v0`
- `MiniGrid-MemoryS13-v0`
- `MiniGrid-MemoryS13Random-v0`
- `MiniGrid-MemoryS17Random-v0`

---

## PALMER Architecture

PALMER is composed of:

- **Observation Encoder**: CNN or MLP depending on the input type (image or vector).
- **Parallel Memory Modules**:
  - **LSTM**: Captures short-term temporal dependencies.
  - **Transformer**: Captures long-term associative memory through self-attention.
- **Adaptive Gating Mechanism**: Learns to blend the LSTM and Transformer outputs into a fused memory vector at each timestep, allowing dynamic memory routing.
- **Policy and Value Heads**: Receive fused memory representation and are trained using PPO.

PALMER is optimized with the PPO algorithm using Generalized Advantage Estimation (GAE), entropy regularization, and clipped policy loss for training stability.

---

##  Code and Implementation

PALMER is built on top of the high-quality RL codebase [CleanRL](https://github.com/vwxyzjn/cleanrl). 
---

## Files

- `COMP0258_Report.pdf`: Full academic report with background, architecture, experiments, and analysis.
- `palmer_policy.py`: Core model implementation.
- `ppo_palmer.py`: PPO training script using the PALMER architecture.
- `colab_demo.ipynb`: Interactive agent demos and visualizations.

---
