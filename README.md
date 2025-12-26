# DeepSeek-MoE: RL Expert Placement

This repository implements a Reinforcement Learning (RL) strategies to optimize the placement of Mixture-of-Experts (MoE) experts across distributed devices.

Traditional methods like **Simulated Annealing (SA)** are effective but computationally expensive (high latency) and struggle with changing traffic patterns. This project introduces two RL strategies that surpass standard approaches:

1.  **Offline Imitation + Fine-Tuning**: A Neural Policy that matches SA's high quality (>28% Communication Reduction) but runs **100x faster** (<1ms inference).
2.  **Dynamic Lifelong Learning**: An Adaptive Agent that intelligently balances Communication Cost vs. Migration Cost in real-time shifting environments.

## Performance Benchmarks

| Method | Static Cost (Lower is Better) | Migration Cost (Dynamic) | Inference Latency |
| :--- | :--- | :--- | :--- |
| **Naive Placement** | ~60,500 (Baseline) | N/A | Instant |
| **Simulated Annealing** | **~43,400 (Optimal)** | Very High (Unstable) | ~100ms (Slow) |
| **RL (Finetuned)** | **~43,500 (Matches SA)** | N/A | **<1ms (Instant)** |
| **RL (Dynamic)** | ~59,000 (Stable) | **Negligible (~0)** | **<1ms (Instant)** |

> **Key Result**: Our *Finetuned RL Agent* achieves the same optimal placement as Simulated Annealing but is suitable for real-time serving paths.

## Installation

```bash
pip install -r requirements.txt
# Requirements: torch, numpy, stable-baselines3, gymnasium
```

## Usage

### 1. Data Preparation
Extract traffic patterns (logits) from DeepSeek MoE:
```bash
python src/prepare_benchmarks.py
```

### 2. Strategy A: "Awesome Offline RL" (Static Optimization)
This strategy trains a neural network to mimic SA and then fine-tunes it to perfection.
```bash
# Step 1: Generate expert demonstrations using SA (Time: ~5 mins)
python src/rl/generate_demos.py

# Step 2: Offline Imitation Learning (Time: ~2 mins)
python src/rl/train_bc.py

# Step 3: Online Fine-Tuning (Time: ~10 mins)
python src/rl/train_finetune.py
```
**Outcome**: A `ppo_finetuned` agent that partitions graphs instantly with optimal cost.

### 3. Strategy B: "Dynamic Adaptation" (Lifelong Learning)
This strategy trains an agent to handle shifting traffic (e.g., Coding -> Reasoning tasks) without "stopping the world".
```bash
# Train the Dynamic Agent
python src/rl/train_dynamic.py

# Benchmark the Dynamic Scenario
python src/rl/benchmark_dynamic.py
```
**Outcome**: An agent that maintains low latency and system stability (near-zero migration overhead).

## Methodology

*   **Environment**: We utilize a constructive placement environment (`src/rl/constructive_env.py`) that places experts sequentially based on traffic density ("Heavy Hitters First").
*   **Imitation Learning**: We treat Simulated Annealing's output as "Expert Trajectories" and clone them using Supervised Learning.
*   **PPO Fine-Tuning**: We initialize the RL policy with the cloned weights, allowing PPO to explore immediately from a high-reward region, avoiding the cold-start problem common in Combinatorial Optimization.

---
*Created by [Xuan-1998]*
