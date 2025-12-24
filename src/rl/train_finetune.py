import torch
import torch.nn as nn
from stable_baselines3 import PPO
from constructive_env import ConstructivePlacementEnv
from sa_placement import simulated_annealing
import numpy as np

# BC Network Definition (for loading)
class PolicyNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, act_dim)
        )
        
    def forward(self, x):
        return self.net(x)

def load_data(path):
    data = torch.load(path, map_location='cpu')
    logits = data.get('logits').squeeze(0)
    _, indices = torch.topk(logits, 6, dim=-1)
    N, K = indices.shape
    one_hot = torch.zeros((N, 64), device=indices.device)
    one_hot.scatter_(1, indices, 1.0)
    P = torch.matmul(one_hot.T, one_hot)
    P.fill_diagonal_(0)
    return P.numpy()

def main():
    print("--- RL Fine-Tuning from BC Initialization ---")
    data_path = "data/benchmarks/winogrande_winogrande_xl.pt"
    P = load_data(data_path)
    
    env = ConstructivePlacementEnv(P, num_devices=8)
    
    # Init PPO (Standard)
    # architecture: [64, 64]
    policy_kwargs = dict(net_arch=dict(pi=[64, 64], vf=[64, 64]))
    model = PPO("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs, learning_rate=1e-4) # Low LR for fine-tuning
    
    # Load BC Weights
    print("Loading BC Weights into PPO Policy...")
    try:
        bc_policy = PolicyNet(env.observation_space.shape[0], env.action_space.n)
        bc_policy.load_state_dict(torch.load("bc_policy.pth", weights_only=False))
        
        # PPO Policy Structure:
        # model.policy.mlp_extractor.policy_net (Shared/Pi layers)
        # model.policy.action_net (Output layer)
        
        # My BC Net:
        # 0: Linear(obs, 64)
        # 2: Linear(64, 64)
        # 4: Linear(64, act)
        
        # Transfer weights
        with torch.no_grad():
            # Layer 0 -> policy_net[0]
            model.policy.mlp_extractor.policy_net[0].weight.copy_(bc_policy.net[0].weight)
            model.policy.mlp_extractor.policy_net[0].bias.copy_(bc_policy.net[0].bias)
            
            # Layer 2 -> policy_net[2]
            model.policy.mlp_extractor.policy_net[2].weight.copy_(bc_policy.net[2].weight)
            model.policy.mlp_extractor.policy_net[2].bias.copy_(bc_policy.net[2].bias)
            
            # Layer 4 -> action_net
            model.policy.action_net.weight.copy_(bc_policy.net[4].weight)
            model.policy.action_net.bias.copy_(bc_policy.net[4].bias)
            
        print("Weights Transferred Successfully.")
        
    except Exception as e:
        print(f"Warning: Failed to transfer weights: {e}")
        # Proceed with random init if fails (user will just get standard RL)
        
    # Train
    print("Starting Fine-tuning (100k steps)...")
    model.learn(total_timesteps=100000)
    
    model.save("ppo_finetuned")
    
    # Quick Check
    obs, _ = env.reset()
    for _ in range(64):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(action)
        if done:
            print(f"Final Cost after Fine-tuning: {info['final_cost']:.2f}")

if __name__ == "__main__":
    main()
