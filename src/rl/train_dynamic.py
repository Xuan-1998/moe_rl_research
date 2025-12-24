import torch
from stable_baselines3 import PPO
from dynamic_env import DynamicExpertPlacementEnv
import matplotlib.pyplot as plt
import numpy as np

def main():
    data_dir = "data/benchmarks"
    
    # Create Dynamic Env
    # Penalize migration heavily to force agent to be smart (only move if worth it)
    env = DynamicExpertPlacementEnv(data_dir, num_devices=8, max_steps=2000, migration_penalty=50.0)
    
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003)
    
    print("Training Dynamic RL Agent...")
    # Train enough to see patterns
    model.learn(total_timesteps=200000)
    model.save("ppo_dynamic_agent")
    
    # Evaluate Adaptation
    print("Evaluating Adaptation...")
    obs, _ = env.reset()
    
    comm_costs = []
    migr_costs = []
    total_costs = []
    
    current_pattern = env.current_pattern_idx
    
    for i in range(500):
        # Deterministic prediction
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        
        comm_costs.append(info['comm_cost'])
        migr_costs.append(info['migr_cost'])
        total_costs.append(-reward * 1000) # Unscale
        
        if info['msg'] == "TRAFFIC_SHIFT":
            print(f"Step {i}: Traffic Shift Detected! New Profile.")
        
        if info['migr_cost'] > 0:
            print(f"Step {i}: Agent performed MIGRATION. Cost: {info['migr_cost']}")
            
        if done: break
        
    avg_comm = np.mean(comm_costs)
    avg_migr = np.mean(migr_costs)
    print(f"Average Communication Cost: {avg_comm:.2f}")
    print(f"Average Migration Cost: {avg_migr:.2f}")
    
    # Validate: Agent should keep Migration Cost LOW but Comm Cost LOW.
    # If Migration is 0, it learned nothing (Static).
    # If Comm is High, it fails.
    
if __name__ == "__main__":
    main()
