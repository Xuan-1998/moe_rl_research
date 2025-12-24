import torch
import numpy as np
import os
import glob
from sa_placement import simulated_annealing
from placement_env import ExpertPlacementEnv

def main():
    data_dir = "data/benchmarks"
    files = glob.glob(os.path.join(data_dir, "*.pt"))
    
    demonstrations = []
    
    print(f"Generating Demonstrations from {len(files)} datasets...")
    
    for f in files:
        print(f"Processing {f}...")
        try:
            # Load Data
            data = torch.load(f, map_location='cpu')
            logits = data.get('logits').squeeze(0)
            _, indices = torch.topk(logits, 6, dim=-1)
            
            # Compute P
            N, K = indices.shape
            one_hot = torch.zeros((N, 64), device=indices.device)
            one_hot.scatter_(1, indices, 1.0)
            P = torch.matmul(one_hot.T, one_hot)
            P.fill_diagonal_(0)
            P_np = P.numpy()
            
            # Run SA to get "Expert" Placement
            # Run longer for better quality
            sa_placement = simulated_annealing(P_np, indices, steps=10000)
            
            # Create "Observations" and "Actions" for BC
            # We want to train an agent that takes Global Obs and outputs Placement?
            # Or an iterative agent?
            # Standard RL Env (ExpertPlacementEnv) is Iterative (Swap).
            # Constructive Env is Sequential.
            # Constructive is better for Zero-Shot generation.
            # Let's use Constructive Logic:
            # At step t, Agent sees Partial Placement -> Outputs Device.
            
            # Reconstruct the constructive sequence from the Final SA Placement.
            # SA gives final map: Expert i -> Device d.
            # We can sort experts by degree (as in ConstructiveEnv).
            degrees = P_np.sum(axis=1)
            sorted_indices = np.argsort(degrees)[::-1]
            
            # Re-map P and Placement to sorted order
            P_sorted = P_np[sorted_indices][:, sorted_indices]
            placement_sorted = sa_placement[sorted_indices]
            
            # Generate Trajectory
            num_experts = 64
            num_devices = 8
            
            node_to_device = np.full(num_experts, -1, dtype=int)
            device_fill = np.zeros(num_devices, dtype=int)
            capacity = num_experts // num_devices
            
            for t in range(num_experts):
                # Construct Observation at step t
                # Obs: [Affinity[num_devices], Fill[num_devices]]
                
                weights = P_sorted[t]
                affinities = np.zeros(num_devices, dtype=np.float32)
                
                # Check neighbors already placed (0 to t-1)
                for placed_idx in range(t):
                   dev = node_to_device[placed_idx]
                   # Weight between current t and placed_idx
                   w = weights[placed_idx]
                   if w > 0:
                       affinities[dev] += w
                       
                # Normalize
                max_aff = np.max(affinities) if np.max(affinities) > 0 else 1.0
                affinities /= max_aff
                
                fills = device_fill / capacity
                
                obs = np.concatenate([affinities, fills]).astype(np.float32)
                
                # Target Action
                target_action = placement_sorted[t]
                
                # Verify validity (SA might technically violate exact balance if we were lax, but SA script enforces strict balance?)
                # SA script in `sa_placement.py` initializes strict balanced and swaps. So it is balanced.
                
                # Store
                demonstrations.append((obs, target_action))
                
                # Update State
                node_to_device[t] = target_action
                device_fill[target_action] += 1
                
        except Exception as e:
            print(f"Skipping {f}: {e}")
            
    # Save Demos
    print(f"Generated {len(demonstrations)} state-action pairs.")
    torch.save(demonstrations, "data/benchmarks/bc_demonstrations.pt")

if __name__ == "__main__":
    main()
