import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import glob
import os

class DynamicExpertPlacementEnv(gym.Env):
    """
    Dynamic Environment where Expert Traffic patterns shift over time.
    The agent must balance Minimizing Communication Cost vs. Minimizing Migration Cost.
    
    State:
        - Current Placement (Device IDs) [N]
        - Current Traffic Affinity (Affinity to Devices) [N x D]
        - (Implicitly: The shift in traffic is the challenge)
        
    Action:
        - Select Expert A (Source)
        - Select Target Device K
        - (Environment performs Greedy Swap on Target)
        - OR: No-Op Action (Stay put) -> Important for minimizing migration!
        
    Reward:
        -  - (Alpha * Communication_Cost + Beta * Migration_Cost)
        - We simplify: Reward = Improvement_in_Total_Cost
        - Total_Cost = Comm_Cost + (Migr_Cost if swap happens)
    """
    def __init__(self, data_dir, num_devices=8, max_steps=1000, migration_penalty=10.0):
        super().__init__()
        self.num_devices = num_devices
        self.max_steps = max_steps
        self.migration_penalty = migration_penalty # Cost of moving ONE expert
        
        # Load Datasets to simulate Traffic Shifts
        self.traffic_patterns = self._load_traffic_patterns(data_dir)
        self.num_patterns = len(self.traffic_patterns)
        if self.num_patterns == 0:
            raise ValueError(f"No .pt files found in {data_dir}")
            
        self.num_experts = self.traffic_patterns[0].shape[0]
        
        # Current State
        self.current_pattern_idx = 0
        self.P = self.traffic_patterns[0]
        
        # Initial placement: Balanced
        self.experts_per_device = self.num_experts // self.num_devices
        self.initial_placement = np.repeat(np.arange(num_devices), self.experts_per_device)
        self.placement = self.initial_placement.copy()
        
        # Observation Space
        # [DeviceIDs (N) + Affinity (N*D)]
        obs_dim = self.num_experts * (1 + self.num_devices)
        self.observation_space = spaces.Box(
            low=-1.0, high=1e6, shape=(obs_dim,), dtype=np.float32
        )
        
        # Action Space: [Expert (64), TargetDevice (8)]
        # Add a "No-Op" action? 
        # Actually, if Agent chooses (Exp=i, Dev=CurrentDev), that IS No-Op.
        self.action_space = spaces.MultiDiscrete([self.num_experts, self.num_devices])
        
        self.current_step = 0
        
        # Shift dynamics
        self.steps_until_shift = 100 # Change traffic every 100 steps
        self.shift_timer = self.steps_until_shift

    def _load_traffic_patterns(self, data_dir):
        files = glob.glob(os.path.join(data_dir, "*.pt"))
        patterns = []
        print(f"Loading {len(files)} traffic patterns for Dynamic Env...")
        for f in files:
            try:
                # Reuse logic from prepare/train
                data = torch.load(f, map_location='cpu')
                logits = data.get('logits').squeeze(0) # [Seq, Exp]
                
                # Check dim
                if logits.shape[1] != 64:
                    continue # Skip mismatches if any
                    
                _, indices = torch.topk(logits, 6, dim=-1)
                
                # Compute P (Vectorized)
                # We need P for this dataset
                N, K = indices.shape
                one_hot = torch.zeros((N, 64), device=indices.device)
                one_hot.scatter_(1, indices, 1.0)
                P_tensor = torch.matmul(one_hot.T, one_hot)
                P_tensor.fill_diagonal_(0)
                
                patterns.append(P_tensor.numpy())
            except Exception as e:
                print(f"Failed to load {f}: {e}")
        return patterns

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Randomize Start Pattern
        self.current_pattern_idx = np.random.randint(0, self.num_patterns)
        self.P = self.traffic_patterns[self.current_pattern_idx]
        
        self.placement = self.initial_placement.copy()
        if options and options.get('randomize'):
            np.random.shuffle(self.placement)
            
        self.current_step = 0
        self.shift_timer = self.steps_until_shift
        
        return self._get_obs(), {}

    def _get_obs(self):
        # 1. Device IDs
        dev_ids = self.placement.astype(np.float32)
        
        # 2. Affinity to Devices (Based on CURRENT P)
        device_mask = np.zeros((self.num_experts, self.num_devices), dtype=np.float32)
        device_mask[np.arange(self.num_experts), self.placement.astype(int)] = 1.0
        
        affinity = self.P @ device_mask
        affinity = affinity / (np.mean(affinity) + 1e-5)
        
        obs = np.concatenate([dev_ids[:, None], affinity], axis=1)
        return obs.flatten().astype(np.float32)

    def _calculate_cut_cost(self):
        dev_a = self.placement[:, None]
        dev_b = self.placement[None, :]
        mask = (dev_a != dev_b).astype(np.float32)
        cost = np.sum(self.P * mask) / 2.0
        return cost

    def step(self, action):
        expert_a, target_device = action
        current_device = self.placement[expert_a]
        
        prev_cost = self._calculate_cut_cost()
        migration_cost = 0.0
        
        reward = 0.0
        
        # Domain Shift Logic
        self.shift_timer -= 1
        info_msg = ""
        if self.shift_timer <= 0:
            # SHIFT TRAFFIC!
            new_idx = np.random.randint(0, self.num_patterns)
            # Make sure it actually changes to force adaptation? 
            # Or just random. Random is fine.
            self.current_pattern_idx = new_idx
            self.P = self.traffic_patterns[self.current_pattern_idx]
            self.shift_timer = self.steps_until_shift # Reset timer
            info_msg = "TRAFFIC_SHIFT"
            
            # Recalculate cost with NEW P (Placement hasn't changed yet, but cost has)
            # This is the "New Baseline" for the step.
            prev_cost = self._calculate_cut_cost() 

        # Execute Action
        if current_device == target_device:
            # No-Op
            # No Migration Cost.
            # Reward? If we interpret Reward as "Negative Cost", then Reward = -CurrentCost.
            # If standard "Improvement" reward, No-Op = 0 improvement.
            # But RL needs to know absolute cost to balance.
            # Let's use Reward = - (CommCost + MigrCost) * Scale
            # This is cleaner for Multi-Objective.
            pass
        else:
            # Greedy Swap Logic
            mask = (self.placement == target_device)
            candidates = np.where(mask)[0]
            
            if len(candidates) > 0:
                best_delta = -float('inf')
                best_candidate = -1
                
                # Check candidates
                for cand in candidates:
                    self.placement[expert_a] = target_device
                    self.placement[cand] = current_device
                    c_cost = self._calculate_cut_cost()
                    delta = prev_cost - c_cost
                    if delta > best_delta:
                        best_delta = delta
                        best_candidate = cand
                    # Revert
                    self.placement[expert_a] = current_device
                    self.placement[cand] = target_device
                
                # Apply Best
                # We APPLY it if it improves OR if the agent forces it.
                # Agent forced it. So we apply.
                if best_candidate != -1:
                    self.placement[expert_a] = target_device
                    self.placement[best_candidate] = current_device
                    
                    # Migration Cost!
                    # We swapped 2 experts.
                    migration_cost = 2.0 * self.migration_penalty 
            else:
                # Fail
                pass

        # Calculate Final Total Cost
        current_comm_cost = self._calculate_cut_cost()
        total_cost = current_comm_cost + migration_cost
        
        # Reward Function
        # We want to minimize Cost.
        # Maximize Negative Cost.
        # Scale: Cost is ~60000. 
        # Reward = -TotalCost / 1000.0
        reward = -total_cost / 1000.0
        
        self.current_step += 1
        terminated = (self.current_step >= self.max_steps)
        
        info = {
            "comm_cost": current_comm_cost,
            "migr_cost": migration_cost,
            "msg": info_msg
        }
        
        return self._get_obs(), reward, terminated, False, info
