import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch

class ConstructivePlacementEnv(gym.Env):
    """
    Constructive RL Environment.
    Places experts one by one, sorted by connection degree.
    Constraint: Each device has max capacity (e.g. 8).
    
    Observation:
        - Affinity of 'Current Expert' to each Device (Sum of weights to experts already on that device).
        - Current Fill Level of each Device (Normalized).
    
    Action:
        - Select Device ID (0..NumDevices-1).
    """
    def __init__(self, cooccurrence_matrix, num_devices=8):
        super().__init__()
        self.P_raw = cooccurrence_matrix # [N, N]
        self.num_experts = self.P_raw.shape[0]
        self.num_devices = num_devices
        self.capacity = self.num_experts // self.num_devices
        
        # Sort Experts by Degree (Heavy hitters first)
        self.degrees = self.P_raw.sum(axis=1)
        self.sorted_indices = np.argsort(self.degrees)[::-1] # Descending
        
        # Reorder P to match sorted order (Crucial step!)
        # P_sorted[i, j] means weight between (i-th heaviest) and (j-th heaviest)
        self.P = self.P_raw[self.sorted_indices][:, self.sorted_indices]
        
        # Mapping back to original indices (for final output)
        self.original_indices = self.sorted_indices
        
        # State
        self.current_expert_idx = 0
        self.node_to_device = np.full(self.num_experts, -1, dtype=int)
        self.device_fill = np.zeros(self.num_devices, dtype=int)
        
        # Observation: [Affinity_to_Dev0...Dev7, Fill_Dev0...Dev7]
        # Size = 2 * NumDevices
        self.observation_space = spaces.Box(
            low=0.0, high=1e6, shape=(2 * self.num_devices,), dtype=np.float32
        )
        
        self.action_space = spaces.Discrete(self.num_devices)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_expert_idx = 0
        self.node_to_device.fill(-1)
        self.device_fill.fill(0)
        return self._get_obs(), {"action_mask": self._get_action_mask()}

    def _get_action_mask(self):
        # Mask out full devices
        mask = (self.device_fill < self.capacity).astype(bool)
        return mask

    def _get_obs(self):
        # Check termination
        if self.current_expert_idx >= self.num_experts:
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        # Calculate affinity of current expert to each device
        affinities = np.zeros(self.num_devices, dtype=np.float32)
        
        if self.current_expert_idx > 0:
            # Get row of current expert in P
            # interactions with ALL experts
            weights = self.P[self.current_expert_idx] 
            
            # Filter for only PLACED experts
            # node_to_device has -1 for unplaced.
            
            # Vectorized sum:
            # We iterate devices? Or iterate placed nodes?
            # Iterating devices is fast (K=8).
            
            placed_mask = (self.node_to_device != -1)
            # Actually, calculating P[i] * mask_dev_k is better.
            
            for k in range(self.num_devices):
                # nodes on device k
                dev_mask = (self.node_to_device == k)
                if np.any(dev_mask):
                    affinities[k] = np.sum(weights[dev_mask])
                    
        # Normalize affinities?
        max_aff = np.max(affinities) if np.max(affinities) > 0 else 1.0
        affinities /= max_aff
        
        # Fill levels (Normalized)
        fills = self.device_fill / self.capacity
        
        return np.concatenate([affinities, fills]).astype(np.float32)

    def step(self, action):
        # Action is Device ID
        
        # Check validity
        reward_penalty = 0.0
        
        if self.device_fill[action] >= self.capacity:
            # Invalid action! 
            # Fallback: Pick valid device with highest affinity to current node.
            # If all affinities 0, pick first valid.
            
            # Mask out full devices
            valid_mask = (self.device_fill < self.capacity)
            valid_indices = np.where(valid_mask)[0]
            
            # Helper to calculate affinity for specific devices
            # We already calculated affinities in _get_obs, but let's re-calc specifically for fallback logic or just pick?
            # Re-calc matches logic.
            
            best_fallback = -1
            best_aff = -1.0
            
            weights = self.P[self.current_expert_idx]
            # Use `node_to_device` to check connections
            
            for dev_idx in valid_indices:
                dev_mask = (self.node_to_device == dev_idx)
                aff = 0.0
                if np.any(dev_mask):
                    aff = np.sum(weights[dev_mask])
                
                if aff > best_aff:
                    best_aff = aff
                    best_fallback = dev_idx
            
            # Override action
            action = best_fallback
            reward_penalty = -10.0 # Small penalty for invalid choice
            
        # Place expert
        self.node_to_device[self.current_expert_idx] = action
        self.device_fill[action] += 1
        
        # Calculate Reward (Incremental)
        # We want to MAXIMIZE Internal Edges.
        
        weights = self.P[self.current_expert_idx]
        dev_mask = (self.node_to_device == action)
        
        # Internal edges added
        internal_gain = np.sum(weights[dev_mask])
        
        # Reward = Gain + Penalty
        reward = internal_gain + reward_penalty
        
        # Advance
        self.current_expert_idx += 1
        terminated = (self.current_expert_idx >= self.num_experts)
        
        info = {"action_mask": self._get_action_mask()}
        
        if terminated:
            # Final Cost Calculation check
            real_cut_cost = self._calculate_final_cost()
            info['final_cost'] = real_cut_cost
        
        return self._get_obs(), reward, terminated, False, info

    def _calculate_final_cost(self):
        dev_a = self.node_to_device[:, None]
        dev_b = self.node_to_device[None, :]
        mask = (dev_a != dev_b).astype(np.float32)
        cost = np.sum(self.P * mask) / 2.0
        return cost
        
    def get_final_placement(self):
        # We need to unsort
        # self.node_to_device is sorted by degree.
        # unsorted[original_indices[i]] = value[i]
        unsorted = np.zeros_like(self.node_to_device)
        unsorted[self.sorted_indices] = self.node_to_device
        return unsorted
