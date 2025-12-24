
import numpy as np
import torch
import random
import math
from tqdm import tqdm

def compute_cooccurrence_matrix(data_path, k=6, num_experts=64):
    print(f"Computing Co-occurrence Matrix from {data_path}...")
    data = torch.load(data_path, map_location='cpu')
    logits = data.get('logits').squeeze(0) # [TotalSeq, Experts]
    if logits.shape[1] != num_experts:
         # simple pad/trim or error
         pass
         
    _, indices = torch.topk(logits, k, dim=-1) # [N, K]
    
    indices = indices.to('cuda' if torch.cuda.is_available() else 'cpu')
    N_samples, K = indices.shape
    one_hot = torch.zeros((N_samples, num_experts), device=indices.device)
    one_hot.scatter_(1, indices, 1.0)
    P = torch.matmul(one_hot.T, one_hot)
    P.fill_diagonal_(0)
    return P.cpu().numpy(), indices.cpu()

def calc_ct(placement, indices):
    # placement: [64] -> device_id
    # indices: [N, K]
    devs = placement[indices] # [N, K]
    # Unique count
    # Fast estimate or loop
    count = 0
    for row in devs:
        count += len(np.unique(row))
    return count / len(devs)

def calc_cut_cost(P, placement):
    dev_a = placement[:, None]
    dev_b = placement[None, :]
    mask = (dev_a != dev_b).astype(np.float32)
    return np.sum(P * mask) / 2.0

def simulated_annealing(P, indices, num_devices=8, steps=10000):
    num_experts = P.shape[0]
    experts_per_device = num_experts // num_devices
    
    # Init: Block
    current_placement = np.repeat(np.arange(num_devices), experts_per_device)
    
    # Or Random?
    # np.random.shuffle(current_placement)
    
    current_cost = calc_cut_cost(P, current_placement)
    best_placement = current_placement.copy()
    best_cost = current_cost
    
    T = 1000.0
    alpha = 0.9995
    
    print(f"Initial Cost: {current_cost}")
    
    for i in tqdm(range(steps)):
        # Propose Swap
        a = random.randint(0, num_experts-1)
        b = random.randint(0, num_experts-1)
        
        while current_placement[a] == current_placement[b]:
             b = random.randint(0, num_experts-1)
             
        # Candidate
        new_placement = current_placement.copy()
        temp = new_placement[a]
        new_placement[a] = new_placement[b]
        new_placement[b] = temp
        
        new_cost = calc_cut_cost(P, new_placement)
        
        delta = new_cost - current_cost
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_placement = new_placement
            current_cost = new_cost
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_placement = current_placement.copy()
                
        T *= alpha
        
    print(f"Final Cost: {best_cost}")
    return best_placement

def main():
    import glob
    import os
    
    files = sorted(glob.glob("data/benchmarks/*.pt"))
    print(f"Found {len(files)} benchmark files: {files}")
    
    # Baseline Placement (Block)
    block_placement = np.repeat(np.arange(8), 8)
    
    results = []
    
    for path in files:
        name = os.path.basename(path).replace(".pt", "")
        print(f"\n{'='*40}\nProcessing {name}...\n{'='*40}")
        
        try:
            P, indices = compute_cooccurrence_matrix(path)
        except Exception as e:
            print(f"Failed to load {name}: {e}")
            continue
            
        # Naive
        ct_naive = calc_ct(block_placement, indices)
        print(f"[{name}] Naive CT: {ct_naive:.4f}")
        
        # Optimize
        # To save time for verifying multiple datasets, we reduce steps slightly or keep 20k
        # 20k is fast enough (few seconds)
        opt_placement = simulated_annealing(P, indices, steps=10000)
        
        ct_opt = calc_ct(opt_placement, indices)
        reduction = ct_naive - ct_opt
        pct = (reduction / ct_naive) * 100
        
        print(f"[{name}] Optimized CT: {ct_opt:.4f}")
        print(f"[{name}] Reduction: {reduction:.4f} ({pct:.1f}%)")
        
        results.append((name, ct_naive, ct_opt, pct))
        
    print("\n\n" + "="*50)
    print(f"{'Dataset':<25} | {'Naive CT':<10} | {'SA CT':<10} | {'Reduction':<10}")
    print("-" * 65)
    for res in results:
        print(f"{res[0]:<25} | {res[1]:.4f}     | {res[2]:.4f}     | {res[3]:.1f}%")
    print("="*50)

if __name__ == "__main__":
    main()
