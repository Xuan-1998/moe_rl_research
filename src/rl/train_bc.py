import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

# Define Policy Network (Same architecture as PPO would pick ideally)
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

def main():
    print("Loading Demonstrations...")
    demos = torch.load("data/benchmarks/bc_demonstrations.pt", weights_only=False)
    
    # Unpack
    obs_list = [d[0] for d in demos]
    act_list = [d[1] for d in demos]
    
    X = torch.tensor(np.array(obs_list), dtype=torch.float32)
    Y = torch.tensor(np.array(act_list), dtype=torch.long)
    
    # Split Train/Val?
    # We want to overfit to 'Offline Knowledge' for now to prove capacity.
    # Generalization is Step 2.
    
    dataset = TensorDataset(X, Y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    obs_dim = X.shape[1]
    act_dim = 8 # 8 devices
    
    policy = PolicyNet(obs_dim, act_dim)
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Training BC Policy on {len(X)} samples for 1000 epochs...")
    
    for epoch in range(1000):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            logits = policy(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
            
        acc = correct / total
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss {total_loss:.4f} | Acc {acc*100:.1f}%")
            
    print("Training Complete.")
    torch.save(policy.state_dict(), "bc_policy.pth")

if __name__ == "__main__":
    main()
