import pandas as pd
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
import os

class MyCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(MyCrossEntropyLoss, self).__init__()
   
    def forward(self, output, target):
        target = F.one_hot(target, num_classes=output.shape[1])
        o = -target * F.log_softmax(output, dim=-1)
        e = o.sum(-1).mean()
        return e


class ObesityDataset(Dataset):
    def __init__(self, file_path):
        df = pd.read_csv(file_path)
        # Skip header row and get target column (last column)
        target = df.iloc[1:, -1].to_numpy().astype(int)  # Skip header, convert to int
        features = torch.tensor(df.iloc[1:, :-1].to_numpy(), dtype=torch.float32)  # Skip header
        
        self.target = torch.tensor(target, dtype=torch.long)  # Keep as class indices
        self.features = features

        #print sizes:
        print(f"Features shape: {features.shape}, Target shape: {self.target.shape}")
        print(f"Target classes: {torch.unique(self.target)}")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.target[idx]
    
class ObesityModel(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=24, output_dim=7):  # Changed to 7 classes
        super(ObesityModel, self).__init__()
        
        # FeedForward layer (equivalent to ff1 in obesity.ng)
        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, hidden_dim//2, bias=False),
            nn.LeakyReLU(),
            #nn.Dropout(0.25),
            nn.Linear(hidden_dim//2, hidden_dim, bias=False),
            nn.Tanh(),
            #nn.Dropout(0.25)
        )
        
        # Output layer (equivalent to out in obesity.ng)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.feed_forward(x)
        x = self.output_layer(x)
        return x


def main():
    model = ObesityModel()
    print(model)
    train_dataset = ObesityDataset("datasets/Obesity_train.csv")
    valdn_dataset = ObesityDataset("datasets/Obesity_valdn.csv")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)  # Increased batch size
    valdn_loader = DataLoader(valdn_dataset, batch_size=32, shuffle=False)  # No shuffle for validation

    print(f"Number of training batches: {len(train_loader)}, validation batches: {len(valdn_loader)}")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)  # Increased learning rate
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)  # Adjusted scheduler

    #print the number of parameters in the model
    print(f"Number of parameters in the model: {sum(p.numel() for p in model.parameters())}")
    yymmddhhmmss = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = "runs/python"
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    f = open(f"{run_dir}/training_metrics_{yymmddhhmmss}.csv", "w")
    f.write("epoch,train_loss,train_acc,valdn_loss,valdn_acc,lr,batches\n")

    loss_fn = MyCrossEntropyLoss()
    epochs = 200
    batches = 0
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for i, (features, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(features)
            output.retain_grad()
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            batches += 1
            # Calculate accuracy
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

        # Move scheduler step outside the inner loop
        lr_scheduler.step()
        
        train_loss /= len(train_loader)
        train_acc = 100 * correct / total

        # evaluate the model on the validation set
        model.eval()
        valdn_loss = 0
        valdn_correct = 0
        valdn_total = 0
        
        with torch.no_grad():
            model.eval()
            for features, target in valdn_loader:
                output = model(features)
                loss = loss_fn(output, target)
                valdn_loss += loss.item()
                
                # Calculate validation accuracy
                _, predicted = torch.max(output.data, 1)
                valdn_total += target.size(0)
                valdn_correct += (predicted == target).sum().item()
            model.train()

        valdn_loss /= len(valdn_loader)
        valdn_acc = 100 * valdn_correct / valdn_total

        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:02d}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Valdn Loss: {valdn_loss:.4f}, Valdn Acc: {valdn_acc:.2f}%, Lr: {lr:.6f}")

        f.write(f"{epoch},{train_loss:.4f},{train_acc:.2f},{valdn_loss:.4f},{valdn_acc:.2f},{lr:.6f},{batches}\n")
        f.flush()
        # load 1 validation batch and print the output of the model
    for features, target in valdn_loader:
        output = model(features)
        print(output)
        break
    f.close()

if __name__ == "__main__":
    main()
