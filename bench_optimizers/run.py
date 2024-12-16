import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        return self.fc(x)

BATCH_SIZE = 64
EPOCHS = 5

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

def train_and_evaluate(optimizer_name):
    wandb.init(project="optimizer_comparison", name=f"{optimizer_name}_run")
    config = wandb.config
    config.batch_size = BATCH_SIZE
    config.epochs = EPOCHS
    config.optimizer = optimizer_name

    model = SimpleNN()
    criterion = nn.CrossEntropyLoss()
    
    if optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=0.01)
    elif optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    elif optimizer_name == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=0.001)

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        wandb.log({"train_loss": train_loss, "epoch": epoch + 1})

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss /= len(test_loader)
        accuracy = 100.0 * correct / total

        wandb.log({"val_loss": val_loss, "accuracy": accuracy, "epoch": epoch + 1})

    wandb.finish()

for optimizer in ["SGD", "Adam", "RMSprop"]:
    train_and_evaluate(optimizer)
