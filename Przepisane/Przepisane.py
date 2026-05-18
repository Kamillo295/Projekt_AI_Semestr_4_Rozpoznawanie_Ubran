import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import random
import numpy as np

CONFIG = {

    "batch_size": 128,
    "epochs": 30,
    "learning_rate": 0.0005,
    "random_seed": 42,    
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class FunctionCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv_layers = nn.Sequential(
            
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1
                ),

            nn.ReLU(),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
                ),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2)
            )
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 14 * 14, 128),

            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 10)
            )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)

        return x

def get_data_loaders(batch_size):

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
                (0.5, ),
                (0.5, )
        )
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
                (0.5, ),
                (0.5, )
        )
    ])

    train_dataset = torchvision.datasets.FashionMNIST(
        
        root="./data",
        train=True,
        download=True,
        transform=train_transform
        
    )

    test_dataset = torchvision.datasets.FashionMNIST(
    
        root="./data",
        train=False,
        download=True,
        transform=test_transform
    )

    train_loader = torch.utils.data.DataLoader(
    
        train_dataset,
        batch_size=batch_size,
        shuffle=True

    )

    test_loader = torch.utils.data.DataLoader(
    
        test_dataset,
        batch_size=batch_size,
        shuffle=False

    )

    return train_loader, test_loader

