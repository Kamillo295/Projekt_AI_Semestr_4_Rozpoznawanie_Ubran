import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import random
import numpy as np

CONFIG = {

    "batch_size": 64,
    "epochs": 30,
    "learning_rate": 0.0003,
    "random_seed": 42,
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

class FashionCNN(nn.Module):

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

def evaluate_model(model, loader, device):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
    accuracy = 100 * correct / total
    return accuracy

def main():
    set_seed(CONFIG["random_seed"])

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
        )

    print(f"\nUrzadzenie: {device}")
    print(f"\nPobieranie danych...")

    train_loader, test_loader = get_data_loaders(
        CONFIG["batch_size"]
        )

    model = FashionCNN().to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=CONFIG["learning_rate"]
        )

    print("\n Rozpoczynanie treningu\n")

    best_accuracy = 0

    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()     # liczy gradienty, czyli oblicza jak trzeba zmieniæ wagi modelu, ¿eby loss by³ mniejszy.
            optimizer.step()    # aktualizacaja wag na podstawie obliczonego gradientu

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()


        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = (
            correct_predictions / total_samples
        ) * 100

        test_accuracy = evaluate_model( model, test_loader, device )

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(
                model.state_dict(),
                "fashion_cnn_model.pth"
                )

        print(
            f"Epoka {epoch:2d}/{CONFIG['epochs']} | "
            f"Loss: {epoch_loss:.4f} | "
            f"Train Accuracy: {epoch_accuracy:.2f}% | "
            f"Test Accuracy: {test_accuracy:.2f}%"
        )

    print("\n====================================")
    print("TRENING ZAKONCZONY")
    print("====================================")

    final_accuracy = evaluate_model(model, test_loader, device)

    print(f"\nNajlepsza dokladnosc: {best_accuracy:.2f}%")
    print("\nModel zapisany jako:")
    print("fashion_cnn_model.pth")

if __name__ == "__main__":
    main()