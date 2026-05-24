import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import random
import numpy as np

# ustawienia programu
CONFIG = {

    "batch_size": 64,
    "epochs": 30,
    "learning_rate": 0.0003,
    "random_seed": 42,
}

# ustawianie losowosci
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # losowosc dla GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# model CNN
class FashionCNN(nn.Module):

    def __init__(self):

        super().__init__()

        # warstwy CNN
        self.conv_layers = nn.Sequential(
            
            # pierwsza konwolucja
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1
                ),

            # funkcja aktywacji
            nn.ReLU(),

            # druga konwolucja
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
                ),

            # funkcja aktywacji
            nn.ReLU(),

            # zmniejszanie danych
            nn.MaxPool2d(kernel_size=2)
            )
        
        # warstwy klasyfikujace
        self.fc_layers = nn.Sequential(

            # splaszczanie danych
            nn.Flatten(),

            # warstwa liniowa
            nn.Linear(64 * 14 * 14, 128),

            nn.ReLU(),

            # ochrona przed overfittingiem
            nn.Dropout(0.3),

            # wyjscie modelu
            nn.Linear(128, 10)
            )

    # przeplyw danych przez model
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)

        return x

# przygotowanie danych
def get_data_loaders(batch_size):

    # transformacje treningowe
    train_transform = transforms.Compose([
        transforms.ToTensor(),

        # normalizacja danych
        transforms.Normalize(
                (0.5, ),
                (0.5, )
        )
    ])

    # transformacje testowe
    test_transform = transforms.Compose([
        transforms.ToTensor(),

        # normalizacja danych
        transforms.Normalize(
                (0.5, ),
                (0.5, )
        )
    ])

    # dane treningowe
    train_dataset = torchvision.datasets.FashionMNIST(
        
        root="./data",
        train=True,
        download=True,
        transform=train_transform
        
    )

    # dane testowe
    test_dataset = torchvision.datasets.FashionMNIST(
    
        root="./data",
        train=False,
        download=True,
        transform=test_transform
    )

    # loader treningowy
    train_loader = torch.utils.data.DataLoader(
    
        train_dataset,
        batch_size=batch_size,
        shuffle=True

    )

    # loader testowy
    test_loader = torch.utils.data.DataLoader(
    
        test_dataset,
        batch_size=batch_size,
        shuffle=False

    )

    return train_loader, test_loader

# test modelu
def evaluate_model(model, loader, device):

    # tryb testowy
    model.eval()

    correct = 0
    total = 0

    # brak gradientow
    with torch.no_grad():

        for images, labels in loader:

            # przenoszenie danych na GPU/CPU
            images = images.to(device)
            labels = labels.to(device)

            # przewidywanie modelu
            outputs = model(images)

            # wybor najlepszej klasy
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            # liczenie poprawnych odpowiedzi
            correct += (predicted == labels).sum().item()
        
    # obliczanie accuracy
    accuracy = 100 * correct / total

    return accuracy

# glowna funkcja programu
def main():

    # ustawienie losowosci
    set_seed(CONFIG["random_seed"])

    # wybor GPU lub CPU
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
        )

    print(f"\nUrzadzenie: {device}")
    print(f"\nPobieranie danych...")

    # pobranie danych
    train_loader, test_loader = get_data_loaders(
        CONFIG["batch_size"]
        )

    # tworzenie modelu
    model = FashionCNN().to(device)

    # funkcja straty
    criterion = nn.CrossEntropyLoss()

    # optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=CONFIG["learning_rate"]
        )

    print("\n Rozpoczynanie treningu\n")

    best_accuracy = 0

    # petla treningowa
    for epoch in range(1, CONFIG["epochs"] + 1):

        # tryb treningowy
        model.train()

        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        # petla batchy
        for images, labels in train_loader:

            # przenoszenie danych
            images = images.to(device)
            labels = labels.to(device)

            # zerowanie gradientow
            optimizer.zero_grad()

            # forward pass
            outputs = model(images)

            # obliczanie bledu
            loss = criterion(outputs, labels)

            # liczenie gradientow
            loss.backward()

            # aktualizacja wag
            optimizer.step()

            # dodawanie loss
            running_loss += loss.item()

            # wybor najlepszej klasy
            _, predicted = torch.max(outputs, 1)

            total_samples += labels.size(0)

            # liczenie poprawnych odpowiedzi
            correct_predictions += (predicted == labels).sum().item()

        # sredni loss epoki
        epoch_loss = running_loss / len(train_loader)

        # accuracy treningowe
        epoch_accuracy = ( correct_predictions / total_samples ) * 100

        # test modelu
        test_accuracy = evaluate_model( model, test_loader, device )

        # zapis najlepszego modelu
        if test_accuracy > best_accuracy:

            best_accuracy = test_accuracy

            torch.save(
                model.state_dict(),
                "fashion_cnn_model.pth"
                )

        # wyswietlanie wynikow
        print(
            f"Epoka {epoch:2d}/{CONFIG['epochs']} | "
            f"Loss: {epoch_loss:.4f} | "
            f"Train Accuracy: {epoch_accuracy:.2f}% | "
            f"Test Accuracy: {test_accuracy:.2f}%"
        )

    print("\n====================================")
    print("TRENING ZAKONCZONY")
    print("====================================")

    # finalny test modelu
    final_accuracy = evaluate_model(model, test_loader, device)

    print(f"\nNajlepsza dokladnosc: {best_accuracy:.2f}%")

    print("\nModel zapisany jako:")
    print("fashion_cnn_model.pth")

# uruchomienie programu
if __name__ == "__main__":
    main()