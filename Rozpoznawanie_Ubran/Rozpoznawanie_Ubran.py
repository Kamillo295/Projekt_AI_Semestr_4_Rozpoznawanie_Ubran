import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import random
import numpy as np

# ============================================================
# ROZPOZNAWANIE UBRAŃ - FASHION MNIST
# PROSTY I CZYTELNY CNN
# ============================================================
#
# Ten projekt:
# - rozpoznaje ubrania na obrazkach
# - używa sieci CNN
# - działa na CPU lub GPU
#
# Projekt jest napisany tak,
# aby łatwo było go zrozumieć początkującej osobie.
#
# ============================================================
# CO TO JEST CNN?
# ============================================================
#
# CNN = Convolutional Neural Network
#
# Jest to specjalny rodzaj sieci neuronowej
# przeznaczony do analizy obrazów.
#
# Zwykłe warstwy Linear:
# - widzą obraz tylko jako liczby
#
# CNN:
# - wykrywa krawędzie
# - wykrywa kształty
# - wykrywa wzory
#
# Dzięki temu dużo lepiej radzi sobie z obrazami.
#
# ============================================================


# ============================================================
# KONFIGURACJA PROGRAMU
# ============================================================

CONFIG = {

    # --------------------------------------------------------
    # BATCH SIZE
    #
    # Ile obrazków sieć analizuje jednocześnie
    # --------------------------------------------------------
    "batch_size": 128,

    # --------------------------------------------------------
    # EPOCHS
    #
    # Ile razy sieć zobaczy cały zbiór danych
    # --------------------------------------------------------
    "epochs": 500,

    # --------------------------------------------------------
    # LEARNING RATE
    #
    # Jak szybko sieć zmienia swoje wagi
    # --------------------------------------------------------
    "learning_rate": 0.0001,

    # --------------------------------------------------------
    # RANDOM SEED
    #
    # Dzięki temu wyniki są bardziej powtarzalne
    # --------------------------------------------------------
    "random_seed": 42,
}


# ============================================================
# USTAWIANIE LOSOWOŚCI
# ============================================================

def set_seed(seed):

    # --------------------------------------------------------
    # Python random
    # --------------------------------------------------------
    random.seed(seed)

    # --------------------------------------------------------
    # NumPy random
    # --------------------------------------------------------
    np.random.seed(seed)

    # --------------------------------------------------------
    # PyTorch random
    # --------------------------------------------------------
    torch.manual_seed(seed)

    # --------------------------------------------------------
    # GPU random
    # --------------------------------------------------------
    torch.cuda.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)


# ============================================================
# GŁÓWNA SIEĆ CNN
# ============================================================

class FashionCNN(nn.Module):

    def __init__(self):

        super().__init__()

        # ====================================================
        # CZĘŚĆ CONVOLUTION
        # ====================================================
        #
        # Ta część analizuje obraz.
        #
        # Conv2D:
        # - wykrywa wzory
        # - wykrywa kształty
        # - wykrywa cechy obrazu
        #
        # ReLU:
        # - wprowadza nieliniowość
        #
        # MaxPool:
        # - zmniejsza rozmiar obrazu
        # - zostawia najważniejsze informacje
        #
        # ====================================================

        self.conv_layers = nn.Sequential(

            # ------------------------------------------------
            # PIERWSZA WARSTWA CNN
            #
            # in_channels=1
            # bo obraz jest czarno-biały
            #
            # out_channels=32
            # tworzymy 32 różne mapy cech
            # ------------------------------------------------

            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),

            # ------------------------------------------------
            # Funkcja aktywacji
            # ------------------------------------------------

            nn.ReLU(),

            # ------------------------------------------------
            # DRUGA WARSTWA CNN
            # ------------------------------------------------

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            # ------------------------------------------------
            # MAX POOLING
            #
            # Zmniejsza obraz:
            #
            # 28x28 -> 14x14
            #
            # Dzięki temu:
            # - mniej obliczeń
            # - szybsze uczenie
            # - mniej szumu
            # ------------------------------------------------

            nn.MaxPool2d(kernel_size=2)
        )

        # ====================================================
        # CZĘŚĆ DECYZYJNA
        # ====================================================
        #
        # Tutaj sieć podejmuje decyzję,
        # jaką klasę widzi na obrazku.
        #
        # ====================================================

        self.fc_layers = nn.Sequential(

            # ------------------------------------------------
            # Zamiana danych 3D -> 1D
            # ------------------------------------------------

            nn.Flatten(),

            # ------------------------------------------------
            # Warstwa Linear
            #
            # 64 kanały
            # 14x14 rozmiar
            #
            # 64 * 14 * 14
            # ------------------------------------------------

            nn.Linear(64 * 14 * 14, 128),

            nn.ReLU(),

            # ------------------------------------------------
            # DROPOUT
            #
            # Losowo wyłącza część neuronów.
            #
            # Pomaga przeciw:
            # OVERFITTINGOWI
            #
            # Czyli sytuacji, gdy:
            # - model zapamiętuje trening
            # - ale gorzej działa na nowych danych
            # ------------------------------------------------

            nn.Dropout(0.3),

            # ------------------------------------------------
            # OSTATNIA WARSTWA
            #
            # 10 wyjść = 10 klas ubrań
            # ------------------------------------------------

            nn.Linear(128, 10)
        )

    # ========================================================
    # FORWARD PASS
    #
    # Określa przepływ danych przez sieć
    # ========================================================

    def forward(self, x):

        # ----------------------------------------------------
        # Analiza obrazu
        # ----------------------------------------------------

        x = self.conv_layers(x)

        # ----------------------------------------------------
        # Podjęcie decyzji
        # ----------------------------------------------------

        x = self.fc_layers(x)

        return x


# ============================================================
# POBIERANIE DANYCH
# ============================================================

def get_data_loaders(batch_size):

    # ========================================================
    # TRANSFORMACJE DLA TRENINGU
    # ========================================================
    #
    # ToTensor():
    # zamienia obraz na Tensor PyTorch
    #
    # Normalize():
    # normalizuje dane
    #
    # Dzięki temu:
    # - trening jest stabilniejszy
    # - sieć uczy się lepiej
    #
    # ========================================================

    train_transform = transforms.Compose([

        transforms.ToTensor(),

        transforms.Normalize(
            (0.5,),
            (0.5,)
        )
    ])

    # ========================================================
    # TEST TRANSFORM
    # ========================================================

    test_transform = transforms.Compose([

        transforms.ToTensor(),

        transforms.Normalize(
            (0.5,),
            (0.5,)
        )
    ])

    # ========================================================
    # DANE TRENINGOWE
    # ========================================================

    train_dataset = torchvision.datasets.FashionMNIST(

        root="./data",

        train=True,

        download=True,

        transform=train_transform
    )

    # ========================================================
    # DANE TESTOWE
    # ========================================================

    test_dataset = torchvision.datasets.FashionMNIST(

        root="./data",

        train=False,

        download=True,

        transform=test_transform
    )

    # ========================================================
    # DATALOADER
    #
    # Dzieli dane na batch'e
    # ========================================================

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


# ============================================================
# TESTOWANIE MODELU
# ============================================================

def evaluate_model(model, loader, device):

    # ========================================================
    # TRYB TESTOWANIA
    # ========================================================

    model.eval()

    correct = 0

    total = 0

    # ========================================================
    # Wyłączenie gradientów
    #
    # Oszczędza pamięć i przyspiesza test
    # ========================================================

    with torch.no_grad():

        for images, labels in loader:

            # ------------------------------------------------
            # GPU / CPU
            # ------------------------------------------------

            images = images.to(device)

            labels = labels.to(device)

            # ------------------------------------------------
            # Predykcja
            # ------------------------------------------------

            outputs = model(images)

            # ------------------------------------------------
            # Wybór klasy z najwyższym wynikiem
            # ------------------------------------------------

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (

                predicted == labels

            ).sum().item()

    accuracy = 100 * correct / total

    return accuracy


# ============================================================
# GŁÓWNA FUNKCJA PROGRAMU
# ============================================================

def main():

    # ========================================================
    # POWTARZALNOŚĆ
    # ========================================================

    set_seed(CONFIG["random_seed"])

    # ========================================================
    # SPRAWDZANIE GPU
    # ========================================================

    device = torch.device(

        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"\nUrządzenie: {device}")

    # ========================================================
    # POBIERANIE DANYCH
    # ========================================================

    print("\nPobieranie danych...")

    train_loader, test_loader = get_data_loaders(

        CONFIG["batch_size"]
    )

    # ========================================================
    # TWORZENIE MODELU
    # ========================================================

    model = FashionCNN().to(device)

    # ========================================================
    # FUNKCJA STRATY
    # ========================================================
    #
    # CrossEntropyLoss:
    # standard dla klasyfikacji
    #
    # ========================================================

    criterion = nn.CrossEntropyLoss()

    # ========================================================
    # OPTYMALIZATOR
    # ========================================================
    #
    # Adam:
    # automatycznie poprawia wagi sieci
    #
    # ========================================================

    optimizer = optim.Adam(

        model.parameters(),

        lr=CONFIG["learning_rate"]
    )

    print("\nRozpoczynam trening...\n")

    # ========================================================
    # NAJLEPSZA DOKŁADNOŚĆ
    # ========================================================

    best_accuracy = 0

    # ========================================================
    # PĘTLA TRENINGOWA
    # ========================================================

    for epoch in range(1, CONFIG["epochs"] + 1):

        # ====================================================
        # TRYB TRENINGU
        # ====================================================

        model.train()

        running_loss = 0.0

        correct_predictions = 0

        total_samples = 0

        # ====================================================
        # ITERACJA PO BATCHACH
        # ====================================================

        for images, labels in train_loader:

            # ------------------------------------------------
            # GPU / CPU
            # ------------------------------------------------

            images = images.to(device)

            labels = labels.to(device)

            # ------------------------------------------------
            # Zerowanie gradientów
            # ------------------------------------------------

            optimizer.zero_grad()

            # ------------------------------------------------
            # FORWARD PASS
            #
            # Sieć próbuje zgadnąć klasy
            # ------------------------------------------------

            outputs = model(images)

            # ------------------------------------------------
            # LOSS
            #
            # Obliczenie błędu
            # ------------------------------------------------

            loss = criterion(outputs, labels)

            # ------------------------------------------------
            # BACKPROPAGATION
            #
            # Obliczanie gradientów
            # ------------------------------------------------

            loss.backward()

            # ------------------------------------------------
            # AKTUALIZACJA WAG
            # ------------------------------------------------

            optimizer.step()

            # ------------------------------------------------
            # STATYSTYKI
            # ------------------------------------------------

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total_samples += labels.size(0)

            correct_predictions += (

                predicted == labels

            ).sum().item()

        # ====================================================
        # WYNIKI EPOKI
        # ====================================================

        epoch_loss = running_loss / len(train_loader)

        epoch_accuracy = (

            correct_predictions / total_samples

        ) * 100

        # ====================================================
        # TEST MODELU
        # ====================================================

        test_accuracy = evaluate_model(

            model,

            test_loader,

            device
        )

        # ====================================================
        # ZAPIS NAJLEPSZEGO MODELU
        # ====================================================

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

    # ========================================================
    # KONIEC
    # ========================================================

    print("\n====================================")
    print("TRENING ZAKOŃCZONY")
    print("====================================")

    final_accuracy = evaluate_model(

        model,

        test_loader,

        device
    )

    print(f"\nNajlepsza dokladnosc: {best_accuracy:.2f}%")
    print("\nModel zapisany jako:")
    print("fashion_cnn_model.pth")


# ============================================================
# START PROGRAMU
# ============================================================

if __name__ == "__main__":

    main()


# ============================================================
# INSTALACJA
# ============================================================
#
# pip install torch torchvision numpy
#
# ============================================================
# OCZEKIWANE WYNIKI
# ============================================================
#
# CPU:
# około 92-94%
#
# GPU:
# około 93-95%
#
# ============================================================
# MOŻLIWE ULEPSZENIA
# ============================================================
#
# 1. Więcej warstw Conv2D
#
# 2. Data Augmentation
#
# 3. Batch Normalization
#
# 4. Scheduler Learning Rate
#
# 5. Early Stopping
#
# ============================================================