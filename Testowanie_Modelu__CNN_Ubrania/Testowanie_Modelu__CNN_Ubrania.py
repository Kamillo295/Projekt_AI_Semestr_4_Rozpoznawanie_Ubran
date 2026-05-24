import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

classes = [

    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]


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


device = torch.device(

    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"\nUrządzenie: {device}")


transform = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize(
        (0.5,),
        (0.5,)
    )
])


test_dataset = torchvision.datasets.FashionMNIST(

    root="./data",

    train=False,

    download=True,

    transform=transform
)


model = FashionCNN().to(device)

model.load_state_dict(

    torch.load(
        "fashion_cnn_model.pth",
        map_location=device
    )
)

model.eval()

print("\nModel został wczytany!\n")


while True:

    random_index = torch.randint(

        0,

        len(test_dataset),

        (1,)
    ).item()

    image, label = test_dataset[random_index]

    image_for_model = image.unsqueeze(0).to(device)

    with torch.no_grad():

        outputs = model(image_for_model)

        _, predicted = torch.max(outputs, 1)

    predicted_class = classes[predicted.item()]

    real_class = classes[label]

    print("===================================")

    print(f"Prawdziwa klasa: {real_class}")

    print(f"Model przewidział: {predicted_class}")

    image_to_show = image * 0.5 + 0.5

    plt.imshow(

        image_to_show.squeeze(),

        cmap="gray"
    )

    plt.title(

        f"Predicted: {predicted_class}"
    )

    plt.axis("off")

    plt.show()

    answer = input(

        "\nPokazać kolejny obrazek? (t/n): "
    )

    if answer.lower() != "t":

        break


print("\nProgram zakończony.")