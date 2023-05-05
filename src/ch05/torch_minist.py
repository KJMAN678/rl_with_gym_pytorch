import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms


class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 192)
        self.fc2 = nn.Linear(192, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

    def predict(self, x):
        """softmaxにより分類問題を予測"""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)


def view_classification(img, probs):
    """数値の分類予測の確率を表示する関数"""
    probs = probs.data.numpy().squeeze()
    fig, (ax1, ax2) = plt.subplots(figsize=(6, 7), ncols=2)
    ax1.imshow(img.numpy().squeeze())
    ax1.axis("off")
    ax2.barh(np.arange(10), probs)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10).astype(int), size="large")
    ax2.set_title("Probability")
    ax2.set_xlim(0, 1.1)
    plt.show()


def main():
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]
    )

    MNIST_data_path = "src/ch05/MNIST_data/"

    trainset = datasets.MNIST(
        MNIST_data_path, download=True, train=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

    testset = datasets.MNIST(
        MNIST_data_path, download=True, train=False, transform=transform
    )
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True)

    data_iter = iter(trainloader)
    images, labels = next(data_iter)

    plt.title(str(images.size()))
    plt.imshow(images[10].numpy().squeeze(), cmap="Greys_r")
    plt.show()

    model = NN()

    data_iter = iter(trainloader)
    images, labels = next(data_iter)
    images.resize_(128, 784)

    img_idx = 0
    logits = model.forward(images)

    prediction = F.softmax(logits, dim=1)

    img = images[0].data
    view_classification(img.view(1, 28, 28), prediction[0])

    # 逆伝播の実行
    model = NN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    epochs = 1
    step = 0
    running_loss = 0
    eval_freq = 10

    for e in range(epochs):
        for images, labels in iter(trainloader):
            step += 1
            images.resize_(images.size()[0], 784)

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            output = model.forward(images)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if step % eval_freq == 0:
                accuracy = 0
                for ii, (images, labels) in enumerate(testloader):
                    images = images.resize_(images.size()[0], 784)
                    predicted = model.predict(images).data
                    equality = labels == predicted.max(1)[1]
                    accuracy += equality.type_as(torch.FloatTensor()).mean()

                print(
                    f"Epoch: {e+1}/{epochs}",
                    f"Loss: {running_loss/eval_freq:.4f}",
                    f"Test accuracy: {accuracy/(ii+1):.4f}",
                )
                running_loss = 0

    logits = model.forward(img[None,]).to("cpu")

    prediction = F.softmax(logits, dim=1)

    view_classification(img.reshape(1, 28, 28), prediction[0])


if __name__ == "__main__":
    time_start = time.time()
    main()
    time_end = time.time()
    elasped_time = time_end - time_start
    print(elasped_time)
