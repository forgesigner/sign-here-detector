from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os


class SignatureDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.image_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.directory, self.image_files[idx])
        image = Image.open(img_name).convert('L')

        if self.transform:
            image = self.transform(image)

        return image


transform = transforms.Compose([
    transforms.Resize((300, 100)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x / 255.0)
])

dataset = SignatureDataset(directory='/home/nikita_khramov/forger/vae_dataset', transform=transform)

import torch
import torch.nn as nn
import torch.nn.functional as F

import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.input_dim = 300 * 100

        self.fc1 = nn.Linear(self.input_dim, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)

        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, self.input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def loss_function(recon_x, x, mu, logvar):
    recon_x_flat = recon_x.view(-1, 30000)
    x_flat = x.view(-1, 30000)

    if x_flat.max() > 1 or x_flat.min() < 0:
        raise ValueError("Target data out of range. Should be in [0, 1]")

    BCE = F.binary_cross_entropy(recon_x_flat, x_flat, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(model, data_loader, optimizer, device, epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(data_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print(
                f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(data_loader.dataset)} ({100. * batch_idx / len(data_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}')

    print(f'====> Epoch: {epoch} Average loss: {train_loss / len(data_loader.dataset):.4f}')


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    model = VAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    epochs = 20
    log_interval = 10

    for epoch in range(1, epochs + 1):
        train(model, train_loader, optimizer, device, epoch)
