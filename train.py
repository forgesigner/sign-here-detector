import sys
import time

sys.path.append('...')
from signheredetectordataset import SignatureDataset
import argparse
import os.path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import SignatureCenterNet


def train_center_net(
        checkpoint_path, learning_rate=0.001, batch_size=32, num_epochs=10
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    model = SignatureCenterNet().to(device)
    train_dataset = SignatureDataset(train=True)
    val_dataset = SignatureDataset(train=False)
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        for i, (images, heatmaps) in enumerate(train_loader):
            images = images.to(device)
            heatmaps = heatmaps.to(device)
            outputs = model(images)
            loss = criterion(outputs, heatmaps)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item()}"
            )

        model.eval()
        with torch.no_grad():
            total_val_loss = 0
            for images, heatmaps in val_loader:
                images = images.to(device)
                heatmaps = heatmaps.to(device)
                outputs = model(images)
                loss = criterion(outputs, heatmaps)
                total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(val_loader)
            print(
                f"Validation Loss after Epoch [{epoch + 1}/{num_epochs}]: {avg_val_loss}"
            )

        checkpoint = {
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(
            checkpoint, os.path.join(checkpoint_path, f"checkpoint_{epoch + 1}.pth")
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(checkpoint, os.path.join(checkpoint_path, "best_checkpoint.pth"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--model", type=str, default="center")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints")
    args = parser.parse_args()
    if args.model == "center":
        train_center_net(
            os.path.join(args.checkpoint_path, 'train' + str(time.time())), args.learning_rate, args.batch_size,
            args.num_epochs
        )
    elif args.model == "unet":
        print("Not implemented yet")
    else:
        print(f"Unknown model: {args.model}")


if __name__ == "__main__":
    main()
