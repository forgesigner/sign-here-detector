import sys
import time
import wandb

sys.path.append('/root/forger/signheredetector')
sys.path.append('/root/forger/')
from signheredetectordataset import SignatureDataset
import argparse
import os.path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from model import SignatureCenterNet


def min_distance_metric(predicted_heatmaps, true_centers):
    if isinstance(predicted_heatmaps, torch.Tensor):
        if predicted_heatmaps.requires_grad:
            predicted_heatmaps = predicted_heatmaps.detach().numpy()
        else:
            predicted_heatmaps = predicted_heatmaps.cpu().numpy()

    zipped_heatmaps = zip(predicted_heatmaps, true_centers)

    min_distances = []
    for heatmap, centers in zipped_heatmaps:
        # k = len(centers)
        flat_indices = np.argpartition(heatmap[0].flatten(), -1)[-1:]
        top_k_indices = flat_indices[np.argsort(-heatmap[0].flatten()[flat_indices])]
        top_k_coords = [np.unravel_index(idx, heatmap[0].shape) for idx in top_k_indices]
        top_k_coords = [(y, x) for x, y in top_k_coords]
        # print(top_k_coords)
        heatmap_min_dist = []
        for center in centers:
            distance = np.linalg.norm(np.array(center) - np.array(top_k_coords[0]))
            heatmap_min_dist.append(distance)

        min_distances.append(np.min(heatmap_min_dist))

    return np.mean(min_distances)


def extract_true_centers_from_heatmaps(heatmaps):
    if isinstance(heatmaps, torch.Tensor):
        heatmaps = heatmaps.cpu().numpy()

    true_centers = []
    for heatmap in heatmaps:
        max_val = np.max(heatmap[0])
        max_positions = np.argwhere(heatmap[0] == max_val)
        true_centers.append([])
        for pos in max_positions:
            true_centers[-1].append((pos[1], pos[0]))

    return true_centers


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

    wandb.init(
        # set the wandb project where this run will be logged
        project="forger",

        # track hyperparameters and run metadata
        config={
            "learning_rate": learning_rate,
            "model": "center",
            "dataset": "CUAD",
            "epochs": num_epochs,
            "batch_size": batch_size
        }
    )

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        total_min_train_distance = 0
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
            total_train_loss += loss.item()
            true_centers = extract_true_centers_from_heatmaps(heatmaps)
            min_distance = min_distance_metric(outputs, true_centers)
            total_min_train_distance += min_distance

        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_min = total_min_train_distance / len(train_loader)
        wandb.log({"train loss": avg_train_loss,
                   "train min distance": avg_train_min})

        model.eval()
        with torch.no_grad():
            total_val_loss = 0
            total_min_distance = 0
            for images, heatmaps in val_loader:
                images = images.to(device)
                heatmaps = heatmaps.to(device)
                outputs = model(images)
                loss = criterion(outputs, heatmaps)
                total_val_loss += loss.item()
                true_centers = extract_true_centers_from_heatmaps(heatmaps)
                min_distance = min_distance_metric(outputs, true_centers)
                total_min_distance += min_distance

            avg_val_loss = total_val_loss / len(val_loader)
            avg_min_distance = total_min_distance / len(val_loader)
            print(
                f"Validation Loss after Epoch [{epoch + 1}/{num_epochs}]: {avg_val_loss}"
            )

        checkpoint = {
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        wandb.log({"eval loss": avg_val_loss,
                   "eval min distance": avg_min_distance})
        torch.save(
            checkpoint, os.path.join(checkpoint_path, f"checkpoint_{epoch + 1}.pth")
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(checkpoint, os.path.join(checkpoint_path, "best_checkpoint.pth"))
            dummy_input = torch.randn(1, 3, 936, 662, device=device)
            torch.onnx.export(model, dummy_input, os.path.join(checkpoint_path, "best_checkpoint.onnx"),
                              operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK, opset_version=11)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--model", type=str, default="center")
    parser.add_argument("--checkpoint_path", type=str, default="/root/forger/checkpoints")
    args = parser.parse_args()
    if args.model == "center":
        train_center_net(
            os.path.join(args.checkpoint_path,
                         f'train_model_{args.model}_lr_{args.learning_rate}_bs_{args.batch_size}_epochs_{args.num_epochs}_{time.strftime("%Y%m%d-%H%M%S")}'
                         ), args.learning_rate, args.batch_size,
            args.num_epochs
        )
    elif args.model == "unet":
        print("Not implemented yet")
    else:
        print(f"Unknown model: {args.model}")


if __name__ == "__main__":
    main()
