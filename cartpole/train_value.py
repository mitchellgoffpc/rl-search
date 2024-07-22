import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from cartpole.models import MLP
from cartpole.dataset import CartPoleDataset

INPUT_SIZE = 4
HIDDEN_SIZE = 64
OUTPUT_SIZE = 2
LEARNING_RATE = 0.001
BATCH_SIZE = 32
NUM_EPOCHS = 5

def train_value(episode_dir, checkpoint_path):
    device = torch.device("cpu")
    model = MLP(INPUT_SIZE, HIDDEN_SIZE, 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    dataset = CartPoleDataset(episode_dir)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Training loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0

        # Run an epoch
        for obs, _, values in tqdm(dataloader, leave=False):
            obs, values = obs.to(device), values.to(device) / 20
            outputs = model(obs)
            loss = F.mse_loss(outputs, values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_samples += values.size(0)

        # Logging
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}")

        if checkpoint_path:
            torch.save(model.state_dict(), checkpoint_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train CartPole value function")
    parser.add_argument('-i', '--input', type=str, default="episodes", help="Directory containing episode files")
    parser.add_argument('-o', '--output', type=str, default=None, help="Path to save the model checkpoint")
    args = parser.parse_args()

    episode_dir = Path(args.input)
    checkpoint_path = Path(args.output) if args.output else None
    train_value(episode_dir, checkpoint_path)