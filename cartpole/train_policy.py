import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
# from cartpole.dataset import CartPoleDataset
from cartpole.models import MLP

INPUT_SIZE = 4
HIDDEN_SIZE = 64
OUTPUT_SIZE = 2
LEARNING_RATE = 0.001
BATCH_SIZE = 32
NUM_EPOCHS = 10

class CartPoleDataset(Dataset):
    def __init__(self, episode_dir):
        self.episodes = []
        for episode_file in episode_dir.glob("episode_*.npz"):
            episode_data = np.load(episode_file)
            observations = episode_data['observations']
            tree_decisions = episode_data['tree_decisions']
            for obs, decision in zip(observations, tree_decisions):
                self.episodes.append((obs, decision))
        print(f"Initialized CartPoleDataset with {len(self)} samples")

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        observation, tree_decision = self.episodes[idx]
        return torch.FloatTensor(observation), torch.LongTensor([tree_decision])


def train_policy(episode_dir, checkpoint_path):
    device = torch.device("cpu")
    model = MLP(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(device)
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
        for obs, tree_decisions in tqdm(dataloader, leave=False):
            obs, tree_decisions = obs.to(device), tree_decisions.to(device).squeeze(-1)
            outputs = model(obs)
            loss = F.cross_entropy(outputs, tree_decisions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == tree_decisions).sum().item()
            total_samples += tree_decisions.size(0)

        # Logging
        torch.save(model.state_dict(), checkpoint_path)
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_samples
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train CartPole model")
    parser.add_argument('-i', '--input', type=str, default="episodes", help="Directory containing episode files")
    parser.add_argument('-o', '--output', type=str, default="policy.ckpt", help="Path to save the model checkpoint")
    args = parser.parse_args()

    episode_dir = Path(args.input)
    checkpoint_path = Path(args.output)
    train_policy(episode_dir, checkpoint_path)