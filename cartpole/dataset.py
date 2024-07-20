import torch
import numpy as np
from torch.utils.data import Dataset

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