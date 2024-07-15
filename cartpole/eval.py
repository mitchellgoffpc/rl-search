import torch
import numpy as np
import gymnasium as gym
from pathlib import Path
from cartpole.train import MLP

def main():
    env = gym.make('CartPole-v1')

    model = MLP(input_size=4, hidden_size=64, output_size=2)
    model.load_state_dict(torch.load(Path(__file__).parent / 'model.ckpt'))
    model.eval()

    num_episodes = 100
    temperature = 1.0
    total_episode_lengths = 0  # Initialize total episode lengths

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        trunc = False
        episode_length = 0

        while not done and not trunc:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action_probs = model(state_tensor)
                action_probs = torch.softmax(action_probs / temperature, dim=1)
                action = torch.multinomial(action_probs, 1).item()

            state, _, done, trunc, _ = env.step(action)
            episode_length += 1

        total_episode_lengths += episode_length
        print(f"Episode {episode + 1} finished with total length: {episode_length}")

    average_episode_length = total_episode_lengths / num_episodes
    print(f"Average episode length: {average_episode_length:.2f}")

if __name__ == "__main__":
    main()