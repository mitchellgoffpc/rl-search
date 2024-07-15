import os
import time
import torch
import argparse
import numpy as np
import gymnasium as gym
import multiprocessing as mp
from tqdm import tqdm
from pathlib import Path
from cartpole.train import MLP

EPSILON = 0.5
NUM_SIMULATIONS = 50
NUM_EPISODES = 100

def run_simulation(env, action):
    done = False
    trunc = False
    total_reward = 0
    steps = 0

    while not done and not trunc:
        observation, reward, done, trunc, _ = env.step(action)
        total_reward += reward
        steps += 1
        action = env.action_space.sample()  # random action for subsequent steps

    return steps

def search_best_action(env):
    save_state_env = gym.make('CartPole-v1')
    action_results = {0: [], 1: []}

    for action in [0, 1]:
        for _ in range(NUM_SIMULATIONS):
            save_state_env.reset()
            save_state_env.unwrapped.state = env.unwrapped.state
            steps = run_simulation(save_state_env, action)
            action_results[action].append(steps)

    avg_steps_0 = np.mean(action_results[0])
    avg_steps_1 = np.mean(action_results[1])

    return 0 if avg_steps_0 > avg_steps_1 else 1, action_results

def play_cartpole(model_path, render=False, save=False):
    model = None
    if model_path:
        model = MLP(input_size=4, output_size=2, hidden_size=64).eval()
        model.load_state_dict(torch.load(model_path))

    env = gym.make('CartPole-v1', render_mode='human' if render else None)
    observation, _ = env.reset()
    env.step(env.action_space.sample())  # state is a numpy array on the first iter, after the first step it becomes a tuple

    done = False
    trunc = False
    total_reward = 1
    steps = 1
    st = time.time()
    episode_data = []

    while not done and not trunc:
        tree_decision = None
        if save or not model:  # only compute the tree decision if we have to
            tree_decision, _ = search_best_action(env)

        if np.random.uniform() < EPSILON:
            action = env.action_space.sample()
        elif model:
            with torch.no_grad():
                # action = model(torch.FloatTensor(observation)).argmax().item()
                action_probs = torch.softmax(model(torch.FloatTensor(observation)), dim=0)
                action = torch.multinomial(action_probs, 1).item()
        else:
            action = tree_decision

        observation, reward, done, trunc, _ = env.step(action)
        total_reward += reward
        steps += 1
        episode_data.append({'step': steps, 'observation': observation, 'action': action, 'tree_decision': tree_decision})

    env.close()
    et = time.time()
    observations, actions, tree_decisions = zip(*[(data['observation'], data['action'], data['tree_decision']) for data in episode_data])

    return steps, observations, actions, tree_decisions, et - st


def _play_cartpole(args):
    return play_cartpole(*args)

def run_multiple_episodes(model_path, episode_dir):
    st = time.time()
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = list(tqdm(pool.imap(_play_cartpole, [(model_path, False, True) for i in range(NUM_EPISODES)]), total=NUM_EPISODES))
    et = time.time()

    if episode_dir is not None:
        for i, (_, observations, actions, tree_decisions, _) in enumerate(results):
            np.savez(episode_dir / f'episode_{i:02d}.npz', observations=observations, actions=actions, tree_decisions=tree_decisions)

    steps, _, _, _, times = zip(*results)
    print(f"Results after {NUM_EPISODES} episodes:")
    print(f"Avg Steps: {np.mean(steps):.2f}")
    print(f"Avg Time: {np.mean(times):.2f} seconds")
    print(f"Total time: {et - st:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run CartPole episodes with optional model.')
    parser.add_argument('-m', '--model', type=str, help='Path to the model checkpoint file')
    parser.add_argument('--render', action='store_true', help='Render the episodes')
    parser.add_argument('--save', action='store_true', help='Save the episodes to disk')
    args = parser.parse_args()

    episode_dir = None
    if args.save:
        episode_dir = Path(__file__).parent / 'episodes'
        episode_dir.mkdir(parents=True, exist_ok=True)
        for f in episode_dir.glob('episode_*.npz'):
            f.unlink()

    if args.render:
        steps, _, _ = play_cartpole(args.model, render=True)
        print(f"Episode finished after {steps} steps.")
    else:
        run_multiple_episodes(args.model, episode_dir)