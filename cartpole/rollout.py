import os
import time
import numpy as np
import gymnasium as gym
import multiprocessing as mp
from tqdm import tqdm

EPSILON = 0.5
NUM_SIMULATIONS = 50
RENDER = os.getenv("RENDER", False)

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

def play_cartpole(i):
    env = gym.make('CartPole-v1', render_mode='human' if RENDER else None)
    observation = env.reset()
    done = False
    trunc = False
    total_reward = 1
    steps = 1
    st = time.time()
    env.step(env.action_space.sample())  # state is a numpy array on the first iter, after the first step it becomes a tuple

    episode_data = []

    while not done and not trunc:
        if np.random.uniform() < EPSILON:
            action = env.action_space.sample()
            tree_decision = None
        else:
            action, tree_decision = search_best_action(env)
        observation, reward, done, trunc, _ = env.step(action)
        total_reward += reward
        steps += 1

        episode_data.append({
            'step': steps,
            'action': action,
            'tree_decision': tree_decision,
            'observation': observation,
            'reward': reward
        })

    env.close()
    et = time.time()

    # Save episode data
    os.makedirs('episodes', exist_ok=True)
    np.savez(f'episodes/episode_{i:02d}.npz', episode_data=episode_data)

    return steps, total_reward, et - st

def run_multiple_episodes(num_episodes=100):
    st = time.time()
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = list(tqdm(pool.imap(play_cartpole, range(num_episodes)), total=num_episodes))
    et = time.time()

    for i, (steps, total_reward, time_taken) in enumerate(results, 1):
        print(f"Episode {i} finished after {steps} steps. Total reward: {total_reward}. Time taken: {time_taken:.2f} seconds.")

    steps, rewards, times = zip(*results)
    print()
    print(f"Results after {num_episodes} episodes:")
    print(f"Avg Steps: {np.mean(steps):.2f}")
    print(f"Avg Reward: {np.mean(rewards):.2f}")
    print(f"Avg Time: {np.mean(times):.2f} seconds")
    print(f"Total time: {et - st:.2f} seconds")

if __name__ == "__main__":
    if RENDER:
        steps, _, _ = play_cartpole(0)
        print(f"Episode finished after {steps} steps.")
    else:
        run_multiple_episodes()