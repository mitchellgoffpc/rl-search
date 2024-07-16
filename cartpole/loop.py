import argparse
from pathlib import Path
from cartpole.train import train
from cartpole.rollout import run_multiple_episodes

EPSILONS = [1.0, 0.5, 0.0]

def main(num_episodes, num_simulations):
    for i, epsilon in enumerate(EPSILONS):
        print(f"\n--- Iteration {i+1} ---")
        model_path = Path(__file__).parent / f'model-{i:01d}.ckpt' if i > 0 else None
        episodes_dir = Path(__file__).parent / f'episodes-{i:01d}'
        episodes_dir.mkdir(exist_ok=True)

        print(f"Running rollouts with epsilon={epsilon}")
        run_multiple_episodes(model_path, episodes_dir, num_episodes if i > 0 else 1000, num_simulations, epsilon)

        print("\nTraining model")
        new_model_path = Path(__file__).parent / f'model-{i+1:01d}.ckpt'
        train(episodes_dir, new_model_path)

    print("Running final rollouts with epsilon=0.0")
    run_multiple_episodes(new_model_path, None, num_episodes, num_simulations, 0.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run CartPole rollouts and training iterations.')
    parser.add_argument('-n', '--num_episodes', type=int, default=100, help='Number of episodes to run in each rollout')
    parser.add_argument('-s', '--num_simulations', type=int, default=50, help='Number of simulations to run for each step')
    args = parser.parse_args()

    main(args.num_episodes, args.num_simulations)
