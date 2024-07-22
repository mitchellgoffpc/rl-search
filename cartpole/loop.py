import argparse
from pathlib import Path
from cartpole.rollout import run_multiple_episodes
from cartpole.train_policy import train_policy

EPSILONS = [1.0, 0.5, 0.0]

def main(num_episodes, num_simulations):
    for i, epsilon in enumerate(EPSILONS):
        print(f"\n--- Iteration {i+1} ---")
        policy_path = Path(__file__).parent / f'policy-{i:01d}.ckpt' if i > 0 else None
        episodes_dir = Path(__file__).parent / f'episodes-{i:01d}'
        episodes_dir.mkdir(exist_ok=True)

        print(f"Running rollouts with epsilon={epsilon}")
        run_multiple_episodes(policy_path, None, episodes_dir, num_episodes if i > 0 else 1000, num_simulations, float('inf'), epsilon)

        print("\nTraining policy model")
        new_policy_path = Path(__file__).parent / f'policy-{i+1:01d}.ckpt'
        train_policy(episodes_dir, new_policy_path)

    print(f"\n--- Final Iteration ---")
    print("Running rollouts with epsilon=0.0")
    run_multiple_episodes(new_policy_path, None, None, num_episodes, num_simulations, float('inf'), 0.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run CartPole rollouts and training iterations.')
    parser.add_argument('-n', '--num_episodes', type=int, default=100, help='Number of episodes to run in each rollout')
    parser.add_argument('-s', '--num_simulations', type=int, default=50, help='Number of simulations to run for each step')
    args = parser.parse_args()

    main(args.num_episodes, args.num_simulations)
