"""Evaluate rideshare policies on testing configurations."""
from typing import List
import warnings
import argparse
import os
import torch
import logging
import pickle
from datetime import datetime, UTC

from free_range_zoo.envs import rideshare_v0
from free_range_zoo.wrappers.action_task import action_mapping_wrapper_v0

warnings.simplefilter('ignore', UserWarning)

FORMAT_STRING = "[%(asctime)s] [%(levelname)s] [%(name)s] [%(filename)21s:%(lineno)03d] %(message)s"
LOG_DATETIME = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def main() -> None:
    """Run the training experiment."""
    global device, args, dataset
    args = handle_args()
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

    if args.threads > 1:
        torch.set_num_threads(args.threads)

    os.makedirs(args.output, exist_ok=True)

    _setup_logger('main')
    _setup_logger('test')

    if os.path.exists(args.output):
        main_logger.warning(f'Output directory {args.output} already exists and may contain artifacts from a previous run.')

    main_logger.info(f'Running the baseline experiment on device {device} and parameters:')
    for key, value in vars(args).items():
        main_logger.info(f'- {key}: {value}')

    torch.use_deterministic_algorithms(True, warn_only=True)
    generator = torch.Generator()
    generator = generator.manual_seed(args.seed)
    torch.manual_seed(torch.randint(0, 100000000, (1, ), generator=generator).item())

    try:
        main_logger.info('TESTING')
        test()
        main_logger.info('DONE TESTING')
    except KeyboardInterrupt:
        main_logger.warning('Testing interrupted by user')
    except Exception as e:
        main_logger.error(f'Error during testing: {e}')
        raise e


@torch.no_grad()
def test() -> None:
    """
    Run the testing episodes for the model.

    Args:
        model: nn.Module - The model to validate.
    """
    with open(args.config, mode='rb') as config_file:
        configuration = pickle.load(config_file)

    env = rideshare_v0.parallel_env(
        parallel_envs=args.testing_episodes,
        max_steps=args.steps,
        configuration=configuration,
        device=device,
    )

    env = action_mapping_wrapper_v0(env)
    observation, _ = env.reset(seed=args.seed)

    agents = {}
    for agent_name in env.agents:
        agents[agent_name] = None  # TODO: Replace this with an initialization of your agent.

    step = 0
    total_rewards = {agent: torch.zeros(args.testing_episodes) for agent in agents}
    while not torch.all(env.finished):
        test_logger.info(f'STEP {step}')
        agent_actions = {}
        for agent_name, agent_model in agents.items():
            agent_model.observe(observation[agent_name])

            actions = agent_model.act(env.action_space(agent_name))
            actions = torch.tensor(actions, device=device, dtype=torch.int32)
            agent_actions[agent_name] = actions

        observation, reward, term, trunc, info = env.step(agent_actions)

        test_logger.info('ACTIONS')
        for batch in range(args.testing_episodes):
            batch_actions = ' '.join(f'{agent_name}: {str(agent_actions[batch].tolist()):<10}\t'
                                     for agent_name, agent_actions in agent_actions.items())
            test_logger.info(f'{batch + 1}:\t{batch_actions}')

        test_logger.info('REWARDS')
        for agent_name in env.agents:
            test_logger.info(f'{agent_name}: {reward[agent_name]}')
            total_rewards[agent_name] += reward[agent_name]

        step += 1

    test_logger.info('TOTAL REWARDS')
    for agent_name, total_reward in total_rewards.items():
        test_logger.info(f'{agent_name}: {total_reward}')

    totals = torch.zeros(args.testing_episodes)
    for agent, reward_tensor in total_rewards.items():
        totals += reward_tensor

    total_mean = round(totals.mean().item(), 3)
    total_std_dev = round(totals.std().item(), 3)

    average_mean = round(total_mean / len(agents), 3)
    average_std_dev = round(total_std_dev / len(agents), 3)

    test_logger.info('REWARD SUMMARY')
    test_logger.info(f'Average Reward: {average_mean} ± {average_std_dev}')
    test_logger.info(f'Total Reward: {total_mean} ± {total_std_dev}')


def _setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    log_format = logging.Formatter(FORMAT_STRING)

    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_format)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(os.path.join(args.output, f'{logger.name}_{LOG_DATETIME}.log'))
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)


def _round_tensor(x: torch.FloatTensor, precision: int = 3) -> List[float]:
    return list(map(lambda x: round(x.item(), precision), x))


def handle_args() -> argparse.Namespace:
    """
    Handle script arguments.

    Returns:
        argparse.Namespace - parsed command-line arguments
    """
    parser = argparse.ArgumentParser(description='Run baseline policies on a given wildfire configuration.')

    general = parser.add_argument_group('General')
    general.add_argument('output', type=str, help='output directory for all experiments artifacts')
    general.add_argument('config', type=str, help='path to environment configuration to utilize')
    general.add_argument('--cuda', action='store_true', help='Utilize cuda if available')
    general.add_argument('--threads', type=int, default=1, help='utilize this many threads for the experiment')

    reproducible = parser.add_argument_group('Reproducibility')
    reproducible.add_argument('--seed', type=int, default=0, help='seed for the experiment')

    validation = parser.add_argument_group('Validation')
    validation.add_argument('--testing_episodes', type=int, default=1, help='number of episodes to run per test')

    return parser.parse_args()


main_logger = logging.getLogger('main')
test_logger = logging.getLogger('test')

if __name__ == '__main__':
    main()
