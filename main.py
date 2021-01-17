import argparse
import datetime
import gym
import numpy as np
import itertools
from pathlib import Path
import logging
import torch
from sac import SAC
from replay_memory import ReplayMemory
from load_demonstrations import load_demonstrations
import apple_gym.env
import pickle
from process_obs import ProcessObservation
# from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter

from progress import RichTQDM
from loguru import logger
from rich import print
from rich.logging import RichHandler
logging.basicConfig(level=logging.INFO, handlers=[RichHandler(rich_tracebacks=True, markup=True)])
logger.configure(handlers=[{"sink": RichHandler(markup=True),
                         "format": "{message}"}])

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
    parser.add_argument('-e', '--env-name', default="ApplePick-v0",
                        help='Mujoco Gym environment (default: ApplePick-v0)')
    parser.add_argument('--policy', default="Gaussian",
                        help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--eval', type=bool, default=True,
                        help='Evaluates a policy a policy every 10 episode (default: True)')
    parser.add_argument('--no-train', dest='train', action='store_false')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                                term against the reward (default: 0.2)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                        help='Automaically adjust α (default: True)')
    parser.add_argument('--seed', type=int, default=123456, metavar='N',
                        help='random seed (default: 123456)')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='batch size (default: 256)')
    parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                        help='maximum number of steps (default: 1000000)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                        help='model updates per simulator step (default: 1)')
    parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                        help='Steps sampling random actions (default: 10000)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                        help='size of replay buffer (default: 10000000)')
    parser.add_argument('--cuda', action="store_true",
                        help='run on CUDA (default: False)')
    parser.add_argument('--demonstrations', default=False, 
                        help='Load demonstrations from https://github.com/erfanMhi/gym-recording-modified')
    parser.add_argument('-l', '--load', default=False, 
                        help='Load models')
    parser.add_argument('-r', '--render', action="store_true",
                        help='show')
    args = parser.parse_args()
    return args


args = get_args()
logger.info(f'args {args}')

# Environment
# env = NormalizedActions(gym.make(args.env_name))
env = gym.make(args.env_name, render=args.render)
env.seed(args.seed)
env.action_space.seed(args.seed)
keys_to_monitor = ['env_reward/apple_pick/tree/min_fruit_dist_reward',
    'env_reward/apple_pick/tree/gripping_fruit_reward',
    'env_reward/apple_pick/tree/force_tree_reward',
    'env_reward/apple_pick/tree/force_fruit_reward', 'env_obs/apple_pick/tree/picks']

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# A visual network
action_dim = env.action_space.shape[0]
observation_dim=env.observation_space.shape[0] 
process_obs=ProcessObservation()
observation_dim=observation_dim - process_obs.reduce_obs_space

logger.info(f"process_obs reduces obs_space {env.observation_space.shape[0]}-{process_obs.reduce_obs_space}={observation_dim}")

# Agent
agent = SAC(observation_dim, env.action_space, args, process_obs)

# TODO
# summary(model, input_size=(batch_size, 1, 28, 28))

#Tensorboard
log_name = '{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
                                                             args.policy, "autotune" if args.automatic_entropy_tuning else "")
writer=SummaryWriter('runs/' + log_name)
logger.info(f"log name {log_name}")

save_dir=Path("models") / log_name

# Memory
memory=ReplayMemory(args.replay_size, args.seed, env.observation_space.shape[0], action_dim)


def save(save_dir):
    try:
        save_dir.mkdir(exist_ok=True)
        logger.info(f'Saving to {save_dir}')
        agent.save_model(save_dir/'actor.pkl', save_dir/'critic.pkl')
        # memory.save(save_dir / 'memory.pkl') # crashes at over 200k
    except Exception as e:
        logging.exception("failed to save")

def load(save_dir):
    agent.load_model(save_dir / 'actor.pkl', save_dir / 'critic.pkl')
    # if args.train:
        # memory.load(save_dir/'memory.pkl')

if args.load:
    if args.load=='auto':
        args.load = sorted(Path('models').glob('*/actor*'))[-1].parent
        logger.info(f'auto loading {args.load}')
    load(Path(args.load))
    logger.info(f"memory {len(memory)} after load")

if args.demonstrations:
    load_demonstrations(memory, args.demonstrations)
    logger.info(f"memory {len(memory)} after demonstrations")

# Training Loop
total_numsteps = 0
updates = 0


with RichTQDM() as prog:
    task1 = prog.add_task("[red]steps", total=args.num_steps)
    task2 = prog.add_task("[red]updates", total=args.num_steps)
    task3 = prog.add_task("[red]test", total=args.num_steps)
    for i_episode in itertools.count(0):
        print('1')
        episode_reward = 0
        episode_steps = 0
        done = False
        state = env.reset()

        while (not done) and args.train:
            if args.start_steps > total_numsteps:
                action = env.action_space.sample()  # Sample random action
            else:
                action = agent.select_action(state)  # Sample action from policy

            if len(memory) > args.batch_size:
                # Number of updates per step in environment
                for i in range(args.updates_per_step):
                    # Update parameters of all the networks
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)

                    writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                    writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                    writer.add_scalar('loss/policy', policy_loss, updates)
                    writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                    writer.add_scalar('entropy_temperature/alpha', alpha, updates)

                    updates += 1
                    prog.update(task2, advance=1)

            next_state, reward, done, info = env.step(action)  # Step
            episode_steps += 1
            total_numsteps += 1
            prog.update(task1, advance=1)
            episode_reward += reward

            # log env stuff
            if total_numsteps == 1:
                logger.info(f'info {info.keys()}')
                logger.debug(f'info {info}')
            for k in keys_to_monitor:
                writer.add_scalar(k, info[k], total_numsteps)

            # Ignore the "done" signal if it comes from hitting the time horizon.  (that is, when it's an artificial terminal signal that isn't based on the agent's state)
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/sac.py)
            mask = 1 if episode_steps == env._max_episode_steps else float(not done)
            memory.push(state, action, reward, next_state, mask)  # Append transition to memory

            state = next_state

        writer.add_scalar('reward/train', episode_reward, i_episode)
        logger.info("\nEpisode: {}, total numsteps: {}, episode steps: {}, reward: {}, updates: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2), updates))
        prog.desc = "e: {}, r: {}, u: {}, m: {}".format(i_episode, round(episode_reward, 2), updates, len(memory))

        if (i_episode % 100 == 0) and (args.eval is True) and i_episode>0:
            avg_reward = 0.
            episodes = 10
            for _  in range(episodes):
                state = env.reset()
                episode_reward = 0
                done = False
                while not done:
                    action = agent.select_action(state, evaluate=True)
                    prog.update(task3, advance=1)

                    next_state, reward, done, _ = env.step(action)
                    episode_reward += reward

                    for k in keys_to_monitor:
                        writer.add_scalar('test/' + k, info[k], total_numsteps)


                    state = next_state
                avg_reward += episode_reward
            avg_reward /= episodes


            writer.add_scalar('avg_reward/test', avg_reward, i_episode)


            logger.info("----------------------------------------")
            logger.info("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
            logger.info("----------------------------------------")

        if total_numsteps >= args.num_steps:
            break

            if args.train:
                save(save_dir)

env.close()
