import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
from sac import SAC
from tqdm.auto import tqdm
import apple_gym.env
import pickle

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('-e', '--env-name', default="ApplePick-v0",
                    help='Mujoco Gym environment (default: ApplePick-v0)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
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
parser.add_argument('--load-actor', type=str, help='e.g. models/actor_2021-01-02_10-26-23_SAC_ApplePick-v0_Gaussian_autotune.pkl')
parser.add_argument('--load-critic', type=str, help='e.g. models/critic_2021-01-02_10-26-23_SAC_ApplePick-v0_Gaussian_autotune.pkl')
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Environment
# env = NormalizedActions(gym.make(args.env_name))
env = gym.make(args.env_name, render=args.render)
env.seed(args.seed)
env.action_space.seed(args.seed)


# Agent
agent = SAC(env.observation_space.shape[0], env.action_space, args)
agent.load_model(args.load_actor, args.load_critic)

# Test
avg_reward = 0.
episodes = 10
for _  in tqdm(range(episodes)):
    state = env.reset()
    episode_reward = 0
    done = False
    while not done:
        action = agent.select_action(state, evaluate=True)

        next_state, reward, done, _ = env.step(action)
        episode_reward += reward


        state = next_state
    avg_reward += episode_reward
avg_reward /= episodes

print("----------------------------------------")
print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
print("----------------------------------------")

env.close()
