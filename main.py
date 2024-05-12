import argparse
import datetime
# import gym
import numpy as np
import itertools
import torch
from sac import SAC
# from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory
# from envs.unicycle_env import UnicycleEnv
from envs.unicycle_env_cs443 import UnicycleEnv
import time
import pandas as pd
import os
import shutil

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="HalfCheetah-v2",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
# False for CS443 project
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0002, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=100000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--max_episodes', type=int, default=1000, metavar='N',
                        help='maximum number of episodes (default: 400)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=5000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
parser.add_argument('--mode', default='train', type=str,
                    help='support option: train/test')
parser.add_argument('--load_model', default='',
                    help='model path')
args = parser.parse_args()

# Environment
# env = NormalizedActions(gym.make(args.env_name))
env = UnicycleEnv()
# env.seed(args.seed)
# env.action_space.seed(args.seed)

# Check if there is the data file
filename = 'data.csv'
backup_dir = "C:\\Users\\Bowen\\Desktop"
if os.path.exists(filename):
    shutil.move(filename, os.path.join(backup_dir, filename))
    
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Agent, it include the initialization of the Q, Q', and policy networks
agent = SAC(env.observation_space.shape[0], env.action_space, args)

if args.mode == 'train':
    #Tesnorboard
    # writer = SummaryWriter('runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
    #                                                             args.policy, "autotune" if args.automatic_entropy_tuning else ""))

    # Memory for Experience Replay
    memory = ReplayMemory(args.replay_size, args.seed)

    # Training Loop
    total_numsteps = 0
    updates = 0
    
    # Train time recording
    accumulate_train_time = 0
    
    # Generate a iteration object to continuous generate numbers
    for i_episode in itertools.count(1):
        output_data = []
        train_start_time = time.time()
        collision = 0
        episode_reward = 0
        episode_steps = 0
        done = False
        state = env.reset()

        while not done:
            # start_steps is the number of steps to take random actions: Get initial data
            if args.start_steps > total_numsteps:
                action = env.action_space.sample()  # Sample random action
            else:
                action = agent.select_action(state)  # Sample action from policy

            # When we get enough data, we start to update the network
            if len(memory) > args.batch_size:
                # Number of updates per step in environment
                for i in range(args.updates_per_step):
                    # Update parameters of all the networks
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)
                    # writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                    # writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                    # writer.add_scalar('loss/policy', policy_loss, updates)
                    # writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                    # writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                    updates += 1
            next_state, reward, done, info = env.step(action) # Step
            if info['cost'] != 0:
                collision += 1
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
            mask = 1 if episode_steps == env.max_episode_steps else float(not done)
            # print("mask: ", done)
            memory.push(state, action, reward, next_state, mask) # Append transition to memory

            state = next_state
            
        train_end_time = time.time()
        episode_duration = train_end_time - train_start_time
        accumulate_train_time += episode_duration  
        
        # Save data to plot
        output_data.append((i_episode, episode_reward, collision, episode_duration, accumulate_train_time))
        df = pd.DataFrame(output_data, columns = ['episode', 'reward', 'collision_time', 'episode_duration', 'accumulate_train_time'])
        header = not os.path.exists('data.csv')
        df.to_csv('data.csv', mode = 'a', index=False, header=header) 
        
        if i_episode > args.max_episodes:
            break
        # writer.add_scalar('reward/train', episode_reward, i_episode)
        print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

        # Every 10 episodes, evaluate the current policy /Just for visulize/
        if i_episode % 20 == 0 and args.eval is True:
            avg_reward = 0.
            episodes = 10
            # env.render_init()
            env.render_flag = True
            for _ in range(episodes):
                # state = env.reset(rand_init=False)
                state = env.reset()
                episode_reward = 0
                done = False
                while not done:               
                    env.render_save()
                    action = agent.select_action(state, evaluate=True)
                    next_state, reward, done, _ = env.step(action)
                    episode_reward += reward
                    state = next_state
                avg_reward += episode_reward
            avg_reward /= episodes
            env.render_flag = False
            # agent.save_checkpoint("unicycle",ckpt_path = 'weight/weight_with_average_reward' + str(int(avg_reward)) + '.pth')
            env.render_activate()   # Only for the last episode
            # writer.add_scalar('avg_reward/test', avg_reward, i_episode)

            print("----------------------------------------")
            print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
            print("----------------------------------------")
else:
    agent.load_checkpoint(args.load_model,evaluate=True)
    test_reward = 0
    env.render_flag = True
    state = env.reset(rand_init=False)
    done = False
    while not done:
        env.render_save()
        action = agent.select_action(state, evaluate=True)
        next_state, reward, done, _ = env.step(action)
        test_reward += reward
        state = next_state
    env.render_activate()
    env.render_flag = False
    print("----------------------------------------")
    print("Reward: {}".format(test_reward))
    print("----------------------------------------")

env.close()

