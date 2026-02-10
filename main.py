import argparse
import numpy as np
import gymnasium as gym
import minigrid
from minigrid.wrappers import ActionBonus

from agent import Policy

'''
tested envs:

"MiniGrid-Empty-16x16-v0"
"MiniGrid-LavaGapS7-v0"
"MiniGrid-FourRooms-v0"
"MiniGrid-Dynamic-Obstacles-5x5-v0"
"MiniGrid-Dynamic-Obstacles-Random-6x6-v0"
"MiniGrid-DoorKey-6x6-v0"
"MiniGrid-DoorKey-8x8-v0"
"MiniGrid-LavaCrossingS9N2-v0"
"MiniGrid-DistShift1-v0"
'''

MINIGRID_ENV = "MiniGrid-DistShift1-v0"


def evaluate(env=None, n_episodes=300, render=False, load_best_model=False):
    agent = Policy(MINIGRID_ENV)
    if load_best_model:
        agent.load_based_on_env(MINIGRID_ENV)
    else:
        agent.load()

    env = gym.make(MINIGRID_ENV, render_mode="rgb_array")
    if render:
        env = gym.make(MINIGRID_ENV, render_mode="human")
        
    rewards = []
    for episode in range(n_episodes):
        total_reward = 0
        done = False
        s, _ = env.reset()
        while not done:
            action = agent.act(s)
            
            s, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
        
        rewards.append(total_reward)
        
    print('Mean Reward:', np.mean(rewards))


def evaluate_all(env=None, n_episodes=100, render=False):
    
    env = gym.make(MINIGRID_ENV, render_mode="rgb_array")
    if render:
        env = gym.make(MINIGRID_ENV, render_mode="human")
        
    iters = 10000
    for i in range(0, iters, 250):
        i = str(i)
        agent = Policy(MINIGRID_ENV)
        model_name = 'model_' + i + '.pt'
        agent.load(model_name)
        
        rewards = []
        for episode in range(n_episodes):
            total_reward = 0
            done = False
            s, _ = env.reset()
            while not done:
                action = agent.act(s)
                
                s, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward
            
            rewards.append(total_reward)
            
        print((i) + ' Mean Reward:', np.mean(rewards))



def train():
    agent = Policy(MINIGRID_ENV)
    agent.train()
    agent.save()



def main():
    training = False
    
    if not training:
        parser = argparse.ArgumentParser(description='Run training and evaluation')
        parser.add_argument('--render', action='store_true')
        parser.add_argument('-t', '--train', action='store_true')
        parser.add_argument('-e', '--evaluate', action='store_true')
        parser.add_argument('-el', '--evaluateall', action='store_true')
        parser.add_argument('-l', '--loadbestmodel', action='store_true')
        args = parser.parse_args()

        if args.train:
            train()
            
        if args.evaluateall:
            evaluate_all(render=args.render)
        elif args.evaluate:
            if args.loadbestmodel:
                evaluate(render=args.render, load_best_model=True)
            else:
                evaluate(render=args.render)
    else:
        train()
        #evaluate()

    
if __name__ == '__main__':
    main()
