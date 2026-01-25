import argparse
import numpy as np
import gymnasium as gym
import minigrid

from agent import Policy


MINIGRID_ENV = "MiniGrid-Empty-16x16-v0"

def evaluate(env=None, n_episodes=10, render=False):
    agent = Policy(MINIGRID_ENV)
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


def train():
    agent = Policy(MINIGRID_ENV)
    agent.train()
    agent.save()


def main():
    parser = argparse.ArgumentParser(description='Run training and evaluation')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-e', '--evaluate', action='store_true')
    args = parser.parse_args()

    if args.train:
        train()
    if args.evaluate:
        evaluate(render=args.render)
    
    #train()
    #evaluate()

    
if __name__ == '__main__':
    main()
