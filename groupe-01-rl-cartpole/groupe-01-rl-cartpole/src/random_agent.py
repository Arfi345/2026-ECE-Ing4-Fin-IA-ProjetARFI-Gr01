import os
import json
import gymnasium as gym
import random

BASE_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

env = gym.make("CartPole-v1")

episodes = 50
rewards = []

for episode in range(episodes):
    obs, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = random.choice([0, 1])
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

    rewards.append(total_reward)
    print(f"Episode {episode + 1}: reward = {total_reward}")

env.close()

with open(os.path.join(RESULTS_DIR, "random_rewards.json"), "w") as f:
    json.dump(rewards, f)

print("Random agent terminé")