import os
import json
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(BASE_DIR, "results")

def load_rewards(filename):
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "r") as f:
        return json.load(f)

dqn_rewards = load_rewards("DQN_rewards.json")
ppo_rewards = load_rewards("PPO_rewards.json")
random_rewards = load_rewards("random_rewards.json")

plt.plot(dqn_rewards, label="DQN")
plt.plot(ppo_rewards, label="PPO")
plt.plot(random_rewards, label="Random")

plt.title("Comparaison RL sur CartPole")
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.legend()
plt.grid(True)

output_path = os.path.join(RESULTS_DIR, "comparaison_DQN_PPO_random.png")
plt.savefig(output_path)
plt.show()

print("Graphique sauvegardé :", output_path)