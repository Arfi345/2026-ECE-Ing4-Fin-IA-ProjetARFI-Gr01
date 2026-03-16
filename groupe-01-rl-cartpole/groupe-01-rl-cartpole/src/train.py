import os
import json
import gymnasium as gym
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor

# dossiers
BASE_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# callback simple pour enregistrer les rewards
class RewardCallback:
    def __init__(self):
        self.rewards = []

    def __call__(self, locals_, globals_):
        infos = locals_.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.rewards.append(info["episode"]["r"])
        return True

def train_model(algo_name, env_id="CartPole-v1", timesteps=50000):
    env = gym.make(env_id)
    env = Monitor(env)

    callback = RewardCallback()

    if algo_name == "DQN":
        model = DQN("MlpPolicy", env, verbose=1)
    elif algo_name == "PPO":
        model = PPO("MlpPolicy", env, verbose=1)
    else:
        raise ValueError("Algorithme non supporté")

    model.learn(
        total_timesteps=timesteps,
        callback=callback
    )

    model_path = os.path.join(MODELS_DIR, f"{algo_name.lower()}_cartpole")
    rewards_path = os.path.join(RESULTS_DIR, f"{algo_name}_rewards.json")

    model.save(model_path)

    with open(rewards_path, "w") as f:
        json.dump(callback.rewards, f)

    env.close()
    print(f"{algo_name} terminé")

if __name__ == "__main__":
    train_model("DQN", timesteps=50000)
    train_model("PPO", timesteps=50000)
    print("Entraînement terminé")