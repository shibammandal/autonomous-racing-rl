import argparse
import os
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from env_wrapper import make_racing_env

def train(algo: str = "sac", timesteps: int = 50_000):
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    train_env = Monitor(make_racing_env())
    eval_env = Monitor(make_racing_env())

    # Automatically save the highest-scoring model during training
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./models/{algo}_best/",
        log_path="./logs/",
        eval_freq=2500,
        deterministic=True,
        render=False,
    )

    if algo.lower() == "sac":
        model = SAC(
            "MlpPolicy",
            train_env,
            learning_rate=5e-4,
            buffer_size=50_000,
            batch_size=256,
            gamma=0.99,
            verbose=1,
            tensorboard_log="./logs/tensorboard/",
        )
    else:
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=64,
            gamma=0.99,
            verbose=1,
            tensorboard_log="./logs/tensorboard/",
        )

    print(f"Training {algo.upper()} for {timesteps} timesteps...")
    model.learn(total_timesteps=timesteps, callback=eval_callback, tb_log_name=algo.upper())
    model.save(f"models/{algo}_racer_final")
    print(f"Training complete! Best model saved in ./models/{algo}_best/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="sac", choices=["sac", "ppo"])
    parser.add_argument("--steps", type=int, default=50_000)
    args = parser.parse_args()
    train(args.algo, args.steps)