import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import SAC
from env_wrapper import make_racing_env

def record_race():
    # Initialize environment with rgb_array rendering for video capture
    env = make_racing_env(render_mode="rgb_array")
    env = RecordVideo(
        env,
        video_folder="./videos/",
        name_prefix="autonomous_racer_sac",
        episode_trigger=lambda x: True,  # Record every episode
    )

    model = SAC.load("./models/sac_best/best_model", env=env)

    obs, _ = env.reset()
    done = truncated = False
    total_reward = 0.0

    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = env.step(action)
        total_reward += reward

    env.close()
    print(f"Evaluation Episode Reward: {total_reward:.2f}")
    print("Video saved to ./videos/ directory!")

if __name__ == "__main__":
    record_race()