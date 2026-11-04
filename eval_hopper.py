import os
os.environ.setdefault("MUJOCO_GL", "egl")    # headless GPU rendering for videos
import cv2
import imageio
import numpy as np, torch
import gymnasium as gym, minari
from model import Policy

ds_expert = minari.load_dataset("mujoco/hopper/expert-v0", download=True)
eval_env = ds_expert.recover_environment(eval_env=True, render_mode="rgb_array")   # same Hopper-v5 the data came from


def render_frame(env):
    """ gymnasium resets the offscreen camera to FREE on every render (ignoring trackbodyid), so follow the hopper
    by moving the camera's lookat to the torso x position """
    u = env.unwrapped
    u.mujoco_renderer._get_viewer("rgb_array").cam.lookat[0] = u.data.qpos[0]
    return env.render()


def label_frames(frames, success, ep_ret):
    """ Draws the episode outcome (green SUCCESS / red FAILURE) and the running step count on every frame """
    text, color = ("SUCCESS", (0, 200, 0)) if success else ("FAILURE", (220, 0, 0))   # RGB
    labeled = []
    for t, f in enumerate(frames):
        f = np.ascontiguousarray(f)
        cv2.putText(f, f"{text}  return {ep_ret:.0f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        cv2.putText(f, f"step {t}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        labeled.append(f)
    return labeled


@torch.no_grad()
def evaluate(policy, env, mu=None, sigma=None, n_episodes=10, seed=0, video_dir=None):
    """ video_dir: if given, saves one mp4 per episode there
    An episode is a success if it reaches the time limit without falling (truncated, not terminated) """
    policy.eval()
    returns, lengths, successes = [], [], []
    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed + i)        # fixed seeds -> comparable across methods
        done, ep_ret, ep_len = False, 0.0, 0
        frames = [render_frame(env)] if video_dir else None
        while not done:
            obs_n = obs if mu is None else (obs - mu) / sigma   # SAME normalization as training (train.py uses none)
            act = policy(torch.as_tensor(obs_n, dtype=torch.float32)).numpy()
            obs, r, terminated, truncated, _ = env.step(act)
            ep_ret += r; ep_len += 1
            done = terminated or truncated
            if video_dir:
                frames.append(render_frame(env))
        success = not terminated
        if video_dir:
            os.makedirs(video_dir, exist_ok=True)
            fps = env.metadata.get("render_fps", 30)
            name = f"episode_{i}_{'success' if success else 'failure'}_return_{ep_ret:.0f}.mp4"
            imageio.mimsave(os.path.join(video_dir, name), label_frames(frames, success, ep_ret), fps=fps)
        returns.append(ep_ret); lengths.append(ep_len); successes.append(success)
    policy.train()
    return np.mean(returns), np.std(returns), np.mean(lengths), np.mean(successes)


if __name__ == "__main__":
    obs_dim = ds_expert.observation_space.shape[0]
    act_dim = ds_expert.action_space.shape[0]
    policy = Policy(obs_dim, act_dim)
    ckpt_path = "/home/mahesh_tapeti/il-experiment/checkpoints/exp_1793597565/model_epoch_10.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    policy.load_state_dict(ckpt["model_state_dict"])

    # evaluate the policy
    video_dir = os.path.join(os.path.dirname(ckpt_path), "videos", os.path.splitext(os.path.basename(ckpt_path))[0])
    mean_ret, std_ret, mean_len, success_rate = evaluate(policy, eval_env, n_episodes=10, video_dir=video_dir)
    print(f"Evaluation over 10 episodes: mean return {mean_ret:.1f} +/- {std_ret:.1f}, mean length {mean_len:.1f}, "
          f"success rate {success_rate:.0%}")
    print(f"Videos saved to {video_dir}")
    eval_env.close()
