import imageio
import numpy as np
import gymnasium as gym


def record_video(env_id, policy, out_directory, fps=30):
    """
    Generate a replay video of the agent
    :param env
    :param Qtable: Qtable of our agent
    :param out_directory
    :param fps: how many frame per seconds (with taxi-v3 and frozenlake-v1 we use 1)
    """
    images = []
    done = False
    env = gym.make(env_id)
    state, _ = env.reset()
    img = env.unwrapped.render(mode='rgb_array')
    images.append(img)
    max_steps = 0
    while not done and max_steps < 60 * 60:
        # Take the action (index) that have the maximum expected future reward given that state
        action, _ = policy.act(state)
        state, reward, done, info, _ = env.step(action)
        max_steps += 1
        img = env.unwrapped.render(mode='rgb_array')
        images.append(img)
    imageio.mimsave(out_directory, [np.array(img) for i, img in enumerate(images)], fps=fps)
