import torch
import torch.optim as optim
import gymnasium as gym

from policy import Policy
from reinforce import reinforce
from evaluate import evaluate_agent

from hub import push_to_hub

device = torch.device("cuda:0" if torch.cuda.is_available() else "mps")
print("Device: ", device)

from gym_pygame.envs.pixelcopter import PixelcopterEnv  # Import PixelcopterEnv


env_id = "Pixelcopter-PLE-v0"
env = gym.make(env_id)
eval_env = gym.make(env_id)

s_size = env.observation_space.shape[0]
a_size = env.action_space.n

print("_____OBSERVATION SPACE_____ \n")
print("The State Space is: ", s_size)
print("Sample observation", env.observation_space.sample())

print("_____OBSERVATION SPACE_____ \n")
print("The Action Space is: ", s_size)
print("Sample observation", env.action_space.sample())

pixelcopter_hyperparameters = {
    "h_size": 64,
    "n_training_episodes": 100,
    "n_evaluation_episodes": 10,
    "max_t": 10000,
    "gamma": 0.99,
    "lr": 1e-4,
    "env_id": env_id,
    "state_space": int(s_size),
    "action_space": int(a_size),
}

pixelcopter_policy = Policy(device, pixelcopter_hyperparameters["state_space"],
                            pixelcopter_hyperparameters["action_space"],
                            pixelcopter_hyperparameters["h_size"]).to(device)
pixelcopter_optimizer = optim.Adam(pixelcopter_policy .parameters(),
                                   lr=pixelcopter_hyperparameters["lr"])

scores = reinforce(
    env,
    pixelcopter_policy,
    pixelcopter_optimizer,
    pixelcopter_hyperparameters["n_training_episodes"],
    pixelcopter_hyperparameters["max_t"],
    pixelcopter_hyperparameters["gamma"],
    1000,
)

mean_reward, std_reward = evaluate_agent(eval_env,
                                         pixelcopter_hyperparameters["max_t"],
                                         pixelcopter_hyperparameters["n_evaluation_episodes"],
                                         pixelcopter_policy)

print("Mean reward", mean_reward, std_reward)

print("Pixelcopter task solved!")
repo_id = "pkalkman/reinforce-pixelcopter-ple-v1"
push_to_hub(env,
            env_id,
            repo_id,
            pixelcopter_policy,
            pixelcopter_hyperparameters,
            eval_env,
            video_fps=30)
