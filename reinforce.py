import torch
from collections import deque
import numpy as np
import time


def reinforce(env, policy, optimizer, n_training_episodes, max_t, gamma, print_every):
    scores_deque = deque(maxlen=100)
    scores = []
    start_time = time.time()  # Start time for the entire training

    for i_episode in range(1, n_training_episodes + 1):
        saved_log_probs = []
        rewards = []
        state, _ = env.reset()

        for t in range(max_t):
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards.append(reward)
            if done:
                break

        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        returns = deque(maxlen=max_t)
        n_steps = len(rewards)

        for t in range(n_steps)[::-1]:
            disc_return_t = (returns[0] if len(returns) > 0 else 0)
            returns.appendleft(gamma * disc_return_t + rewards[t])

        eps = np.finfo(np.float32).eps.item()
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)

        policy_loss = []
        for log_prob, disc_return in zip(saved_log_probs, returns):
            policy_loss.append(-log_prob * disc_return)
        policy_loss = torch.cat(policy_loss).sum()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        # Logging every print_every episodes
        if i_episode % print_every == 0:
            elapsed_time = time.time() - start_time
            avg_score = np.mean(scores_deque)
            min_score = np.min(scores_deque)
            max_score = np.max(scores_deque)
            print(
                'Episode {}\tAverage Score: {:.2f}\tMin Score: {:.2f}\tMax Score: {:.2f}\tTime: {:.2f} seconds'.format(
                    i_episode, avg_score, min_score, max_score, elapsed_time
                )
            )
            start_time = time.time()  # Reset start time for the next interval

    return scores
