import numpy as np
import torch
from typing import Optional

from env.env_mm import MarketMakingEnv
from agents.dqn import DQNAgent

def train_dqn(
    num_episodes: int = 500,
    max_steps_per_episode: Optional[int] = None,
    seed: int = 42
):
    env = MarketMakingEnv(episode_length = 200)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n   

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        gamma=0.99,
        lr=1e-3,
        batch_size=64,
        buffer_capacity=100_000,
        min_buffer_size=2_000,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=50_000,
        target_update_freq=1_000,
    )
    
    returns = []

    for episode in range(num_episodes):
        obs, _ = env.reset(seed=seed + episode) # at the start of each episode reseting
        done = False
        ep_return = 0.0
        steps = 0
        
        while not done:
            action = agent.select_action(obs, train=True)
            next_obs, reward, done, info = env.step(action)

            agent.store_transition(obs, action, reward, next_obs, done)
            agent.update()

            obs = next_obs
            ep_return += reward
            steps += 1

            if max_steps_per_episode is not None and steps >= max_steps_per_episode: # when reaching the last step in a episode, break
                break
        returns.append(ep_return)
        avg_return = np.mean(returns[-20:])

        print(
            f"Episode {episode+1}/{num_episodes} | "
            f"Return: {ep_return:.2f} | "
            f"Avg(20): {avg_return:.2f} | "
            f"Epsilon: {agent.epsilon:.3f}"
        )
    env.close()
    return agent, returns

if __name__ == "__main__":
    agent, returns = train_dqn(num_episodes=300)

    import numpy as np, os, torch

    project_dir = os.path.dirname(__file__)

    # Save training returns
    np.save(os.path.join(project_dir, "returns.npy"), np.array(returns))

    # Save the trained Q-network weights
    model_path = os.path.join(project_dir, "dqn_mm.pt")
    torch.save(agent.q_net.state_dict(), model_path)
    print("Saved trained model to:", model_path)