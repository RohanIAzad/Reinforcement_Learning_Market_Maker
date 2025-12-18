
import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from env.env_mm import MarketMakingEnv
from agents.dqn import DQNAgent
from baseline_as import ASHeuristicPolicy

# -----------------------------
# 0. Paths and basic settings
# -----------------------------

MODEL_PATH = os.path.join(os.path.dirname(__file__), "dqn_mm.pt")

NUM_EPISODES = 10
EPISODE_LENGTH = 200

# -----------------------------
# 1. Create an env just to get dims
# -----------------------------

env_for_dims = MarketMakingEnv(episode_length=EPISODE_LENGTH)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found at {MODEL_PATH}. "
        "Run python train_dqn.py first so it saves dqn_mm.pt."
    )

# -----------------------------
# 2. Build a DQN agent and load trained weights
# -----------------------------

state_dim = env_for_dims.observation_space.shape[0]
action_dim = env_for_dims.action_space.n

dqn_agent = DQNAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    gamma=0.99,
    lr=1e-3,
    batch_size=64,
    buffer_capacity=100_000,
    min_buffer_size=2_000,
    epsilon_start=0.0,  # no exploration in eval
    epsilon_end=0.0,
    epsilon_decay_steps=1,
    target_update_freq=1_000,
)

state_dict = torch.load(MODEL_PATH, map_location="cpu")
dqn_agent.q_net.load_state_dict(state_dict)
dqn_agent.target_net.load_state_dict(dqn_agent.q_net.state_dict())
dqn_agent.epsilon = 0.0  # always greedy in evaluation

# -----------------------------
# 3. Evaluate DQN policy
# -----------------------------

print("=== Evaluating DQN policy ===")

env_dqn = MarketMakingEnv(episode_length=EPISODE_LENGTH)

dqn_returns = []

# For the first DQN episode, record full paths for plotting
dqn_pnl_path = []
dqn_inv_path = []
dqn_mid_path = []
dqn_spread_path = []
dqn_skew_path = []
dqn_vol_proxy_path = []

for ep in range(NUM_EPISODES):
    obs, _ = env_dqn.reset()
    done = False
    ep_return = 0.0
    mids = []

    while not done:
        # DQN chooses an action index (0, 1, 2, ...)
        action = dqn_agent.select_action(obs, train=False)

        # Look up the half-spread and skew for that action
        half_spread, skew = env_dqn.action_map[action]

        # Step the environment
        next_obs, reward, done, info = env_dqn.step(action)
        ep_return += reward

        # For the first episode, record paths
        if ep == 0:
            dqn_pnl_path.append(info["pnl"])
            dqn_inv_path.append(env_dqn.sim.inventory)
            dqn_mid_path.append(env_dqn.sim.mid)
            dqn_spread_path.append(2.0 * half_spread)  # full spread
            dqn_skew_path.append(skew)

            mids.append(env_dqn.sim.mid)
            if len(mids) > 1:
                dqn_vol_proxy_path.append(abs(mids[-1] - mids[-2]))
            else:
                dqn_vol_proxy_path.append(0.0)

        obs = next_obs

    dqn_returns.append(ep_return)
    print(f"[DQN] Episode {ep+1}/{NUM_EPISODES} | Return ≈ {ep_return:.4f}")

# DQN summary stats
dqn_mean_ret = float(np.mean(dqn_returns))
dqn_std_ret = float(np.std(dqn_returns))
print("\n=== DQN Summary ===")
print(f"Mean episode return: {dqn_mean_ret:.4f}")
print(f"Std  episode return: {dqn_std_ret:.4f}")

env_dqn.close()

# -----------------------------
# 4. Evaluate A–S heuristic baseline
# -----------------------------

print("\n=== Evaluating A–S heuristic baseline ===")

as_policy = ASHeuristicPolicy()
env_as = MarketMakingEnv(episode_length=EPISODE_LENGTH)

as_returns = []

# For the first heuristic episode, record paths
as_pnl_path = []
as_inv_path = []
as_mid_path = []
as_spread_path = []
as_skew_path = []
as_vol_proxy_path = []

for ep in range(NUM_EPISODES):
    obs, _ = env_as.reset()
    done = False
    ep_return = 0.0
    mids = []

    while not done:
        # Heuristic policy has select_action(obs, env) -> action index
        action = as_policy.select_action(obs, env_as)

        half_spread, skew = env_as.action_map[action]

        next_obs, reward, done, info = env_as.step(action)
        ep_return += reward

        if ep == 0:
            as_pnl_path.append(info["pnl"])
            as_inv_path.append(env_as.sim.inventory)
            as_mid_path.append(env_as.sim.mid)
            as_spread_path.append(2.0 * half_spread)
            as_skew_path.append(skew)

            mids.append(env_as.sim.mid)
            if len(mids) > 1:
                as_vol_proxy_path.append(abs(mids[-1] - mids[-2]))
            else:
                as_vol_proxy_path.append(0.0)

        obs = next_obs

    as_returns.append(ep_return)
    print(f"[AS] Episode {ep+1}/{NUM_EPISODES} | Return ≈ {ep_return:.4f}")

# Heuristic summary stats
as_mean_ret = float(np.mean(as_returns))
as_std_ret = float(np.std(as_returns))
print("\n=== A–S Heuristic Summary ===")
print(f"Mean episode return: {as_mean_ret:.4f}")
print(f"Std  episode return: {as_std_ret:.4f}")

env_as.close()

# -----------------------------
# 5. Plot DQN episode 1 paths
# -----------------------------

steps_dqn = np.arange(len(dqn_pnl_path))

plt.figure()
plt.plot(steps_dqn, dqn_pnl_path)
plt.xlabel("Step")
plt.ylabel("PnL")
plt.title("DQN - PnL over episode 1")

plt.figure()
plt.plot(steps_dqn, dqn_inv_path)
plt.xlabel("Step")
plt.ylabel("Inventory")
plt.title("DQN - Inventory over episode 1")

plt.figure()
plt.plot(steps_dqn, dqn_mid_path)
plt.xlabel("Step")
plt.ylabel("Mid price")
plt.title("DQN - Mid price over episode 1")

plt.figure()
plt.scatter(dqn_vol_proxy_path, dqn_spread_path, alpha=0.5)
plt.xlabel("Abs mid-price change (vol proxy)")
plt.ylabel("Full spread")
plt.title("DQN - Vol proxy vs chosen spread (episode 1)")

plt.figure()
plt.scatter(dqn_inv_path, dqn_skew_path, alpha=0.5)
plt.xlabel("Inventory")
plt.ylabel("Skew")
plt.title("DQN - Inventory vs skew (episode 1)")

# -----------------------------
# 6. Plot A–S heuristic episode 1 paths
# -----------------------------

steps_as = np.arange(len(as_pnl_path))

plt.figure()
plt.plot(steps_as, as_pnl_path)
plt.xlabel("Step")
plt.ylabel("PnL")
plt.title("A–S Heuristic - PnL over episode 1")

plt.figure()
plt.plot(steps_as, as_inv_path)
plt.xlabel("Step")
plt.ylabel("Inventory")
plt.title("A–S Heuristic - Inventory over episode 1")

plt.figure()
plt.plot(steps_as, as_mid_path)
plt.xlabel("Step")
plt.ylabel("Mid price")
plt.title("A–S Heuristic - Mid price over episode 1")

plt.figure()
plt.scatter(as_vol_proxy_path, as_spread_path, alpha=0.5)
plt.xlabel("Abs mid-price change (vol proxy)")
plt.ylabel("Full spread")
plt.title("A–S Heuristic - Vol proxy vs chosen spread (episode 1)")

plt.figure()
plt.scatter(as_inv_path, as_skew_path, alpha=0.5)
plt.xlabel("Inventory")
plt.ylabel("Skew")
plt.title("A–S Heuristic - Inventory vs skew (episode 1)")

plt.show()
