## Reinforcement Learning Market Maker
This project implements a reinforcement learning–based market maker that dynamically quotes bid and ask prices in a simulated single-asset market. The agent is trained to balance profitability from spread capture with inventory risk management, inspired by the classical Avellaneda–Stoikov market-making framework.

The environment, simulator, and agent are built from scratch to highlight how reward design, inventory penalties, and price dynamics shape learned market-making behavior.

### Project Overview

Market makers provide liquidity by continuously quoting buy (bid) and sell (ask) prices. Their goal is not directional prediction, but to:

* Capture the bid–ask spread
* Manage inventory risk
* Avoid adverse selection during volatile regimes

In this project:

* A stochastic market simulator models order arrivals as Poisson processes
* Quote competitiveness controls fill probability via exponential intensity decay
* A Deep Q-Network (DQN) agent learns optimal bid/ask placement
* Rewards are based on mark-to-market PnL with inventory penalties

---

### Key Concepts

* Reinforcement Learning (DQN)
* Market Microstructure
* Inventory-Aware Quoting
* Poisson Order Arrivals
* Mean-Reverting Price Dynamics
* Reward Shaping & Risk Control

---

### Repository Structure
```
├── simulator.py      # Market simulator (order arrivals, fills, mid-price dynamics)
├── env_mm.py         # Gym-style market making environment
├── dqn.py            # Deep Q-Network agent implementation
├── train_dqn.py      # Training loop
├── eval_mm.py        # Evaluation & heuristic comparison
├── requirements.txt  # Python dependencies
```
