import numpy as np
import gym
from gym import spaces
from typing import Optional

from .simulator import SimpleMarketSimulator, MarketSimConfig

class MarketMakingEnv(gym.Env):
    
    metadata = {"render.modes": ["human"]}  
    
    def __init__(self,
                 episode_length: int = 1000,
                 half_spread_grid=None,  # can input spread and skew as required. If nothing is inputted then there is a default array below
                 skew_grid=None,
                 sim_config: Optional[MarketSimConfig] = None):
        super().__init__()
        
        self.episode_length = episode_length
        
        if half_spread_grid is None:
            half_spread_grid = np.array([0.01, 0.02, 0.03], dtype=float)
        if skew_grid is None:
            skew_grid = np.array([-0.01, 0.0, 0.01], dtype=float)
            
        self.half_spread_grid = half_spread_grid
        self.skew_grid = skew_grid
        
        self.action_map = self._build_action_map() # creates pairs of spread, skew for each time step to set bid, ask price
        self.action_space = spaces.Discrete(len(self.action_map))
        
        obs_low = np.array([0.0, -1.0, 0.0], dtype=np.float32)
        obs_high = np.array([1.0,  1.0, np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32) 
        
        if sim_config is None:
            sim_config = MarketSimConfig()

        self.sim = SimpleMarketSimulator(sim_config) # simulator.py
        self._last_pnl = 0.0  # for rendering
        
    def _build_action_map(self):
        pairs = []
        for hs in self.half_spread_grid:
            for skew in self.skew_grid:
                pairs.append((hs, skew))
        return pairs
    
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            # re-seed underlying simulator
            self.sim.rng = np.random.default_rng(seed)

        state = self.sim.reset()
        self._last_pnl = self._compute_pnl(state)
        obs = self._state_to_obs(state)
        return obs, {}
    
    def step(self, action: int):
        half_spread, skew = self.action_map[action]

        # Build quotes from mid, half_spread, skew
        mid = self.sim.mid

        # Convention: skew > 0 biases to buy (more aggressive bid), skew < 0 biases to sell
        bid = mid - half_spread + skew
        ask = mid + half_spread + skew

        # Ensure bid < ask
        if bid >= ask:
            # If broken, penalize heavily and no trade
            bid = mid - 0.001
            ask = mid + 0.001

        # One step in simulator
        raw_state, reward, done, info = self.sim.step(bid, ask, self.episode_length) # calls the simulator.py

        obs = self._state_to_obs(raw_state)
        self._last_pnl = self._compute_pnl(raw_state)

        return obs, float(reward), bool(done), info
    
    def _state_to_obs(self, st: dict) -> np.ndarray:
        t = st["t"]
        T = st["T"] if st["T"] is not None else self.episode_length
        time_remaining = (T - t) / max(T, 1)

        inv = st["inventory"]
        inv_scaled = inv / float(self.sim.cfg.max_inventory)

        mid = st["mid"]

        obs = np.array([time_remaining, inv_scaled, mid], dtype=np.float32)  
        return obs
    
    def _compute_pnl(self, st: dict) -> float:
        return st["cash"] + st["inventory"] * st["mid"]
    
    def render(self, mode="human"):
        print(
            f"t={self.sim.t}, mid={self.sim.mid:.4f}, "
            f"inv={self.sim.inventory}, cash={self.sim.cash:.2f}, "
            f"PnL={self._last_pnl:.4f}"
        )

    def close(self):
        pass
    
        