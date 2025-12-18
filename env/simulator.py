import numpy as np
from dataclasses import dataclass
from typing import Optional

# defining starting and default variables
@dataclass
class MarketSimConfig:
    init_mid: float = 100.0
    sigma: float = 0.6
    dt: float = 1.0
    kappa: float = 5.0     # base order arrival intensity
    alpha: float = 50.0    # sensitivity of arrival to distance
    max_inventory: int = 2
    inventory_penalty: float = 5.0 
    terminal_inventory_penalty: float = 50.0
    
    theta: float = 0.5                     # mean reversion strength
    mu: float = 100.0                       # long-term mean mid price
    seed: Optional[int] = None
    
# the simulator needs to be able to -
## 1. take bid and ask price
## 2. take the difference between bid and mid, ask and mid
## 3. model order arrival intensity as an exponential decay as in high intensity when quoting near the mid, the opposite when quoting far away from the mid.
## 4. calculate fill probabilities for bid and ask based on the intesity taking into account the spread (mid - bid, ask - mid)
## 5. update mid price based on random walk.
## 6. calculate reward based on PnL and inventory
## at the end of the episode , if there is still inventiry left apply a big penalty
class SimpleMarketSimulator:
    
    def __init__(self, cfg: MarketSimConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.reset()
    
    def reset(self):
        self.t = 0 # if there are 500 trading days (episodes) in the simulation and in each day there are 1000 trades (steps) being done, 
                   # at the end of each episode, set it to 0 
        self.mid = self.cfg.init_mid  # similarly at the start of a new episode set mid to inital mid
        self.inventory = 0
        self.cash = 0.0
        self.done = False
        return self._get_state()
    
    def step(self, bid_price, ask_price, T): # 1. take bid and ask prices
        if self. done:
            raise RuntimeError("Simulator is done. Call reset().")
        
        if bid_price >= ask_price:
            raise ValueError("Bid price must be < ask price.")
        
        # 2. take the difference between bid and mid, ask and mid. computing distances from mid.
        delta_b = max(0.0, self.mid - bid_price)  # distance from mid
        delta_a = max(0.0, ask_price - self.mid)
        
        # 3. model order arrival intensity
        kappa = self.cfg.kappa
        alpha = self.cfg.alpha  
        dt = self.cfg.dt
        
        lambda_b = kappa * np.exp(-alpha * delta_b)
        lambda_a = kappa * np.exp(-alpha * delta_a)
        
        # 4. calculate fill probabilities (Poisson process)
        
        p_fill_b = 1.0 - np.exp(-lambda_b * dt)
        p_fill_a = 1.0 - np.exp(-lambda_a * dt)
        
        fill_bid = self.rng.random() < p_fill_b # if random returns a value less than the probability, the fill happens
        fill_ask = self.rng.random() < p_fill_a # if random returns a value less than the probability, the fill happens
        
        
        # if bid order is completed then end up with 1 more inventory and if ask order is completed then end up with -1 inventory.
        inv_before = self.inventory
        
        if fill_bid and self.inventory+1 <= self.cfg.max_inventory:
            self.inventory += 1
            self.cash -= bid_price
        
        if fill_ask and self.inventory - 1 >= -self.cfg.max_inventory:
            self.inventory -= 1
            self.cash += ask_price
        
        # 5. Update mid price based on random walk
        # dw = self.rng.normal(0.0, np.sqrt(dt))
        # old_mid = self.mid
        # self.mid = max(0.01, self.mid + self.cfg.sigma * dw)
        
        # Ornstein–Uhlenbeck mean-reverting price process
        eps = self.rng.normal()

        theta = self.cfg.theta       # mean reversion strength
        mu = self.cfg.mu             # long-term mean mid-price
        sigma = self.cfg.sigma       # volatility
        dt = 1.0                     # or your chosen step size

        old_mid = self.mid
        self.mid = old_mid + theta * (mu - old_mid) * dt + sigma * eps
        
        # 6. reward based on PnL and inventory
        old_pnl = self._pnl(old_mid, inv_before, self.cash)
        new_pnl = self._pnl(self.mid, self.inventory, self.cash)
        delta_pnl = new_pnl - old_pnl
        
        # running inventory penalty (quadratic)
        inv_penalty = self.cfg.inventory_penalty * (self.inventory ** 2) * dt

        reward = delta_pnl - inv_penalty

        # 7) Time update & terminal condition
        self.t += 1
        self.done = self.t >= T   
        
        if self.done:
            # terminal penalty on inventory
            reward -= self.cfg.terminal_inventory_penalty * (self.inventory ** 2)
            
        state = self._get_state(T)
        info = {
            "fill_bid": fill_bid,
            "fill_ask": fill_ask,
            "old_mid": old_mid,
            "new_mid": self.mid,
            "delta_pnl": delta_pnl,
            "inventory_penalty": inv_penalty,
            "pnl": new_pnl,
        }
        return state, reward, self.done, info
    
    def _pnl(self, mid, inventory, cash):
        return cash + inventory * mid    
    
    def _get_state(self, T: Optional[int] = None):
        # For now just return raw components; the Gym env will turn this into a vector
        return {
            "t": self.t,
            "mid": self.mid,
            "inventory": self.inventory,
            "cash": self.cash,
            "done": self.done,
            "T": T,
        }
    
    