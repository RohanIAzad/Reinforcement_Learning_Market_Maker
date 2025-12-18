# baseline_as.py

import numpy as np


class ASHeuristicPolicy:
    """
    Simple Avellaneda–Stoikov-style heuristic baseline.

    - half_spread increases slightly with |inventory|
    - skew is proportional to -inventory (lean against position)
    - maps continuous (half_spread, skew) to nearest discrete action in env.action_map
    """

    def __init__(
        self,
        base_half_spread: float = 0.02,
        spread_inv_coeff: float = 0.01,
        skew_coeff: float = 0.02,
    ):
        """
        base_half_spread: base half-spread when inventory is zero
        spread_inv_coeff: how much to widen the half-spread per unit of |q_scaled|
        skew_coeff: controls how aggressively we skew vs inventory
        """
        self.base_half_spread = base_half_spread
        self.spread_inv_coeff = spread_inv_coeff
        self.skew_coeff = skew_coeff

    def select_action(self, obs, env):
        """
        obs = [time_remaining, inventory_scaled, mid_price]
        env.action_map: list of (half_spread, skew) combos
        Returns: discrete action index (int)
        """
        time_remaining, q_scaled, mid = obs

        # 1) Choose half-spread: widen slightly with inventory magnitude
        half_spread = self.base_half_spread + self.spread_inv_coeff * abs(q_scaled)

        # 2) Choose skew: lean against inventory
        #    If q_scaled > 0 (long)  → skew negative → push quotes downward → sell bias
        #    If q_scaled < 0 (short) → skew positive → buy bias
        skew = -self.skew_coeff * q_scaled

        # 3) Map (half_spread, skew) to nearest discrete action in env.action_map
        best_action = None
        best_dist = float("inf")

        for a, (hs_a, skew_a) in enumerate(env.action_map):
            dist = (half_spread - hs_a) ** 2 + (skew - skew_a) ** 2
            if dist < best_dist:
                best_dist = dist
                best_action = a

        return best_action
