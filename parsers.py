from typing import Any

import gym.spaces
import numpy as np
from gym.spaces import MultiDiscrete
from rlgym.utils.action_parsers import ActionParser
from rlgym.utils.gamestates import GameState

# Structure:
# (throttle, steer, pitch, yaw, roll, jump, boost, handbrake)

class LookupAction(ActionParser):
    def __init__(self, bins=None):
        super().__init__()
        if bins is None:
            self.bins = [(-1, 0, 1)] * 5
        elif isinstance(bins[0], (float, int)):
            self.bins = [bins[0]] * 5
        else:
            self.bins = bins
        assert len(self.bins) == 5, "Need bins for throttle, steer, pitch, yaw and roll"
        self._lookup_table = self.make_lookup_table(self.bins)

    @staticmethod
    def make_lookup_table(bins):
        actions = []
        # Ground
        for throttle in bins[0]:
            for steer in bins[1]:
                for boost in (0, 1):
                    for handbrake in (0, 1):
                        if boost == 1 and throttle != 1:
                            continue
                        actions.append([throttle or boost, steer, 0, steer, 0, 0, boost, handbrake])
        # Aerial
        for pitch in bins[2]:
            for yaw in bins[3]:
                for roll in bins[4]:
                    for jump in (0, 1):
                        for boost in (0, 1):
                            for handbrake in (0, 1):
                                if pitch == roll == jump == 0:  # Duplicate with ground
                                    continue
                                actions.append([boost, yaw, pitch, yaw, roll, jump, boost, handbrake])
        actions = np.array(actions)
        return actions

    def get_action_space(self) -> gym.spaces.Space:
        return MultiDiscrete([len(self._lookup_table)] * 1)

    def parse_actions(self, actions: Any, state: GameState) -> np.ndarray:
        actions = actions.reshape(-1)
        return self._lookup_table[actions]