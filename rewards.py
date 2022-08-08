from rlgym.utils.reward_functions import RewardFunction
from rlgym.utils import math
from rlgym.utils.gamestates import GameState, PlayerData
from rlgym.utils.common_values import BALL_RADIUS, CAR_MAX_SPEED, SUPERSONIC_THRESHOLD
import numpy as np

class SpeedReward(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        linear_velocity = player.car_data.linear_velocity
        reward = float(math.vecmag(linear_velocity))
        
        return reward

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return 0


class QuickestTouchReward(RewardFunction):
    def __init__(self, timeout=0, tick_skip=8):
        self.timeout = timeout * 120 # convert to ticks
        self.tick_skip = tick_skip

    def reset(self, initial_state: GameState):
        self.timer = 0

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        self.timer += self.tick_skip
                
        return 0

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.ball_touched:
            
            return ((self.timeout - self.timer) / self.timeout) * 600 # Normalize to 0-100
        return -100


class SustainedVelocityPlayerToBallReward(RewardFunction):
    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        vel = player.car_data.linear_velocity
        pos_diff = state.ball.position - player.car_data.position
        
        norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
        scalar_vel_towards_ball = float(np.dot(norm_pos_diff, vel))

        norm_scalar_vel_towards_ball = scalar_vel_towards_ball / SUPERSONIC_THRESHOLD
        return norm_scalar_vel_towards_ball

class AccelerationPlayerToBallReward(RewardFunction):
    def __init__(self, tick_skip=8):
        super().__init__()
        self.tick_skip = tick_skip
        self.CAR_MAX_ACC = (CAR_MAX_SPEED) / self.tick_skip


    def reset(self, initial_state: GameState):
        self.previous_velocity = 0.0

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        vel = player.car_data.linear_velocity
        pos_diff = state.ball.position - player.car_data.position
        
        norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
        scalar_vel_towards_ball = float(np.dot(norm_pos_diff, vel))

        acc_towards_ball = (scalar_vel_towards_ball - self.previous_velocity) / self.tick_skip

        self.previous_velocity = scalar_vel_towards_ball

        if acc_towards_ball >= 0:
            return (acc_towards_ball / self.CAR_MAX_ACC) * 1
        else:
            norm_acc_towards_ball = (acc_towards_ball / self.CAR_MAX_ACC) * 100
            return  norm_acc_towards_ball

class FaceBallReward(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        pos_diff = state.ball.position - player.car_data.position
        norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
        return float(np.dot(player.car_data.forward(), norm_pos_diff))