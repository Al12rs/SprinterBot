from rlgym.utils.reward_functions import RewardFunction
from rlgym.utils import math
from rlgym.utils.gamestates import GameState, PlayerData
from rlgym.utils.common_values import BALL_RADIUS, CAR_MAX_SPEED, SUPERSONIC_THRESHOLD
import numpy as np

def scalar_speed_towards_ball(player :PlayerData, state: GameState) -> float:
    vel = player.car_data.linear_velocity
    pos_diff = state.ball.position - player.car_data.position
    
    norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
    scalar_vel_towards_ball = float(np.dot(norm_pos_diff, vel))
    return scalar_vel_towards_ball

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
    def __init__(self, timeout=0, tick_skip=8, num_agents=1):
        self.timeout = timeout * 120 # convert to ticks
        self.tick_skip = tick_skip
        self.num_agents = num_agents

    def reset(self, initial_state: GameState):
        self.timer = 0
    
    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        self.timer += self.tick_skip
        return 0

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        self.timer = self.timer / self.num_agents
        if player.ball_touched:
            #print("Touched ball after steps: ", self.timer)
            reward = ((self.timeout - self.timer) / self.timeout) * 100 + 30 
        else:
            norm_elapsed_time = self.timer / self.timeout
            player_ball_distance = float(np.linalg.norm(state.ball.position - player.car_data.position))
            norm_distance = player_ball_distance / 3500.0 # starting distance is 3500
            reward = -((norm_elapsed_time * 100) + (norm_distance * 100)) - 30

        # print("QuickestTouchReward.FinalReward(): ", reward)
        return reward

class SustainedVelocityPlayerToBallReward(RewardFunction):
    def __init__(self):
        super().__init__()
        self.comulative_reward = 0

    def reset(self, initial_state: GameState):
        self.comulative_reward = 0

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        vel = player.car_data.linear_velocity
        pos_diff = state.ball.position - player.car_data.position
        
        norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
        scalar_vel_towards_ball = float(np.dot(norm_pos_diff, vel))

        norm_scalar_vel_towards_ball = scalar_vel_towards_ball / SUPERSONIC_THRESHOLD

        if norm_scalar_vel_towards_ball > 0 and norm_scalar_vel_towards_ball < 1:
            reward = (scalar_vel_towards_ball / CAR_MAX_SPEED) / 50
        else:
            if norm_scalar_vel_towards_ball < 0:
                reward = (scalar_vel_towards_ball / CAR_MAX_SPEED) * 4
            else:
                reward = (scalar_vel_towards_ball / CAR_MAX_SPEED)

        self.comulative_reward += reward
        return reward
    
    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        # print("VelocityPlayerToBall: Cumulative: ", self.comulative_reward)
        return 0
class AccelerationPlayerToBallReward(RewardFunction):
    def __init__(self, tick_skip=8):
        super().__init__()
        self.tick_skip = tick_skip
        self.CAR_MAX_ACC = (CAR_MAX_SPEED ) / self.tick_skip
        self.cumulative_reward = 0

    def reset(self, initial_state: GameState):
        self.previous_velocity = 0.0
        self.cumulative_reward = 0

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        vel = player.car_data.linear_velocity
        pos_diff = state.ball.position - player.car_data.position
        
        norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
        scalar_vel_towards_ball = float(np.dot(norm_pos_diff, vel))

        acc_towards_ball = (scalar_vel_towards_ball - self.previous_velocity) / self.tick_skip

        self.previous_velocity = scalar_vel_towards_ball

        if acc_towards_ball >= 0:
            reward = (acc_towards_ball / self.CAR_MAX_ACC) * 0
        else:
            norm_acc_towards_ball = (acc_towards_ball / self.CAR_MAX_ACC) * 1
            reward = norm_acc_towards_ball
        
        self.cumulative_reward += reward
        return reward
    
    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        # print("AccelerationPlayerToBall: Cumulative: ", self.cumulative_reward)
        return 0

class SpeedOnBallTouchReward(RewardFunction):
    def __init__(self):
        super().__init__()
        self.cumulative_reward = 0
    
    def reset(self, initial_state: GameState):
        self.cumulative_reward = 0

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.ball_touched==True:
            speed_to_ball = scalar_speed_towards_ball(player, state)
            reward = ((speed_to_ball / SUPERSONIC_THRESHOLD) - 1) * 20
        else: 
            reward = 0

        self.cumulative_reward += reward
        return reward

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        # print("SpeedOnBallTouchReward: Cumulative: ", self.cumulative_reward)
        return 0

class FaceBallReward(RewardFunction):
    def __init__(self):
        super().__init__()
        self.cumulative_reward = 0

    def reset(self, initial_state: GameState):
        self.cumulative_reward = 0

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        pos_diff = state.ball.position - player.car_data.position
        norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
        reward = float(np.dot(player.car_data.forward(), norm_pos_diff)) / 100

        self.cumulative_reward += reward
        return reward
    
    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        # print("FaceBallReward: Cumulative: ", self.cumulative_reward)
        return 0

class OnGroundReward(RewardFunction):
    def __init__(self):
        super().__init__()
        self.cumulative_reward = 0

    def reset(self, initial_state: GameState):
        self.cumulative_reward = 0

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.on_ground:
            reward = 0.7
        else:
            reward = -1
        reward = reward / 100
        self.cumulative_reward += reward
        return reward

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        # print("OnGroundReward: Cumulative: ", self.cumulative_reward)
        return 0