from subprocess import NORMAL_PRIORITY_CLASS
from rlgym.utils.state_setters import StateSetter
from rlgym.utils.state_setters import StateWrapper
from rlgym.utils.common_values import BLUE_TEAM, ORANGE_TEAM, CEILING_Z
import numpy as np

class SpeedFlipStateSetter(StateSetter):
    def reset(self, state_wrapper: StateWrapper):
        
        # Imitate musty speedflip test training positions
        state_wrapper.ball.position = np.array([1.74, 5.19, 92.75])
        
        car_location = [3457.8198, -1.1, 17.01]
        
        car_pitch = 100
        car_yaw = 32748
        car_roll = 0

        car_boost = 33

        for car in state_wrapper.cars:
            car.set_pos(*car_location)
            car.set_rot(pitch=car_pitch, yaw=car_yaw, roll=car_roll)
            car.boost = car_boost
