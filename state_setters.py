from dataclasses import dataclass
from subprocess import NORMAL_PRIORITY_CLASS
from tkinter import CallWrapper
from rlgym.utils.state_setters import StateSetter
from rlgym.utils.state_setters.wrappers import CarWrapper
from rlgym.utils.state_setters import StateWrapper
from rlgym.utils.common_values import BLUE_TEAM, ORANGE_TEAM, CEILING_Z
import numpy as np
import random as rnd
import math



@dataclass
class CarData:
    car_x: float
    car_y: float
    car_z: float
    car_yaw: float
    car_boost: float

class RandomMirrorSetterOneVOne(StateSetter):
    X_RANGE = [1024.0, 3500.0]
    Y_RANGE = [-3700.0, -1024.0]
    YAW_RANGE = [-math.pi, math.pi]

    @staticmethod
    def rnd_sign():
        return 1 if rnd.random() > 0.5 else -1

    def reset(self, state_wrapper: StateWrapper):
        
        state_wrapper.ball.position = np.array([0.0, 0.0, 92.75])
        
        car_boost = 100

        #car_location = [3457.8198, -1.1, 17.01]

        car_x = rnd.uniform(*self.X_RANGE) * self.rnd_sign()
        car_y = rnd.uniform(*self.Y_RANGE)
        car_z = 17.01 # At rest height for octane
        car_yaw = rnd.uniform(*self.YAW_RANGE)

        for car in state_wrapper.cars:

            if car.team_num == ORANGE_TEAM:
                car_x = -car_x
                car_y = -car_y
                car_yaw = -car_yaw

            car.set_pos(car_x, car_y, car_z)
            car.set_rot(yaw = car_yaw)
            car.boost = car_boost

class RandomMirrorSetterFlexible(StateSetter):
    X_RANGE = [1024.0, 3500.0]
    Y_RANGE = [-3700.0, -1024.0]
    YAW_RANGE = [-math.pi, math.pi]
    
    DISTANCE = 3500.0
    MAX_THETA = math.pi
    
    @staticmethod
    def rnd_sign():
        return 1 if rnd.random() > 0.5 else -1

    @staticmethod
    def gen_car_location():
        car_x = rnd.uniform(*RandomMirrorSetterFlexible.X_RANGE) * RandomMirrorSetterFlexible.rnd_sign()
        car_y = rnd.uniform(*RandomMirrorSetterFlexible.Y_RANGE)
        car_z = 17.01
        car_boost = 100
        car_yaw = rnd.uniform(*RandomMirrorSetterFlexible.YAW_RANGE)

        return CarData(car_x, car_y, car_z, car_yaw, car_boost)

    @staticmethod
    def gen_car_loc_fixed_distance(idx = 0, num_cars = 1):
        rho = RandomMirrorSetterFlexible.DISTANCE

        theta_partition = RandomMirrorSetterFlexible.MAX_THETA / num_cars
        theta = rnd.uniform(0, theta_partition) + (idx * theta_partition)
        
        car_x = rho * np.cos(theta)
        car_y = -(rho * np.sin(theta))
        car_z = 17.01
        car_boost = 100
        car_yaw = rnd.uniform(*RandomMirrorSetterFlexible.YAW_RANGE)

        return CarData(car_x, car_y, car_z, car_yaw, car_boost)
    
    @staticmethod
    def mirror_car_data(car_data: CarData):
        return CarData(-car_data.car_x, -car_data.car_y, car_data.car_z, -car_data.car_yaw, car_data.car_boost)

    @staticmethod
    def apply_car_data(car: CarWrapper, car_data: CarData):
        car.set_pos(car_data.car_x, car_data.car_y, car_data.car_z)
        car.set_rot(yaw = car_data.car_yaw)
        car.boost = car_data.car_boost

    def reset(self, state_wrapper: StateWrapper):
        blue_cars = state_wrapper.blue_cars()
        orange_cars = state_wrapper.orange_cars()

        num_blue= len(blue_cars)
        num_orange = len(orange_cars)

        state_wrapper.ball.position = np.array([0.0, 0.0, 92.75])

        assert num_blue == num_orange or num_blue == 1 , "Number of blue and orange cars must be equal, or there must be only one blue car"
        cars = state_wrapper.cars
        
        if len(cars) == 1:
            self.apply_car_data(cars[0], self.gen_car_loc_fixed_distance(0, 1))
        else:
            for i in range(num_blue):
                car_data = self.gen_car_loc_fixed_distance(i, num_blue)
                
                self.apply_car_data(blue_cars[i], car_data)
                self.apply_car_data(orange_cars[i], self.mirror_car_data(car_data))
