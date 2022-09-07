import rlgym
import time

import numpy as np
from rlgym.envs import Match
from rlgym.utils.action_parsers import DiscreteAction
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize, VecCheckNan
from stable_baselines3.ppo import MlpPolicy

from rlgym.utils.obs_builders import AdvancedObs
from rlgym.utils.state_setters import DefaultState
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, GoalScoredCondition
from rlgym_tools.sb3_utils import SB3MultipleInstanceEnv
from rlgym_tools.extra_obs.advanced_padder import AdvancedObsPadder
from rlgym_tools.extra_obs.advanced_stacker import AdvancedStacker
from rlgym.utils.reward_functions.common_rewards.misc_rewards import EventReward
from rlgym.utils.reward_functions.common_rewards.player_ball_rewards import VelocityPlayerToBallReward
from rlgym.utils.reward_functions import CombinedReward

from rewards import QuickestTouchReward, SustainedVelocityPlayerToBallReward, AccelerationPlayerToBallReward, FaceBallReward, OnGroundReward, SpeedOnBallTouchReward
from state_setters import RandomMirrorSetterFlexible, BallTouchedCondition, BackwardsStateSetter
from parsers import LookupAction

half_life_seconds = 3   # Easier to conceptualize, after this many seconds the reward discount is 0.5

frame_skip = 4 

fps = 120 / frame_skip
gamma = np.exp(np.log(0.5) / (fps * half_life_seconds))  # Quick mafs
agents_per_match = 1
num_instances = 1
target_steps = 1_000_000
steps = target_steps // (num_instances * agents_per_match) #making sure the experience counts line up properly
batch_size = target_steps//10 #getting the batch size down to something more manageable - 100k in this case was /10 before
training_interval = 1_000_000
mmr_save_frequency = 1_000_000
ep_len_seconds = 8

max_steps = int(round(ep_len_seconds * fps))

reward_function=CombinedReward(
            (   
                SustainedVelocityPlayerToBallReward(),
                QuickestTouchReward(timeout=ep_len_seconds, tick_skip=frame_skip, num_agents=agents_per_match),
                AccelerationPlayerToBallReward(tick_skip=frame_skip),
                FaceBallReward(),
                OnGroundReward(),
            ),
            (1.5, 1.0, 5.0, 2.0, 1.0))

env = rlgym.make(game_speed=1, 
                    tick_skip=4,
                    spawn_opponents=False, 
                    team_size=1, 
                    terminal_conditions=[TimeoutCondition(max_steps),BallTouchedCondition()],
                    reward_fn=reward_function,
                    obs_builder=AdvancedStacker(),
                    action_parser=DiscreteAction(),
                    state_setter=BackwardsStateSetter()
                    )  

model_file_name = "exit_save"


try:
    model = PPO.load(
        model_file_name + '.zip',
        env,
        device="auto"
    )
    print("Loaded previous exit save.")
except:
    print("No saved model found, creating new model.")
    from torch.nn import Tanh
    policy_kwargs = dict(
        activation_fn=Tanh,
        net_arch=[512, 512, dict(pi=[256, 256, 256], vf=[256, 256, 256])],
    )
    model = PPO(
        MlpPolicy,
        env,
        n_epochs=10,                 # PPO calls for multiple epochs
        policy_kwargs=policy_kwargs,
        learning_rate=5e-5,          # Around this is fairly common for PPO
        ent_coef=0.01,               # From PPO Atari was 0.01, trying with 0.02
        vf_coef=1.,                  # From PPO Atari
        gamma=gamma,                 # Gamma as calculated using half-life
        verbose=3,                   # Print out all the info as we're going
        batch_size=batch_size,             # Batch size as high as possible within reason
        n_steps=steps,                # Number of steps to perform before optimizing network
        tensorboard_log="/logs",  # `tensorboard --logdir out/logs` in terminal to see graphs
        device="auto"                # Uses GPU if available
    )


while True:
    obs = env.reset()
    obs_1 = np.array(obs)
    # obs_2 = obs[1]
    done = False
    steps = 0
    ep_reward = 0
    t0 = time.time()
    while not done:
        # self.actor.predict(state, deterministic=True)
        actions_1, _ = model.predict(obs_1, deterministic=True)
        # actions_2, _ = model.predict(obs_2, deterministic=True)

        actions = actions_1
        new_obs, reward, done, state = env.step(actions)
        # ep_reward += reward
        obs_1 = np.array(new_obs)
        # obs_2 = new_obs[1]
        steps += 1

    length = time.time() - t0
    print("Step time: {:1.5f} | Episode time: {:.2f} | Episode Reward: {:.2f}".format(length / steps, length, 0))