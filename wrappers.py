import gym
import numpy as np

#CartPole observation space:
#Type: Box(4)
#Num	Observation                 Min         Max
#0	Cart Position             -4.8            4.8
#1	Cart Velocity             -Inf            Inf
#2	Pole Angle                 -24 deg        24 deg
#3	Pole Velocity At Tip      -Inf            Inf

"""
This script will contain the modules for the different encoding types:

1. Nothing-encoding (Do nothing and just pass the observation as it is)

2. Scaled Directional Encoding - If the feature is limited to a certain range, then we normalize it to the range [0, 2pi]
                                 and if the feature is not limited, we simply rotate it byt pi if it's bigger than 0

3. Scaled Continuous Encoding - If the feature is limited to a certain range, then we normalize it to the range [0, 2pi]
                                and if not, we apply the arctan of the observation and feed it

4. Continuous Encoding - We apply the arctan of the observation and feed it
"""

class NothingEncoding(gym.ObservationWrapper):
    def observation(self, observation):
        return np.array(observation)

class ScaledDirectionalEncoding(gym.ObservationWrapper):
    def observation(self, obs):
        #CartPole Position [-4.8,4.8] -> [0,2pi]
        obs[0] = ((obs[0] + 4.8) / 9.6) * 2 * np.pi

        # Scale pole angle (range [-0.418, 0.418]) to range [0, 2pi]
        obs[2] = ((obs[2] + 0.418) / 0.836) * 2 * np.pi

        #Continuous Encoding
        obs[1] = np.pi if obs[1] > 0 else 0
        obs[3] = np.pi if obs[3] > 0 else 0
        return obs

class ScaledContinuousEncoding_normal(gym.ObservationWrapper):
    def observation(self, obs):
        #CartPole Position [-4.8,4.8] -> [0,2pi]
        obs[0] = ((obs[0] + 4.8) / 9.6) * 2 * np.pi

        # Scale pole angle (range [-0.418, 0.418]) to range [0, 2pi]
        obs[2] = ((obs[2] + 0.418) / 0.836) * 2 * np.pi

        #Continuous Encoding
        obs[1] = np.arctan(obs[1])
        obs[3] = np.arctan(obs[3])
        return obs

class ScaledContinuousEncoding_mine(gym.ObservationWrapper):
    def observation(self, obs):
        #CartPole Position [-4.8,4.8] -> [0,2pi]
        obs[0] = ((obs[0] + 2.4) / 4.8) * 2 * np.pi

        # Scale pole angle (range [-0.418, 0.418]) to range [0, 2pi]
        obs[2] = ((obs[2] + 0.2095) / 0.418) * 2 * np.pi

        #Continuous Encoding
        obs[1] = np.arctan(obs[1])
        obs[3] = np.arctan(obs[3])
        return obs

class ContinuousEncoding(gym.ObservationWrapper):
    def observation(self, obs):
        return np.arctan(obs)
    
class AcrobotEncoding(gym.ObservationWrapper):
    def observation(self, obs):
        obs[4] = ((obs[4] + 4*np.pi) / (8*np.pi)) -1
        obs[5] = ((obs[5] + 9*np.pi) / (18*np.pi)) -1
        return obs
    
class ScaledEncoding(gym.ObservationWrapper):
    def observation(self, obs):
        # obs[0], obs[1], obs[2] and obs[3] are all between -1 and 1 and I want to scale them to [0, 2pi]
        obs[0] = ((obs[0] + 1) / 2) * 2 * np.pi
        obs[1] = ((obs[1] + 1) / 2) * 2 * np.pi
        obs[2] = ((obs[2] + 1) / 2) * 2 * np.pi
        obs[3] = ((obs[3] + 1) / 2) * 2 * np.pi
        #obs[4] is between -4pi and 4pi and I want to scale it to [0, 2pi]
        obs[4] = ((obs[4] + 4*np.pi) / (8*np.pi)) * 2 * np.pi
        #obs[5] is between -9pi and 9pi and I want to scale it to [0, 2pi]
        obs[5] = ((obs[5] + 9*np.pi) / (18*np.pi)) * 2 * np.pi
        return obs

class AcrobotEncodingV2(gym.ObservationWrapper):
    def observation(self, obs):
        theta_1 = np.arccos(obs[0])
        theta_2 = np.arccos(obs[2])
        theta1_angular = ((obs[4] + 4*np.pi) / (8*np.pi)) * 2 * np.pi
        theta2_angular = ((obs[5] + 9*np.pi) / (18*np.pi)) * 2 * np.pi
        new_observations = np.array([theta_1, theta_2, theta1_angular, theta2_angular])
        return new_observations

