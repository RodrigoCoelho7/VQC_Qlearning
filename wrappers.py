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
        return observation

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

