import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import os


class Analysis():
    """
    This Class is used to load the data of the trained models from the respective pickle files
    so that we can easily analyse the results. It should be used as follows:

    analysis = Analysis(path, filename, number_of_agents)

    where path is the path to the folder where the pickle files are stored, filename is the name of the pickle files
    excluding "_{i}.pkl" where i is the number of the agent, e.g. "agent_1.pkl", and number_of_agents is the number 
    of agents that were trained using that model.

    The class has the following methods:

    load_data(): loads the data from the pickle files into a list of dictionaries
    get_final_weights(): returns a list of the final weights of the model
    get_loss(): returns a list of lists with the loss at every training step
    get_qvalues(): returns the q-values of the model at every training step
    get_gradients(): returns the gradients of the model at every training step
    get_rewards(): returns the rewards of the model of every episode
    get_moving_average(window_size): returns the moving average of the rewards of the model acording to the window size given
    get_gradients_all_params(): returns the mean and variance of the gradients of all the parameters of the model
    """

    def __init__(self, path_to_dir):
        self.path = path_to_dir
        self.pickle_files = [f for f in os.listdir(self.path) if f.endswith(".pkl")]
        self.number_of_agents = len(self.pickle_files)
        self.data = []
        self.load_data()

    def load_data(self):
        for pickle_file in self.pickle_files:
            with open(os.path.join(self.path, pickle_file), 'rb') as f:
                self.data.append(pickle.load(f))
            
    def get_final_weights(self):
        return [self.data[i]["weights"] for i in range(self.number_of_agents)]
    
    def get_loss(self):
        return [self.data[i]["loss_array"] for i in range(self.number_of_agents)]
    
    def get_qvalues(self):
        return [self.data[i]["q_values"] for i in range(self.number_of_agents)]
    
    def get_gradients(self):
        return [self.data[i]["gradients"] for i in range(self.number_of_agents)]
    
    def get_rewards(self):
        return [self.data[i]["episode_reward_history"] for i in range(self.number_of_agents)]
    
    def get_parameters_relative_change(self):
        return [self.data[i]["parameters_relative_change_array"] for i in range(self.number_of_agents)]

    def get_moving_average(self, window_size):
        rewards = self.get_rewards()
        moving_averages = []
        for reward in rewards:
            moving_averages.append(pd.Series(reward).rolling(window_size).mean())
        return moving_averages
    
    def calculate_mean_variance_gradients(self, return_max = False, return_min = False):
        gradients = self.get_gradients()
        min_length = min([len(gradients[i]) for i in range(len(gradients))])

        gradients = [gradients[i][:min_length] for i in range(len(gradients))]

        def flatten_gradients(gradients):
            for i in range(len(gradients)):
                for j in range(len(gradients[i])):
                    gradients[i][j] = np.concatenate([lista.flatten() for lista in gradients[i][j]], axis = 0)

        flatten_gradients(gradients)

        gradients_array = np.array(gradients)

        magnitudes_gradients = np.linalg.norm(gradients_array, axis = 2)

        mean_magnitudes_gradients = np.mean(magnitudes_gradients, axis = 0)

        std_magnitudes_gradients = np.std(magnitudes_gradients, axis = 0)

        max_magnitudes_gradients = np.max(magnitudes_gradients, axis = 0)

        min_magnitudes_gradients = np.min(magnitudes_gradients, axis = 0)

        max_index = np.argmax(np.max(gradients_array, axis = 0), axis = 1)

        min_index = np.argmin(np.min(gradients_array, axis = 0), axis = 1)

        if return_max and return_min:
            return mean_magnitudes_gradients, std_magnitudes_gradients, max_magnitudes_gradients, max_index, min_magnitudes_gradients, min_index
        elif return_max:
            return mean_magnitudes_gradients, std_magnitudes_gradients, max_magnitudes_gradients, max_index
        elif return_min:
            return mean_magnitudes_gradients, std_magnitudes_gradients, min_magnitudes_gradients, min_index
        else:
            return mean_magnitudes_gradients, std_magnitudes_gradients