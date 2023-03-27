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

    def get_moving_average(self, window_size):
        rewards = self.get_rewards()
        moving_averages = []
        for reward in rewards:
            moving_averages.append(pd.Series(reward).rolling(window_size).mean())
        return moving_averages
    
    def get_gradients_all_params(self):
        mean = []
        variance = []
        aux = []
        gradients = self.get_gradients()
        gradients_all_parameters = []

        for i in range(self.number_of_agents):
            for j in range(len(gradients[i])):
                aux.append(gradients[i][j][0][0])
            gradients_all_parameters.append(aux)
            aux = []

        gradients_counts = [len(gradients_all_parameters[i]) for i in range(self.number_of_agents)]
        gradients_min = min(gradients_counts)

        aux_mean = []

        for i in range(gradients_min):
            for j in range(self.number_of_agents):
                aux_mean.append(gradients_all_parameters[j][i])
            mean.append(np.mean(np.linalg.norm(aux_mean)))
            variance.append(np.var(aux_mean))
            aux_mean = []
        return mean, variance
    