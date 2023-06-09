
import sys
import os
import datetime
import multiprocessing as mp
import numpy as np
import optuna
from collections import deque
import pickle
import logging
from vqc.vqc_circuits import UQC

if __name__ == "__main__":
    seed = 42
    np.random.seed(seed)

    path_to_file = sys.argv[1]

    sys.path.append(path_to_file.rsplit('/', 1)[0])

    import_name = path_to_file.rsplit('/', 1)[1][:-3]

    def train_agent(steps_per_target_update, num_layers):
            from model.q_learning_agent import QLearningAgent
            from DQN.dqn import DQN
            import tensorflow as tf
            tf.random.set_seed(seed)

            script = __import__(import_name)

            vqc = UQC(script.num_qubits, num_layers)


            #Create the models
            model = QLearningAgent(vqc, script.observables, False, script.state_dim, script.rescaling_type, script.activation)
            model_target = QLearningAgent(vqc, script.observables, True, script.state_dim, script.rescaling_type, script.activation)
            model_target.set_weights(model.get_weights())

            # Create the agent
            agent = DQN(model, model_target, script.gamma, script.num_episodes, script.max_memory_length,
                        script.replay_memory, script.policy, script.batch_size,
                        script.steps_per_update, steps_per_target_update, script.optimizer_in, script.optimizer_out, script.optimizer_var,
                        script.optimizer_bias, script.w_in, script.w_var, script.w_out,script.w_bias, script.input_encoding, script.early_stopping,
                        script.operator, script.parameters_relative_change)

            agent.train(script.environment, script.num_actions, script.acceptance_reward, script.necessary_episodes)

            return agent.episode_reward_history
    
    def sample_model_params(trial:optuna.Trial):
        steps_per_target_update = trial.suggest_categorical("steps_per_target_update", [100,250])
        num_layers = trial.suggest_categorical("num_layers", [7, 10, 15])
        return  steps_per_target_update, num_layers
    
    def objective_function(results):
        results_mean = np.mean(results, axis=0)
        area = np.abs(np.trapz(results_mean))
        maximum_performance_area = float(len(results[0]) * 500)

        # Now we want to create a metric called performance area and normalize it between 0 and 1
        performance_area = area / maximum_performance_area
        return performance_area

    def objective(trial):
        num_agents = 5

        with mp.Pool(num_agents) as p:
            results = p.starmap(train_agent, [(sample_model_params(trial)) for _ in range(num_agents)])

        # results is a list of lists containing the returns of each agent. We need to calculate the mean and the standard deviation
        performance = objective_function(results)
        return performance

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    search_space = {"steps_per_target_update": [100, 250], "num_layers": [7, 10, 15]}
    sampler = optuna.samplers.GridSampler(search_space=search_space,seed=seed)
    study_name = "acrobot_grid"
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(direction="minimize", sampler = sampler,study_name=study_name, storage=storage_name, load_if_exists=True)
    study.optimize(objective, n_trials=10)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    dictionary = {"Best Parameters": study.best_params, "Best Value": study.best_value, "Best Trial": study.best_trial, "All Trials": study.trials}

    path_to_save = path_to_file.replace("configs", "../results")[:-3] + "/"

    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)

    current_time = datetime.datetime.now().strftime("%H:%M:%S")

    filename = f"agent{current_time}.pkl"
    with open(path_to_save + filename, 'wb') as f:
        pickle.dump(dictionary, f)