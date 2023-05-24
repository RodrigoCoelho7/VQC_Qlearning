
import sys
import os
import datetime
import multiprocessing as mp
import numpy as np
import optuna
from collections import deque
import pickle
import logging

if __name__ == "__main__":
    seed = 42
    np.random.seed(seed)

    path_to_file = sys.argv[1]

    sys.path.append(path_to_file.rsplit('/', 1)[0])

    import_name = path_to_file.rsplit('/', 1)[1][:-3]

    def train_agent(max_memory_length, batch_size, steps_per_update, steps_per_target_update, learning_rate_in, learning_rate_var, num_layers):
            from model.q_learning_agent import QLearningAgent
            from DQN.dqn import DQN
            from vqc.vqc_circuits import UQC
            import tensorflow as tf
            import numpy as np
            tf.random.set_seed(seed)
            np.random.seed(seed)

            script = __import__(import_name)
            num_qubits = 2
            vqc = UQC(num_qubits, num_layers)
            replay_memory = deque(maxlen=max_memory_length)

            #Create the models
            model = QLearningAgent(vqc, script.observables, False, script.state_dim, script.rescaling_type, script.activation, script.pqc)
            model_target = QLearningAgent(vqc, script.observables, True, script.state_dim, script.rescaling_type, script.activation, script.pqc)
            model_target.set_weights(model.get_weights())

            steps_per_target_update = steps_per_target_update * steps_per_update

            optimizer_in =  tf.keras.optimizers.Adam(learning_rate=learning_rate_in, amsgrad=True)
            optimizer_var = tf.keras.optimizers.Adam(learning_rate=learning_rate_var, amsgrad=True)
            optimizer_bias = tf.keras.optimizers.Adam(learning_rate=learning_rate_var, amsgrad=True)
            optimizer_out = tf.keras.optimizers.Adam(learning_rate=script.learning_rate_out, amsgrad=True)

            # Create the agent
            agent = DQN(model, model_target, script.gamma, script.num_episodes, max_memory_length,
                        replay_memory, script.policy, batch_size,
                        steps_per_update, steps_per_target_update, optimizer_in, optimizer_out, optimizer_var,
                        optimizer_bias, script.w_in, script.w_var, script.w_out,script.w_bias, script.input_encoding, script.early_stopping,
                        script.operator)

            agent.train(script.environment, script.num_actions, script.acceptance_reward, script.necessary_episodes)

            return agent.episode_reward_history
    
    def sample_model_params(trial:optuna.Trial):
        max_memory_length = trial.suggest_int("max_memory_length", 10000, 100000, step = 10000)
        batch_size = trial.suggest_categorical("batch_size", [32,48])
        steps_per_update = trial.suggest_categorical("steps_per_update", [1,2,3,5])
        steps_per_target_update = trial.suggest_categorical("steps_per_target_update", [1,2,3,5,10])
        learning_rate_in = trial.suggest_categorical("learning_rate_in", [0.0001,0.001])
        learning_rate_var = trial.suggest_categorical("learning_rate_var", [0.0001,0.001])
        num_layers = trial.suggest_int("num_layers", 4, 7, step = 1)
        return  max_memory_length, batch_size, steps_per_update, steps_per_target_update, learning_rate_in, learning_rate_var, num_layers
    
    def objective_function(results):
        results_mean = np.mean(results, axis=0)
        area = np.abs(np.trapz(results_mean))
        maximum_performance_area = float(len(results[0]) * 200)

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
    sampler = optuna.samplers.TPESampler(seed=seed)
    study_name = "2qubituqc_cartpole"
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(direction="maximize", sampler = sampler,study_name=study_name, storage=storage_name, load_if_exists=True)
    study.optimize(objective, n_trials=500)

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