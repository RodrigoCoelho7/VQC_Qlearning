import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import multiprocessing as mp
import pandas as pd
import numpy as np
import pickle

# sys.argv[1] = path to config file wanted

if __name__ == "__main__":
    path_to_file = sys.argv[1]

    sys.path.append(path_to_file.rsplit('/', 1)[0])

    import_name = path_to_file.rsplit('/', 1)[1][:-3]

    script = __import__(import_name)


    def train_and_evaluate(num_agents,learning_rate, learning_rate_out, max_memory_length, steps_per_update, steps_per_target_update):
        from model.q_learning_agent import QLearningAgent
        from DQN.dqn import DQN
        import tensorflow as tf
        from collections import deque

        model = QLearningAgent(script.num_qubits, script.num_layers,script.observables, script.circuit_arch, script.data_reuploading, False,script.measurement,script.state_dim, script.rescaling_type)
        model_target = QLearningAgent(script.num_qubits, script.num_layers,script.observables, script.circuit_arch, script.data_reuploading, True, script.measurement,script.state_dim, script.rescaling_type)
        model_target.set_weights(model.get_weights())

        optimizer_in = tf.keras.optimizers.Adam(learning_rate=learning_rate, amsgrad=True)
        optimizer_var = tf.keras.optimizers.Adam(learning_rate=learning_rate, amsgrad=True)
        optimizer_bias = tf.keras.optimizers.Adam(learning_rate=learning_rate, amsgrad=True)
        optimizer_out = tf.keras.optimizers.Adam(learning_rate=learning_rate_out, amsgrad=True)
        replay_memory = deque(maxlen=max_memory_length)
        steps_per_target_update = steps_per_target_update * steps_per_update

        agent = DQN(model, model_target, script.gamma, script.num_episodes, max_memory_length,
                    replay_memory, script.policy, script.batch_size,
                    steps_per_update, steps_per_target_update, optimizer_in, optimizer_out, optimizer_var,
                    optimizer_bias, script.w_in, script.w_var, script.w_out,script.w_bias, script.input_encoding, script.early_stopping,
                    script.operator)
        
        agent.train(script.environment, script.num_actions, script.acceptance_reward, script.necessary_episodes)

        return agent.episode_reward_history

    # Let's run 5 agents in parallel

    num_agents = 5

    path_to_save = path_to_file.replace("configs", "../results")[:-3] + "/"
    filename = "grid_hyperparameters.pkl"

    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)

    for learning_rate in script.learning_rate_in_and_var:
            for learning_rate_out in script.learning_rate_out:
                for max_memory_length in script.max_memory_length:
                    for steps_per_update in script.steps_per_update:
                        for steps_per_target_update in script.steps_per_target_update:

                            with mp.Pool(num_agents) as p:
                                results = p.starmap(train_and_evaluate, [(num_agents,learning_rate, learning_rate_out, max_memory_length, steps_per_update, steps_per_target_update) for _ in range(num_agents)])

                            # Results is a list of lists, each sublist is the episode reward history of one agent
                            # Let's see how long it took each agent to solve the environment
                            min_number_of_episodes = 750
                            best_mean_moving_average = 0

                            num_episodes_to_solve = []
                            for result in results:
                                num_episodes_to_solve.append(len(result))
                            
                            # Let's just keep the last 25 episodes of each agent
                            for i in range(len(results)):
                                results[i] = results[i][-25:]

                            #Now, lets calculate the mean of the rewards of each agent
                            mean_rewards = []
                            for result in results:
                                mean_rewards.append(np.mean(result))

                            #Finally, let's calculate the mean over every agent
                            mean_mean_rewards = np.mean(mean_rewards)


                            mean_num_episodes_to_solve = np.mean(num_episodes_to_solve)
                            if mean_num_episodes_to_solve < min_number_of_episodes or mean_mean_rewards > best_mean_moving_average:
                                min_number_of_episodes = mean_num_episodes_to_solve
                                best_learning_rate = learning_rate
                                best_learning_rate_out = learning_rate_out
                                best_max_memory_length = max_memory_length
                                best_steps_per_update = steps_per_update
                                best_steps_per_target_update = steps_per_target_update
                            
                            if np.sum(np.array(num_episodes_to_solve) < 750) >= 1 or mean_mean_rewards > 75:
                                values = {'episode_reward_history': results, 'learning_rate_in_and_var': learning_rate,
                                          'learning_rate_out': learning_rate_out, 'max_memory_length': max_memory_length,
                                          'steps_per_update': steps_per_update, 'steps_per_target_update': steps_per_target_update,
                                          'number_of_episodes_to_solve': num_episodes_to_solve, 'mean_number_of_episodes_to_solve': mean_num_episodes_to_solve}
                                
                                with open(path_to_save + filename, 'ab') as f:
                                    pickle.dump(values, f)
    
    path_to_save = path_to_file.replace("configs", "../results")[:-3] + "/"
    filename = "grid_best_hyperparameters.pkl"

    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)

    values = {'learning_rate_in_and_var': best_learning_rate,
                'learning_rate_out': best_learning_rate_out, 'max_memory_length': best_max_memory_length,
                'steps_per_update': best_steps_per_update, 'steps_per_target_update': best_steps_per_target_update,
                'min_number_of_episodes': min_number_of_episodes}
    
    with open(path_to_save + filename, 'wb') as f:
        pickle.dump(values, f)
    
    