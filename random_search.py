import sys
import os
import multiprocessing as mp
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import datetime

# sys.argv[1] = path to config file wanted

if __name__ == "__main__":
    path_to_file = sys.argv[1]

    sys.path.append(path_to_file.rsplit('/', 1)[0])

    import_name = path_to_file.rsplit('/', 1)[1][:-3]

    script = __import__(import_name)

    
    def train_agent(agent_number):
        from model.q_learning_agent import QLearningAgent
        from DQN.dqn import DQN
        import tensorflow as tf
        import random
        from collections import deque
        import pickle

        #Create the models
        model = QLearningAgent(script.num_qubits, script.num_layers,script.observables, script.circuit_arch, script.data_reuploading, False,script.measurement,script.state_dim, script.rescaling_type)
        model_target = QLearningAgent(script.num_qubits, script.num_layers,script.observables, script.circuit_arch, script.data_reuploading, True, script.measurement,script.state_dim, script.rescaling_type)
        model_target.set_weights(model.get_weights())

        learning_rate_in_and_var = random.choice(script.learning_rate_in_and_var)
        optimizer_in =  tf.keras.optimizers.Adam(learning_rate=learning_rate_in_and_var, amsgrad=True)
        optimizer_var = tf.keras.optimizers.Adam(learning_rate=learning_rate_in_and_var, amsgrad=True)
        optimizer_bias = tf.keras.optimizers.Adam(learning_rate=learning_rate_in_and_var, amsgrad=True)
        optimizer_out = tf.keras.optimizers.Adam(learning_rate=script.learning_rate_out, amsgrad=True)

        max_memory_length = random.choice(script.max_memory_length)
        replay_memory = deque(maxlen=max_memory_length)
        steps_per_update = random.choice(script.steps_per_update)
        steps_per_target_update = random.choice(script.steps_per_target_update) * steps_per_update

        # Create the agent
        agent = DQN(model, model_target, script.gamma, script.num_episodes, script.max_memory_length,
                    replay_memory, script.policy, script.batch_size,
                    steps_per_update, steps_per_target_update, optimizer_in, optimizer_out, optimizer_var,
                    optimizer_bias, script.w_in, script.w_var, script.w_out,script.w_bias, script.input_encoding, script.early_stopping,
                    script.operator)
        
        rewards = agent.train(script.environment, script.num_actions, script.acceptance_reward, script.necessary_episodes)

        min_number_of_episodes = 3000
        best_params = None
        score = 3000
        all_params = []

        path_to_save = path_to_file.replace("configs", "../results")[:-3] + "/"
        filename = "hyperparameters.pkl"

        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)

        #We want to know how quickly it solved the environment, so we take the average of the last 25 episodes
        solved_environment = True if sum(rewards[-25:])/25 >=195 else False

        #I want to store all the hyperparameters that solved the environment and their rewards in a pickle file
        if score < sum(rewards[-25:])/25 and not solved_environment:
            score = sum(rewards[-25:])/25
            all_params.append((learning_rate_in_and_var, script.learning_rate_out, max_memory_length, steps_per_update, steps_per_target_update, rewards, solved_environment))
            
            with open(path_to_save + filename, "wb") as f:
                pickle.dump(all_params, f)
        elif solved_environment:
            all_params.append((learning_rate_in_and_var, script.learning_rate_out, max_memory_length, steps_per_update, steps_per_target_update, rewards, solved_environment))

            with open(path_to_save + filename, "wb") as f:
                pickle.dump(all_params, f)
            if min_number_of_episodes > len(rewards):
                min_number_of_episodes = len(rewards)
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        filename = f"agent{current_time}.pkl"
        agent.store_pickle(path_to_save, filename)

    num_agents = 6

    while(True):
        with mp.Pool(num_agents) as p:
            p.map(train_agent, range(num_agents))
    
