import sys
import os
import multiprocessing as mp
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import datetime
import numpy as np

# sys.argv[1] = path to config file wanted

if __name__ == "__main__":
    path_to_file = sys.argv[1]

    sys.path.append(path_to_file.rsplit('/', 1)[0])

    import_name = path_to_file.rsplit('/', 1)[1][:-3]
    
    def train_agent(agent_number):
        from model.q_learning_agent import QuantumQLearningAgent, NNQLearningAgent
        from DQN.dqn import QuantumDQN, ClassicalDQN
        script = __import__(import_name)
        
        #Create the models
        if script.model_quantum == True:
            model = QuantumQLearningAgent(script.vqc,script.quantum_model, script.observables, False, script.state_dim, script.rescaling_type, script.activation)
            model_target = QuantumQLearningAgent(script.vqc,script.quantum_model, script.observables, True, script.state_dim, script.rescaling_type, script.activation)
            model_target.set_weights(model.get_weights())

            # Create the agent
            agent = QuantumDQN(model, model_target, script.gamma, script.num_episodes, script.max_memory_length,
                        script.replay_memory, script.policy, script.batch_size,
                        script.steps_per_update, script.steps_per_target_update, script.optimizer_in, script.optimizer_out, script.optimizer_var,
                        script.optimizer_bias, script.w_in, script.w_var, script.w_out,script.w_bias, script.input_encoding, script.early_stopping,
                        script.operator, script.parameters_relative_change, script.entanglement_study)
        
        else:
            model = NNQLearningAgent(script.state_dim, script.num_actions, script.activation)
            model_target = NNQLearningAgent(script.state_dim, script.num_actions, script.activation)
            model_target.set_weights(model.get_weights())

            agent = ClassicalDQN(model, model_target, script.gamma, script.num_episodes, script.max_memory_length,
                        script.replay_memory, script.policy, script.batch_size,
                        script.steps_per_update, script.steps_per_target_update, script.optimizer, script.input_encoding, script.early_stopping,
                        script.operator, script.parameters_relative_change, script.entanglement_study)


        agent.train(script.environment, script.num_actions, script.acceptance_reward, script.necessary_episodes)

        path_to_save = path_to_file.replace("configs", "../results")[:-3] + "/"

        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)

        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        random_number = str(np.random.randint(1000))

        filename = f"agent{current_time}_{random_number}.pkl"
        agent.store_pickle(path_to_save, filename)

    num_agents = 10

    with mp.Pool(num_agents) as p:
        p.map(train_agent, range(num_agents))
    
