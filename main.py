import sys
from model.q_learning_agent import QLearningAgent
from DQN.dqn import DQN
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tensorflow as tf
import datetime

# sys.argv[1] = path to config file wanted

if __name__ == "__main__":
    path_to_file = sys.argv[1]

    sys.path.append(path_to_file.rsplit('/', 1)[0])

    import_name = path_to_file.rsplit('/', 1)[1][:-3]

    script = __import__(import_name)

    #Create the models
    model = QLearningAgent(script.num_qubits, script.num_layers,script.observables, script.circuit_arch, script.data_reuploading, False,script.measurement,script.state_dim, script.rescaling_type)
    model_target = QLearningAgent(script.num_qubits, script.num_layers,script.observables, script.circuit_arch, script.data_reuploading, True, script.measurement,script.state_dim, script.rescaling_type)
    model_target.set_weights(model.get_weights())

    # Create the agent
    agent = DQN(model, model_target, script.gamma, script.num_episodes, script.max_memory_length,
                script.replay_memory, script.policy, script.batch_size,
                script.steps_per_update, script.steps_per_target_update, script.optimizer_in, script.optimizer_out, script.optimizer_var,
                script.optimizer_bias, script.w_in, script.w_var, script.w_out,script.w_bias, script.input_encoding, script.early_stopping)
    
    agent.train(script.environment, script.num_actions, script.acceptance_reward, script.necessary_episodes)

    path_to_save = path_to_file.replace("configs", "../results")[:-3] + "/"

    current_time = datetime.datetime.now().strftime("%H:%M:%S")
    
    filename = f"agent{current_time}.pkl"
    agent.store_pickle(path_to_save, filename)