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
        #Create the models
        model = QLearningAgent(script.vqc, script.observables, False, script.state_dim, script.rescaling_type, script.activation)
        model_target = QLearningAgent(script.vqc, script.observables, True, script.state_dim, script.rescaling_type, script.activation)
        model_target.set_weights(model.get_weights())

        # Create the agent
        agent = DQN(model, model_target, script.gamma, script.num_episodes, script.max_memory_length,
                    script.replay_memory, script.policy, script.batch_size,
                    script.steps_per_update, script.steps_per_target_update, script.optimizer_in, script.optimizer_out, script.optimizer_var,
                    script.optimizer_bias, script.w_in, script.w_var, script.w_out,script.w_bias, script.input_encoding, script.early_stopping,
                    script.operator, script.hessian)

        agent.train(script.environment, script.num_actions, script.acceptance_reward, script.necessary_episodes)

        path_to_save = path_to_file.replace("configs", "../results")[:-3] + "/"

        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)

        current_time = datetime.datetime.now().strftime("%H:%M:%S")

        filename = f"agent{current_time}.pkl"
        agent.store_pickle(path_to_save, filename)

    num_agents = 5

    with mp.Pool(num_agents) as p:
        p.map(train_agent, range(num_agents))
    
