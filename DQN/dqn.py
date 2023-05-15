import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import tensorflow as tf
import numpy as np
import gym
import pickle

class DQN():
    def __init__(self, model, model_target, gamma, num_episodes,max_memory_length,
                 replay_memory, policy,batch_size,
                 steps_per_update, steps_per_target_update,
                 optimizer_in, optimizer_out, optimizer_var, optimizer_bias,
                 w_in, w_var, w_out,w_bias, input_encoding, early_stopping,
                 operator):
        
        self.model = model
        self.model_target = model_target
        self.gamma = gamma
        self.num_episodes = num_episodes
        self.max_memory_length = max_memory_length
        self.replay_memory = replay_memory
        self.policy = policy
        self.batch_size = batch_size
        self.steps_per_update = steps_per_update
        self.steps_per_target_update = steps_per_target_update
        self.optimizer_in = optimizer_in
        self.optimizer_out = optimizer_out
        self.optimizer_var = optimizer_var
        self.optimizer_bias = optimizer_bias
        self.w_in = w_in
        self.w_var = w_var
        self.w_out = w_out
        self.w_bias = w_bias
        self.input_encoding = input_encoding
        self.early_stopping = early_stopping
        self.operator = operator
        self.episode_reward_history = []
        self.gradients = []
        self.loss_array = []
        self.q_values_array = []
    
    @tf.function
    def Q_learning_update(self,states, actions, rewards, next_states, done, n_actions):
        states = tf.convert_to_tensor(states)
        actions = tf.convert_to_tensor(actions)
        rewards = tf.convert_to_tensor(rewards)
        next_states = tf.convert_to_tensor(next_states)
        done = tf.convert_to_tensor(done)

        # Compute their target q_values and the masks on sampled actions
        future_rewards = self.model_target([next_states])
        target_q_values = rewards + (self.gamma * self.operator.apply(future_rewards)
                                                       * (1.0 - done))
        masks = tf.one_hot(actions, n_actions)

        with tf.GradientTape() as tape:
            tape.watch(self.model.trainable_variables)
            q_values = self.model([states])
            q_values_masked = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
            loss = tf.keras.losses.MSE(target_q_values, q_values_masked)
        grads = tape.gradient(loss, self.model.trainable_variables)


        if self.optimizer_in is not None:
            if self.optimizer_bias is not None:
                if self.optimizer_out is not None:
                    for optimizer, w in zip([self.optimizer_in, self.optimizer_var, self.optimizer_out, self.optimizer_bias], [self.w_in, self.w_var, self.w_out, self.w_bias]):
                        optimizer.apply_gradients([(grads[w], self.model.trainable_variables[w])])
                else:
                    for optimizer, w in zip([self.optimizer_in, self.optimizer_var, self.optimizer_bias], [self.w_in, self.w_var, self.w_bias]):
                        optimizer.apply_gradients([(grads[w], self.model.trainable_variables[w])])
            else:
                if self.optimizer_out is not None:
                    for optimizer, w in zip([self.optimizer_in, self.optimizer_var, self.optimizer_out], [self.w_in, self.w_var, self.w_out]):
                        optimizer.apply_gradients([(grads[w], self.model.trainable_variables[w])])
                else:
                    for optimizer, w in zip([self.optimizer_in, self.optimizer_var], [self.w_in, self.w_var]):
                        optimizer.apply_gradients([(grads[w], self.model.trainable_variables[w])])
        else:
            if self.optimizer_bias is not None:
                if self.optimizer_out is not None:
                    for optimizer, w in zip([self.optimizer_var, self.optimizer_out, self.optimizer_bias], [self.w_var, self.w_out, self.w_bias]):
                        optimizer.apply_gradients([(grads[w], self.model.trainable_variables[w])])
                else:
                    for optimizer, w in zip([self.optimizer_var, self.optimizer_bias], [self.w_var, self.w_bias]):
                        optimizer.apply_gradients([(grads[w], self.model.trainable_variables[w])])
            else:
                if self.optimizer_out is not None:
                    for optimizer, w in zip([self.optimizer_var, self.optimizer_out], [self.w_var, self.w_out]):
                        optimizer.apply_gradients([(grads[w], self.model.trainable_variables[w])])
                else:
                    for optimizer, w in zip([self.optimizer_var], [self.w_var]):
                        optimizer.apply_gradients([(grads[w], self.model.trainable_variables[w])]) 
        return grads, loss, q_values
    
    def validate(self,agent, environment):
        env = gym.make(environment)
        env = self.input_encoding(env)

        total_reward = 0
        state = env.reset()
        done = False
        while not done:
            state_array = np.array(state) 
            state = tf.convert_to_tensor([state_array])
            q_vals = agent([state])
            action = int(tf.argmax(q_vals[0]).numpy())
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            state = next_state

        return total_reward

    def train(self, environment, n_actions, acceptance_reward, necessary_episodes):
        is_training = True
        env = gym.make(environment)
        env = self.input_encoding(env)
        step_count = 0
        for episode in range(self.num_episodes):
            episode_reward = 0
            state = env.reset()

            while True:
                # Choose action to interact with the environment
                action = self.policy.select_action(state, self.model, n_actions)

                # Apply sampled action in the environment, receive reward and next state
                next_state, reward, done, _ = env.step(action)

                interaction = {'state': np.array(state), 'action': action, 'next_state': next_state.copy(),
                               'reward': reward, 'done':float(done)}

                # Store interaction in the replay memory
                self.replay_memory.append(interaction)

                state = interaction['next_state']
                episode_reward += interaction['reward']
                step_count += 1

                # Update model
                if is_training:
                    if step_count % self.steps_per_update == 0:
                        # Sample a batch of interactions and update Q_function,
                        training_batch = np.random.choice(self.replay_memory, size=self.batch_size)
                        grads, loss, q_values = self.Q_learning_update(np.asarray([x['state'] for x in training_batch]),
                                                                        np.asarray([x['action'] for x in training_batch]),
                                                                        np.asarray([x['reward'] for x in training_batch], dtype=np.float32),
                                                                        np.asarray([x['next_state'] for x in training_batch]),
                                                                        np.asarray([x['done'] for x in training_batch], dtype=np.float32),
                                                                        n_actions)

                        grads_numpy = [grad.numpy() for grad in grads]
                        self.gradients.append(grads_numpy)

                        loss_np = loss.numpy()
                        self.loss_array.append(loss_np)

                        #Convert Tensorflow Q-Values to numpy arrays
                        q_values_np = q_values.numpy()
                        self.q_values_array.append(q_values_np)

                    # Update target model
                    if step_count % self.steps_per_target_update == 0:
                        self.model_target.set_weights(self.model.get_weights())

                # Check if the episode is finished
                if interaction['done']:
                    break
                
            # Decay epsilon
            self.policy.update_epsilon()
            self.episode_reward_history.append(episode_reward)
            if episode >= necessary_episodes:
                avg_rewards = np.mean(self.episode_reward_history[-necessary_episodes:])
                if avg_rewards >= acceptance_reward and is_training:
                    print("Environment solved in {} episodes, average last {} rewards {}".format(
                        episode+1, necessary_episodes, avg_rewards))
                    if self.early_stopping:
                        break
                    else:
                        is_training = False
                if (episode+1)%50 == 0:
                    print("Episode {}/{}, average last {} rewards {}".format(
                        episode+1, self.num_episodes, necessary_episodes, avg_rewards))

    def store_pickle(self, path, filename):
        values = {'episode_reward_history': self.episode_reward_history, 'gradients': self.gradients, 'loss_array': self.loss_array, 'q_values_array': self.q_values_array,
                  'weights': self.model.get_weights()}
        with open(path + filename, 'wb') as f:
            pickle.dump(values, f)
