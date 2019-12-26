import numpy
import tensorflow as tf
import utilz
import agent
import zipfile
import json


from tensorflow.keras.layers import Input, Concatenate, Dense, Flatten, LSTM
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MSE

import numpy as np

from huskarl.policy import EpsGreedy, Greedy
from huskarl.core import Agent, HkException
from huskarl import memory, save_utils

"""
SAC (Soft Actor Critic)
"""

class SAC(agent.Agent):
    def __init__(self, policy=None, actor=None, critic_q1=None, critic_q2=None, optimizer_critics=None, optimizer_actor=None,
                 test_policy=None, memsize=1000, target_update=1e-3, action_dim=None, state_dim=None,
				 gamma=0.99, batch_size=32, nsteps=1, epochs=100, learning_rate=3e-4, policy_frequency=2):

        self.actor = self.build_actor if actor is None else actor
        self.critic_q1 = self.build_critic if critic_q1 is None else critic_q1
        self.critic_q2 = self.build_critic if critic_q2 is None else critic_q2
        self.learning_rate = learning_rate

        self.action_dim = action_dim
        self.state_dim = state_dim

        self.optimizer_actor = Adam(lr=self.learning_rate) if optimizer_actor is None else optimizer_actor
        self.optimizer_critics = Adam(lr=self.learning_rate) if optimizer_critics is None else optimizer_critics

        self.policy = EpsGreedy(0.1) if policy is None else policy
        self.test_policy = EpsGreedy(0.1) if test_policy is None else test_policy

        self.memsize = memsize
        self.memory = memory.PrioritizedExperienceReplay(memsize, nsteps, prob_alpha=0.2)

        self.target_update = target_update
        self.gamma = gamma
        self.batch_size = batch_size
        self.nsteps = nsteps
        self.training = True

		# Clone models to use for delayed Q targets
        self.target_actor = tf.keras.models.clone_model(self.actor)
        self.target_critic_q1 = tf.keras.models.clone_model(self.critic_q1)
        self.target_critic_q2 = tf.keras.models.clone_model(self.critic_q2)


        # self.actor = build_actor()
        # self.actor_target = build_actor()
		# Define loss function that computes the MSE between target Q-values and cumulative discounted rewards

		# Define loss function that computes the MSE between target Q-values and cumulative discounted rewards
		# If using PrioritizedExperienceReplay, the loss function also computes the TD error and updates the trace priorities
        def q_loss(data, qvals):
            """Computes the MSE between the Q-values of the actions that were taken and	the cumulative discounted
			rewards obtained after taking those actions. Updates trace priorities if using PrioritizedExperienceReplay.
			"""
            target_qvals = data[:, 0, np.newaxis]
            if isinstance(self.memory, memory.PrioritizedExperienceReplay):
                def update_priorities(_qvals, _target_qvals, _traces_idxs):
                    """Computes the TD error and updates memory priorities."""
                    td_error = np.abs((_target_qvals - _qvals).numpy())[:, 0]
                    _traces_idxs = (tf.cast(_traces_idxs, tf.int32)).numpy()
                    self.memory.update_priorities(_traces_idxs, td_error)
                    return _qvals
                qvals = tf.py_function(func=update_priorities, inp=[qvals, target_qvals, data[:, 1]], Tout=tf.float32)
            return MSE(target_qvals, qvals)


    def save(self, save_path, overwrite=False):
        """Saves the model parameters to the specified file."""
        data = {
            "learning_rate": self.learning_rate,
            "buffer_size": self.buffer_size,
            "learning_starts": self.learning_starts,
            "train_freq": self.train_freq,
            "batch_size": self.batch_size,
            "tau": self.tau,
            "ent_coef": self.ent_coef if isinstance(self.ent_coef, float) else 'auto',
            "target_entropy": self.target_entropy,
            "replay_buffer": self.memory,
            "gamma": self.gamma,
            "verbose": self.verbose,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "policy": self.policy,
            "n_envs": self.n_envs,
            "n_cpu_tf_sess": self.n_cpu_tf_sess,
            "seed": self.seed,
            "action_noise": self.action_noise,
            "random_exploration": self.random_exploration,
            "_vectorize_action": self._vectorize_action,
            "policy_kwargs": self.policy_kwargs
        }
        serialized_data = save_utils.data_to_json(data)

        # Check postfix if save_path is a string
        with open('data.json', 'w', encoding='utf-8') as f:
            json.dump(serialized_data, f, ensure_ascii=False, indent=4)

        self.actor.save(filename, overwrite=overwrite)
        self.critic_q1.save(filename, overwrite=overwrite)
        self.critic_q2.save(filename, overwrite=overwrite)

    def load(self, filename):
        """Loads the model from a specified file"""
        data = save_utils.json_to_data(filename)
        model.__dict__.update(data)

        self.actor.load_model(filename)
        self.critic_q1.load_model(filename)
        self.critic_q2.load_model(filename)

    def act(self, state, instance=0):
        """Returns the action to be taken given a state."""
        action = self.actor.predict(np.array([state]))[0]
        return self.policy.act(action) if self.training else self.test_policy.act(action)

    #build the actor
    def build_actor(self):
        inputs = Input(shape=self.state_dim, name='autoencoded_observations')
        actor = Dense(self.action_dim, kernel_initializer='random_uniform', activation='tanh')(inputs) 
        actor = Model(inputs=inputs, outputs=actor)
        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        actor.compile(loss='mean_squared_error', optimizer=adam)
        return actor

    #build the critics
    def build_critic(self):
        action_input = Input(shape=(self.action_dim,), name='action_input')
        observation_input = Input(shape=self.state_dim, name='observation_input')
        inputs = Concatenate()([action_input, observation_input])
        critic = Dense(64, activation='relu', kernel_initializer='random_uniform', name='Dense_layer_1')(inputs)
        critic = Dense(32, activation='relu', kernel_initializer='random_uniform', name='Dense_layer_2')(critic)
        critic = Dense(32, activation='relu', kernel_initializer='random_uniform', name='Dense_layer_3')(critic)
        critic = Dense(60, activation='linear', kernel_initializer='random_uniform')(critic)
        return Model(inputs=[action_input, observation_input], outputs=critic)

    def build(self):
        self.actor = self.build_actor()
        self.actor_target = self.build_actor()

        self.critic = self.build_critic()
        self.critic_target = self.build_critic()

    def train(self, elf, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        for it in range(iterations):
            """Trains the agent for one step."""
            if len(self.memory) == 0:
                return
            
            # Train even when memory has fewer than the specified batch_size
            batch_size = min(len(self.memory), self.batch_size)

            # Sample batch_size traces from memory
            state_batch, action_batch, reward_batches, end_state_batch, not_done_mask = self.memory.get(batch_size)
            # Select action according to policy and add clipped noise 

            # Compute the target Q value



            # Get current Q estimates

            # Compute critic loss


            pass

