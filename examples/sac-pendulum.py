from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Concatenate

import matplotlib.pyplot as plt
import gym

import huskarl as hk

if __name__ == "__main__":

	# Setup gym environment
	create_env = lambda: gym.make('Pendulum-v0')
	dummy_env = create_env()
	action_size = dummy_env.action_space.shape[0]
	state_shape = dummy_env.observation_space.shape


	# Create Deep Deterministic Policy Gradient agent
	agent = hk.agent.SAC(action_dim=action_size,state_dim=state_shape,nsteps=2)

	def plot_rewards(episode_rewards, episode_steps, done=False):
		plt.clf()
		plt.xlabel('Step')
		plt.ylabel('Reward')
		for ed, steps in zip(episode_rewards, episode_steps):
			plt.plot(steps, ed)
		plt.show() if done else plt.pause(0.001) # Pause a bit so that the graph is updated

	# Create simulation, train and then test
	sim = hk.Simulation(create_env, agent)
	sim.train(max_steps=30_000, visualize=True, plot=plot_rewards)
	sim.test(max_steps=5_000)

