# CartPole
CartPole with DQN

Standard implementation of Deep Q-Learning with OpenAI's Gym Environment.
An epsilon-greedy exploration policy is used alongside a cyclic episodic buffer 
where old experiences are overwritten by newer ones. Currently observing stochastic
convergence behavior.

Experimenting with: 
Sampling of episodes weighted by their difference in expected vs. actual rewards.
