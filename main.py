import gym
import tensorflow as tf
import numpy as np
from Q_model import QModel

#//Main//#
def run_episode(model, env, replay_buffer, cumulative_steps, epsilon):
    observation_new = env.reset()
    done = False
    time = 0

    while done == False:
        #env.render()
        observation = observation_new
        action_values = sess.run([model.output], feed_dict={model.state: [observation]})

        explore = np.random.rand(1)
        if (explore < epsilon):
            action = np.random.randint(2)
        else:
            action = np.argmax(action_values)

        observation_new, reward, done, info = env.step(action)

        replay_buffer[len(replay_buffer)] = [observation, action, reward, observation_new, int(done == False)]
        time = time + 1
    return replay_buffer, cumulative_steps+time

def update_model(model, replay_buffer):

    sample_count = 1000

    #randomly draw samples from our replay_buffer based
    random_indices = np.random.permutation(len(replay_buffer))[1:min(sample_count,len(replay_buffer))]
    replay_samples = [replay_buffer.get(x) for x in random_indices]

    #residual learning with cost = (R + discount_factor*Q_old(S',A') - Q_old(S,A))^2
    discount_factor = 0.9

   
    observations = np.array([item[0] for item in replay_samples])
    actions_taken = np.array([[item[1]] for item in replay_samples])
    cur_rewards = np.array([item[2] for item in replay_samples])
    observations_new = np.array([item[3] for item in replay_samples])
    ongoing = np.array([item[4] for item in replay_samples])

    actions_values_new = sess.run(model.output, feed_dict={model.state: observations_new})

    rewards = np.expand_dims(cur_rewards + discount_factor*np.amax(actions_values_new,axis=1)*ongoing, axis=1)

    for i in range(100):
        out, loss_val = sess.run([model.opt, model.loss], feed_dict={model.state: observations, model.action_taken: actions_taken, model.label: rewards})
    print("MSE is ", loss_val)

def test(model,env):

    observation = env.reset()
    done = False
    num_steps = 0
    while done != True:
        env.render()
        num_steps = num_steps + 1
        print("taken ",num_steps," steps")
        action_values = sess.run(model.output, feed_dict={model.state: [observation]})
        best_action = np.argmax(action_values)
        observation, reward, done, info = env.step(best_action)


#//Main//#
CartPole_env = gym.make('CartPole-v0')
Q_model = QModel(dropout_keep_prob=1)

with tf.Session(graph=Q_model.graph) as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    replay_buffer = {}
    for i in range(100):
        cumulative_steps = 0
        print(i)
        for j in range(100):
            replay_buffer, cumulative_steps = run_episode(Q_model, CartPole_env, replay_buffer, cumulative_steps, 0.1)
        print("Average of ",cumulative_steps/100," steps taken.")
        update_model(Q_model,replay_buffer)

    test(Q_model, CartPole_env)