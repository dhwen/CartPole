import gym
import tensorflow as tf
import numpy as np
from Q_model import QModel

#//Main//#
def run_episode(model, env, replay_buffer, replay_idx, cumulative_steps, epsilon):
    observation_new = env.reset()
    done = False
    time = 0

    while done == False:
        #env.render()
        observation = observation_new
        action_values = sess.run([model.output], feed_dict={model.state: [observation]})

        explore = np.random.rand(1) < epsilon
        if (explore):
            action = np.random.randint(2)
        else:
            action = np.argmax(action_values)

        observation_new, reward, done, info = env.step(action)

        action_values_new = sess.run([model.output], feed_dict={model.state: [observation_new]})
        surprisal = np.exp(1/10*np.abs(reward + np.max(action_values_new) - action_values[0][0][action]))
        #surprisal = 1

        replay_idx = replay_idx % 10000

        if(len(replay_buffer) < 10000):
            replay_buffer.append([observation, action, reward, observation_new, int(done == False), surprisal])
        else:
            replay_buffer[replay_idx] = [observation, action, reward, observation_new, int(done == False), surprisal]

        time = time + 1
    return replay_buffer, cumulative_steps+time

def update_model(model, replay_buffer):

    sample_count = 1000

    #randomly draw samples from our replay_buffer based on surprisal
    selection_weights = np.array([replay_buffer[x][5] for x in range(len(replay_buffer))])
    selection_weights = selection_weights/sum(selection_weights)
    random_indices = np.random.choice(a=len(replay_buffer), size=min(sample_count, len(replay_buffer)), replace=False, p=selection_weights)

    #residual learning with cost = (R + discount_factor*Q_old(S',A') - Q_old(S,A))^2
    discount_factor = 0.9

    observations = np.array([replay_buffer[x][0] for x in random_indices])
    actions_taken = np.array([[replay_buffer[x][1]] for x in random_indices])
    cur_rewards = np.array([replay_buffer[x][2] for x in random_indices])
    observations_new = np.array([replay_buffer[x][3] for x in random_indices])
    ongoing = np.array([replay_buffer[x][4] for x in random_indices])

    actions_values_new = sess.run(model.output, feed_dict={model.state: observations_new})

    rewards = np.expand_dims(cur_rewards + discount_factor*np.amax(actions_values_new,axis=1)*ongoing, axis=1)

    for i in range(1000):
        out, loss_val = sess.run([model.opt, model.loss], feed_dict={model.state: observations, model.action_taken: actions_taken, model.label: rewards, model.bIsTrain : True})
    print("MSE is ", loss_val)

    actions_values = sess.run(model.output, feed_dict={model.state: observations})
    actions_values_new = sess.run(model.output, feed_dict={model.state: observations_new})

    i = 0
    for random_index in random_indices:
        replay_buffer[random_index][5] = np.exp(1/10*np.abs(rewards[i] + np.max(actions_values_new[i]) - actions_values[i][actions_taken[i][0]]))[0]
        #replay_buffer[random_index][5] = 1
        i = i + 1

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
Q_model = QModel(drop_prob=0)

with tf.Session(graph=Q_model.graph) as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    replay_buffer = []
    replay_idx = 0
    for i in range(100):
        cumulative_steps = 0
        print(i)
        for j in range(100):
            replay_buffer, cumulative_steps = run_episode(Q_model, CartPole_env, replay_buffer, replay_idx, cumulative_steps, 0.1)
        print("Average of ",cumulative_steps/100," steps taken.")
        update_model(Q_model,replay_buffer)

    test(Q_model, CartPole_env)