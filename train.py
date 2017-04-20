import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import logging
import numpy as np
import multiprocessing as mp
import tensorflow as tf
import gym
import a3c


S_DIM = 4
A_DIM = 2
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
NUM_AGENTS = 16
TRAIN_SEQ_LEN = 500  # take as a train batch
TRAIN_EPOCH = 10000
MODEL_SAVE_INTERVAL = 100
RANDOM_SEED = 42
RAND_RANGE = 1000
SUMMARY_DIR = './results'
MODEL_DIR = './models'
TRAIN_TRACES = './cooked_traces/'
# NN_MODEL = './results/nn_model_ep_10800.ckpt'
NN_MODEL = None


def central_agent(net_params_queues, exp_queues):

    assert len(net_params_queues) == NUM_AGENTS
    assert len(exp_queues) == NUM_AGENTS

    with tf.Session() as sess, open(SUMMARY_DIR + '/log_central', 'wb') as log_file:

        actor = a3c.ActorNetwork(sess, state_dim=S_DIM, action_dim=A_DIM, learning_rate=ACTOR_LR_RATE)
        critic = a3c.CriticNetwork(sess, state_dim=S_DIM, learning_rate=CRITIC_LR_RATE)

        summary_ops, summary_vars = a3c.build_summaries()

        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)  # training monitor
        saver = tf.train.Saver()  # save neural net parameters

        # restore neural net parameters
        nn_model = NN_MODEL
        if nn_model is not None:  # nn_model is the path to file
            saver.restore(sess, nn_model)
            print("Model restored.")

        # while True:  # assemble experiences from agents, compute the gradients
        for ep in xrange(TRAIN_EPOCH): 
            # synchronize the network parameters of work agent
            actor_net_params = actor.get_network_params()
            critic_net_params = critic.get_network_params()
            for i in xrange(NUM_AGENTS):
                net_params_queues[i].put([actor_net_params, critic_net_params])

            # record average reward and td loss change
            # in the experiences from the agents
            total_batch_len = 0.0
            total_reward = 0.0
            total_td_loss = 0.0
            total_agents = 0.0 

            # assemble experiences from the agents
            actor_gradient_batch = []
            critic_gradient_batch = []

            for i in xrange(NUM_AGENTS):
                s_batch, a_batch, r_batch, terminal = exp_queues[i].get()

                actor_gradient, critic_gradient, td_batch = \
                    a3c.compute_gradients(
                        s_batch=np.vstack(s_batch),
                        a_batch=np.vstack(a_batch),
                        r_batch=np.vstack(r_batch),
                        terminal=terminal, actor=actor, critic=critic)

                actor_gradient_batch.append(actor_gradient)
                critic_gradient_batch.append(critic_gradient)

                total_reward += np.sum(r_batch)
                total_td_loss += np.sum(td_batch)
                total_batch_len += len(r_batch)
                total_agents += 1.0

            # compute aggregated gradient
            assert NUM_AGENTS == len(actor_gradient_batch)
            assert len(actor_gradient_batch) == len(critic_gradient_batch)

            for i in xrange(len(actor_gradient_batch)):
                actor.apply_gradients(actor_gradient_batch[i])
                critic.apply_gradients(critic_gradient_batch[i])

            # log training information
            avg_reward = total_reward  / total_agents
            avg_td_loss = total_td_loss / total_batch_len

            log_file.write('Epoch: ' + str(ep) +
                         ' TD_loss: ' + str(avg_td_loss) +
                         ' Avg_reward: ' + str(avg_reward) + '\n')
            log_file.flush()

            summary_str = sess.run(summary_ops, feed_dict={
                summary_vars[0]: avg_td_loss,
                summary_vars[1]: avg_reward
            })

            writer.add_summary(summary_str, ep)
            writer.flush()

            if ep % MODEL_SAVE_INTERVAL == 0:
                # Save the neural net parameters to disk.
                save_path = saver.save(sess, MODEL_DIR + "/nn_model_ep_" +
                                       str(ep) + ".ckpt")


def agent(agent_id, net_params_queue, exp_queue):

    env = gym.make("CartPole-v0")
    env.force_mag = 100.0

    with tf.Session() as sess, open(SUMMARY_DIR + '/log_agent_' + str(agent_id), 'wb') as log_file:
        actor = a3c.ActorNetwork(sess,
                                 state_dim=S_DIM, action_dim=A_DIM,
                                 learning_rate=ACTOR_LR_RATE)
        critic = a3c.CriticNetwork(sess,
                                   state_dim=S_DIM,
                                   learning_rate=CRITIC_LR_RATE)

        # initial synchronization of the network parameters from the coordinator
        actor_net_params, critic_net_params = net_params_queue.get()
        actor.set_network_params(actor_net_params)
        critic.set_network_params(critic_net_params)

        time_stamp = 0
        for ep in xrange(TRAIN_EPOCH): 

            obs = env.reset()

            s_batch = []
            a_batch = []
            r_batch = []

            for step in xrange(TRAIN_SEQ_LEN):

                s_batch.append(obs)

                action_prob = actor.predict(np.reshape(obs, (1, S_DIM)))
                action_cumsum = np.cumsum(action_prob)
                a = (action_cumsum > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()

                action_vec = np.zeros(A_DIM)
                action_vec[a] = 1
                a_batch.append(action_vec)

                obs, rew, done, info = env.step(a)

                r_batch.append(rew)

                if done:
                    break

            exp_queue.put([s_batch, a_batch, r_batch, done])

            actor_net_params, critic_net_params = net_params_queue.get()
            actor.set_network_params(actor_net_params)
            critic.set_network_params(critic_net_params)

            log_file.write('epoch' + str(ep) + 'reward' + str(np.sum(rew)) + 'step' + str(len(r_batch)))
            log_file.flush()


def main():

    np.random.seed(RANDOM_SEED)

    # inter-process communication queues
    net_params_queues = []
    exp_queues = []
    for i in xrange(NUM_AGENTS):
        net_params_queues.append(mp.Queue(1))
        exp_queues.append(mp.Queue(1))

    # create a coordinator and multiple agent processes
    # (note: threading is not desirable due to python GIL)
    coordinator = mp.Process(target=central_agent,
                             args=(net_params_queues, exp_queues))
    coordinator.start()

    agents = []
    for i in xrange(NUM_AGENTS):
        agents.append(mp.Process(target=agent,
                                 args=(i,
                                       net_params_queues[i],
                                       exp_queues[i])))
    for i in xrange(NUM_AGENTS):
        agents[i].start()

    # wait unit training is done
    coordinator.join()


if __name__ == '__main__':
    main()
