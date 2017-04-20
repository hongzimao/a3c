import numpy as np
import tensorflow as tf
import gym
import a3c


S_DIM = 4
A_DIM = 2
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
RAND_RANGE = 1000
# NN_MODEL = './models/nn_model_eps_len_300.ckpt'
NN_MODEL = './models/nn_model_force_100.ckpt'


def main():

    env = gym.make("CartPole-v0")
    # env.force_mag = 100.0

    with tf.Session() as sess:
        actor = a3c.ActorNetwork(sess, state_dim=S_DIM, action_dim=A_DIM, learning_rate=ACTOR_LR_RATE)
        critic = a3c.CriticNetwork(sess, state_dim=S_DIM, learning_rate=CRITIC_LR_RATE)
        saver = tf.train.Saver()
        saver.restore(sess, NN_MODEL)
    
        for eps in xrange(100):
            obs = env.reset()
            reward = 0
            for _ in range(300):
                env.render()

                action_prob = actor.predict(np.reshape(obs, (1, S_DIM)))
                action_cumsum = np.cumsum(action_prob)
                a = (action_cumsum > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()

                obs, rew, done, info = env.step(a)

                reward += rew
                if done:
                    break
            print eps, reward, done


if __name__ == '__main__':
    main()
