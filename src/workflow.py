import gym
import tensorflow as tf
import numpy as np
import utils


class PPO:
    """
    Paper: https://arxiv.org/abs/1707.06347
    """

    def __init__(self, gamma=.99, lambda_val=1.) -> None:
        self.gamma = gamma
        self.lambda_val = lambda_val
        self.alpha_1 = .1
        self.espilon = .2
        self.state_size = 5

        self.dtKeys = ['states', 'policies', 'actions', 'val_fs', 'rewards', 'dones', 'next_states']
        self.memoryDt = np.dtype([
            ('states', 'float32', (self.state_size, ))
            , ('policies', 'float32', (2, ))
            , ('actions', 'float32', (1, ))
            , ('val_fs', 'float32', (1, ))
            , ('rewards', 'float32', (1, ))
            , ('dones', 'float32', (1, ))
            , ('next_states', 'float32', (self.state_size, ))
            , ('next_val_fs', 'float32', (1, ))
        ])

        self.build_graph()

    def build_graph(self):
        with tf.variable_scope('ppo_core'):
            self.state = tf.placeholder(tf.float32, shape=[None, None, self.state_size])

            n_hid = 128

            # Init https://arxiv.org/pdf/1901.03611.pdf
            fan_in = 4
            fan_out = 128
            w_mean = 0
            w_stddev = 2 / fan_out
            W1 = tf.get_variable(
                "W1",
                [fan_in, n_hid],
                tf.float32,
                tf.random_normal_initializer(w_mean, w_stddev)
            )
            b1 = tf.get_variable("b1", [fan_out], tf.float32, tf.constant_initializer(0.))
            # import pdb; pdb.set_trace()
            a1 = tf.matmul(self.state, W1) + b1

            fan_in = 128
            fan_out = 128
            w_mean = 0
            w_stddev = 2 / fan_out
            W2 = tf.get_variable(
                "W2",
                [n_hid, n_hid],
                tf.float32,
                tf.random_normal_initializer(w_mean, w_stddev)
            )
            b2 = tf.get_variable("b2", [fan_out], tf.float32, tf.constant_initializer(0.))
            a2 = tf.matmul(a1, W2) + b2

        with tf.variable_scope('ppo_policy_head'):
            fan_in = 128
            fan_out = 2
            w_mean = 0
            w_stddev = 2 / fan_out
            W_act = tf.get_variable(
                "W_act",
                [n_hid, fan_out],
                tf.float32,
                tf.random_normal_initializer(w_mean, w_stddev)
            )
            b_act = tf.get_variable("b_act", [fan_out], tf.float32, tf.constant_initializer(0.))
            a_act = tf.matmul(a2, W_act) + b_act
            self.policy = tf.nn.softmax(a_act, axis=1)
            self.act_pred = tf.argmax(self.policy, axis=1)

        with tf.variable_scope('ppo_value_f_head'):
            fan_in = 128
            fan_out = 1
            w_mean = 0
            w_stddev = 2 / fan_out
            W_val = tf.get_variable(
                "W_val",
                [n_hid, fan_out],
                tf.float32,
                tf.random_normal_initializer(w_mean, w_stddev)
            )
            b_val = tf.get_variable("b_val", [fan_out], tf.float32, tf.constant_initializer(0.))
            self.val_pred = tf.matmul(a2, W_val) + b_val

        with tf.variable_scope('train'):
            self.ex_rewards = tf.placeholder(tf.float32, shape=[None, None, 1])
            self.policy_old = tf.placeholder(tf.float32, shape=[None, None, 2])
            self.advantages = tf.placeholder(tf.float32, shape=[None, None, 1])

            loss_vf = 1 / 2 * tf.reduce_mean(tf.square(self.val_pred - self.ex_rewards))

            r_t = self.policy[self.act_pred] / self.policy_old[self.act_pred]
            clipped_rt = tf.clip(r_t, 1 - self.epsilon, 1 + self.espilon)
            loss_clip = tf.reduce_mean(tf.min(r_t * self.advantages, clipped_rt * self.advantages))

            loss = loss_clip - self.alpha_1 * loss_vf

    def act(self, sess, obs) -> int:
        policy, act_pred, val_pred = sess.run([self.policy, self.act_pred, self.val_pred], {
            self.state: [[obs]]
        })
        return policy[0][0], act_pred[0][0], val_pred[0][0]


    # def build_datasets(self, sess, trajectories):
    #     if type(trajectories) is not np.ndarray:
    #         trajectories = np.array(trajectories)
    #
    #     # Get All the values
    #     # Trajectories dimension [bs, sequence, (obs, a, r, n_obs, done)]
    #     obss, acts, rewards, next_obss, dones = np.split(trajectories, 5, axis=2)
    #
    #
    #
    #     input_obs, initial_shape = utils.get_obs_from_traj(trajectories)
    #     values = sess.run(self.val_pred, {
    #         self.state: input_obs
    #     }).reshape(initial_shape)
    #     # ex_rewards = utils.get_expected_rewards(self.gamma, trajectories)
    #     #
    #     # advantages = ex_rewards - values


rng, seed = utils.set_all_seeds()

env = gym.make('CartPole-v0')
env.seed(seed)
act_space = env.action_space

agent = PPO()
N = 2
K = 2
batch_size = 10
max_iter = 1;

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    while t < max_iter:
        # Gather trajectories and build the dataset
        trajectories = []
        for i in range(N):
            trajectory = []
            done = False
            t = 0
            obs = env.reset()
            state = (t, ) + obs
            policy, act, val_f = agent.act(sess, state)
            while not done:
                next_obs, reward, done, _ = env.step(act)
                t += 1
                next_state = (t, ) + next_obs
                next_policy, next_act, next_val_f = agent.act(sess, next_state)

                memory = np.array([(
                    state,
                    policy
                    act,
                    val_f,
                    reward,
                    int(done),
                    next_state,
                    next_val_f
                )], dtype=agent.memoryDt)
                trajectory = np.append(trajectory, memory)

                if not done:
                    state = next_state
                    policy = next_policy
                    act = next_act
                    val_f = next_val_f

            trajectories = np.append(trajectory)

        datasets = agent.build_datasets(sess, trajectories)

        agent.train(sess, datasets, epoch=K, batch_size=bs)
