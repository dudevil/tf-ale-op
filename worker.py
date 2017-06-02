import tensorflow as tf
import threading
import numpy as np
from aleop import ale
from model import _process_frame, FFPolicy


def discount(tensor, gamma=.9):
    return tf.scan(lambda a, x: a * gamma + x, tensor[::-1])[::-1]


def stack_states_channel(states):
    """
    Copies and concatenates three consecutive ale frames in the channel dimensions
    e.g states 0, 1, 2, 3
    become states
    [0 0 0]
    [0 0 1]
    [0 1 2]
    [1 2 3]
    """
    states_1 = tf.concat([states[:1], states[:-1]], 0)
    states_2 = tf.concat([states[:1], states[:1], states[:-2]], 0)
    return tf.concat([states_2, states_1, states], 3)


class AleThread(threading.Thread):

    def __init__(self,
                 queue,
                 coord,
                 session,
                 summary_writer,
                 env='breakout',
                 action_space=4,
                 ale_kwargs={'frameskip_min': 4, 'frameskip_max': 4}):
        threading.Thread.__init__(self)
        self.daemon = True
        self.queue = queue
        self.env_name = env
        self.ale_kwargs = ale_kwargs
        self.coord = coord
        self.action_space = action_space
        self.sess = session
        self.summary_writer = summary_writer
        
    # def start_ale(self, sess, summary_writer):
    #     self.start()
        
    def rollout(self, max_step=10, gamma=0.9):
        
        rewards_arr = tf.TensorArray(tf.float32, size=1, dynamic_size=True, element_shape=(), name="rewards")
        actions_arr = tf.TensorArray(tf.float32, size=1, dynamic_size=True, element_shape=(1, self.policy.action_space), 
                                 name="actions") 
        states_arr = tf.TensorArray(tf.float32, size=1, dynamic_size=True, element_shape=(42, 42, 1), name="states")
        values_arr = tf.TensorArray(tf.float32, size=1, dynamic_size=True, element_shape=(), name="values")
        done = tf.Variable(initial_value=False, trainable=False, dtype=tf.bool, expected_shape=())
        step = 0

        def body(step, states, rewards, values, actions, done):
            action_ohe, value = self.policy.act()
            action = tf.cast(tf.argmax(action_ohe, axis=1), tf.int32)[0]
            r, d, s = ale(action, self.env_name, **self.ale_kwargs)
            n_state = self.state.assign(_process_frame(s))
            
            states = states_arr.write(step, n_state[0])
            rewards = rewards_arr.write(step, r)
            values = values_arr.write(step, value[0])
            actions = actions_arr.write(step, action_ohe)
            return step + 1, states, rewards, values, actions, d
        
        def cond(step, states, rewards, values, actions, done):
            return tf.logical_and(step < max_step, tf.logical_not(done))
    
        step, states, rewards, values, actions, done = tf.while_loop(
            cond, 
            body, 
            [step, states_arr, rewards_arr, values_arr, actions_arr, done], 
            parallel_iterations=1, 
            back_prop=False)
        
        final_r = tf.cond(done, 
                          lambda: self.policy.value()[0],
                          lambda: tf.convert_to_tensor(0., dtype=tf.float32))
    
        rewards.write(step, final_r)
        all_rewards = rewards.stack()
        
        values.write(step, final_r)
        all_values = values.stack()
        delta_t = all_rewards[:-1] + gamma * all_values[1:] - all_values[:-1]
    
        batch_adv = discount(delta_t, gamma=gamma)
        batch_rewards = discount(all_rewards, gamma=gamma)
        batch_actions = actions.stack()
        
        batch_states = stack_states_channel(states.stack())
        return step, batch_states, batch_actions, batch_adv, batch_rewards, done

    def run(self):
        with self.sess.as_default():
            done = tf.Variable(initial_value=False, trainable=False, dtype=tf.bool, expected_shape=())
            self.state = tf.Variable(initial_value=np.zeros((1, 42, 42, 1)), trainable=False, dtype=tf.float32)
            ep_reward = tf.Variable(initial_value=0.0, trainable=False, dtype=tf.float32, expected_shape=())
            
            self.policy = FFPolicy(self.state, self.action_space)
            
            step, bs, ba, badv, br, done = self.rollout()
            
            enqueue_op = self.queue.enqueue_many([bs, ba, badv, br])
            update_reward = tf.assign_add(tf.reduce_sum(br))

            do_rollout = tf.group([enqueue_op, update_reward])
            reward_summary = tf.summary_scalar('reward_per_episode', ep_reward)
            with tf.control_dependecies([reward_summary]):
                reset_reward = tf.assign(ep_reward, 0.)
                
            while not self.coord.should_stop():
                terminal = self.sess.run([do_rollout, done])
                if terminal:
                    self.sess.run(reset_reward)

