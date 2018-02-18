""" dqn """
import sys
import os
import itertools
import random
from collections import namedtuple
import gym.spaces
import numpy as np
import tensorflow as tf
from dqn_utils import *

def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# log dirs
LOGS_DIR = 'dqn_pong_logs/'
OPENAI_LOGS = LOGS_DIR + 'open_ai_logs' # to save the openai gym related logs
TENSORFLOW_MODEL_DIR = LOGS_DIR + 'model.ckpt' # to save the model snapshot
TENSORBOARD_LOG_DIR = LOGS_DIR + 'train' # to save tensorboard related logs

# make sure logs exists
#mkdir(OPENAI_LOGS)
mkdir(TENSORFLOW_MODEL_DIR)
mkdir(TENSORBOARD_LOG_DIR)

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs", "lr_schedule"])

def learn(env,
          q_func,
          optimizer_spec,
          session,
          exploration=LinearSchedule(1000000, 0.1),
          stopping_criterion=None,
          replay_buffer_size=1000000,
          batch_size=32,
          gamma=0.99,
          learning_starts=50000,
          learning_freq=4,
          frame_history_len=4,
          target_update_freq=10000,
          grad_norm_clipping=10):
    """Run Deep Q-learning algorithm.

    You can specify your own convnet using q_func.

    All schedules are w.r.t. total number of steps taken in the environment.

    Parameters
    ----------
    env: gym.Env
        gym environment to train on.
    q_func: function
        Model to use for computing the q function. It should accept the
        following named arguments:
            img_in: tf.Tensor
                tensorflow tensor representing the input image
            num_actions: int
                number of actions
            scope: str
                scope in which all the model related variables
                should be created
            reuse: bool
                whether previously created variables should be reused.
    optimizer_spec: OptimizerSpec
        Specifying the constructor and kwargs, as well as learning rate schedule
        for the optimizer
    session: tf.Session
        tensorflow session to use.
    exploration: rl_algs.deepq.utils.schedules.Schedule
        schedule for probability of chosing random action.
    stopping_criterion: (env, t) -> bool
        should return true when it's ok for the RL algorithm to stop.
        takes in env and the number of steps executed so far.
    replay_buffer_size: int
        How many memories to store in the replay buffer.
    batch_size: int
        How many transitions to sample each time experience is replayed.
    gamma: float
        Discount Factor
    learning_starts: int
        After how many environment steps to start replaying experiences
    learning_freq: int
        How many steps of environment to take between every experience replay
    frame_history_len: int
        How many past frames to include as input to the model.
    target_update_freq: int
        How many experience replay rounds (not steps!) to perform between
        each update to the target Q network
    grad_norm_clipping: float or None
        If not None gradients' norms are clipped to this value.
    """
    assert type(env.observation_space) == gym.spaces.Box
    assert type(env.action_space) == gym.spaces.Discrete

    ###############
    # BUILD MODEL #
    ###############

    if len(env.observation_space.shape) == 1:
        # This means we are running on low-dimensional observations (e.g. RAM)
        input_shape = env.observation_space.shape
    else:
        img_h, img_w, img_c = env.observation_space.shape
        input_shape = (img_h, img_w, frame_history_len * img_c)
    num_actions = env.action_space.n

    # set up placeholders
    # placeholder for current observation (or state)
    obs_t_ph = tf.placeholder(tf.uint8, [None] + list(input_shape))
    # placeholder for current action
    act_t_ph = tf.placeholder(tf.int32, [None])
    # placeholder for current reward
    rew_t_ph = tf.placeholder(tf.float32, [None])
    # placeholder for next observation (or state)
    obs_tp1_ph = tf.placeholder(tf.uint8, [None] + list(input_shape))
    # placeholder for end of episode mask
    # this value is 1 if the next state corresponds to the end of an episode,
    # in which case there is no Q-value at the next state; at the end of an
    # episode, only the current state reward contributes to the target, not the
    # next state Q-value (i.e. target is just rew_t_ph, not rew_t_ph + gamma * q_tp1)
    done_mask_ph = tf.placeholder(tf.float32, [None])

    # casting to float on GPU ensures lower data transfer times.
    obs_t_float = tf.cast(obs_t_ph, tf.float32)/255.0
    obs_tp1_float = tf.cast(obs_tp1_ph, tf.float32)/255.0

    # q function and target q function

    prediction = q_func(obs_t_float, num_actions, scope="q_func", reuse=False)

    target_prediction = q_func(obs_tp1_float, num_actions, scope="target_q_func", reuse=False)

    # greedy exploration
    predicted_action = tf.argmax(prediction, 1)

    # error
    y = tf.cond(
        tf.equal(done_mask_ph[0], 1.0),
        lambda: rew_t_ph, lambda: tf.add(rew_t_ph, tf.multiply(gamma, tf.reduce_max(target_prediction, 1))))

    total_error = tf.reduce_mean(tf.square(y - tf.reduce_max(prediction, 1)))

    q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_func')
    target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_q_func')

    # construct optimization op (with gradient clipping)
    learning_rate = tf.placeholder(tf.float32, (), name="learning_rate")
    optimizer = optimizer_spec.constructor(learning_rate=learning_rate, **optimizer_spec.kwargs)
    train_fn = minimize_and_clip(
        optimizer,
        total_error,
        var_list=q_func_vars,
        clip_val=grad_norm_clipping
    )

    # update_target_fn will be called periodically to copy Q network to target Q network
    update_target_fn = []
    for var, var_target in zip(sorted(q_func_vars, key=lambda v: v.name),
                               sorted(target_q_func_vars, key=lambda v: v.name)):
        update_target_fn.append(var_target.assign(var))
    update_target_fn = tf.group(*update_target_fn)

    # construct the replay buffer
    replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)

    ###############
    # RUN ENV     #
    ###############
    model_initialized = False
    num_param_updates = 0
    mean_episode_reward = -float('nan')
    best_mean_episode_reward = -float('inf')
    last_obs = env.reset()
    LOG_EVERY_N_STEPS = 10000
    init = tf.global_variables_initializer()
    session.run(init)

    # some book keeping stuff
    # episode wise rewards
    summary_reward_sum = tf.placeholder("float")
    tf.summary.scalar('EpisodeReward', summary_reward_sum)
    # episode wise learning rate
    tf.summary.scalar('LearningRate', learning_rate)
    # episode wise exploration factior
    summary_exploration = tf.placeholder("float")
    tf.summary.scalar('Exploration', summary_exploration)
    # global time step count summary
    summary_time_step = tf.placeholder('int32')
    tf.summary.scalar('GlobalStepNumber', summary_time_step)
    # global episode count summary
    summary_episode_count = tf.placeholder('int16')
    tf.summary.scalar('EpisodeCount', summary_episode_count)

    train_writer = tf.summary.FileWriter(TENSORBOARD_LOG_DIR, session.graph)
    merged = tf.summary.merge_all()
    model_saver = tf.train.Saver(max_to_keep=100)

    for t in itertools.count():
        ### 1. Check stopping criterion
        if stopping_criterion is not None and stopping_criterion(env, t):
            break


        ### 2. Step the env and store the transition
        idx = replay_buffer.store_frame(last_obs)
        action = session.run(
            predicted_action,
            feed_dict={
                obs_t_ph:replay_buffer.encode_recent_observation().reshape(1, 84, 84, 4)
            }
        )[0]

        if random.random() < exploration.value(t):
            action = env.action_space.sample()

        obs, reward, done, info = env.step(action)
        replay_buffer.store_effect(idx, action, reward, done)
        if done:
            _rewards = get_wrapper_by_name(env, "Monitor").get_episode_rewards()
            summary = session.run(
                merged,
                feed_dict={
                    summary_reward_sum: _rewards[-1],
                    learning_rate: optimizer_spec.lr_schedule.value(t),
                    summary_exploration: exploration.value(t),
                    summary_time_step: t,
                    summary_episode_count: len(_rewards)
                }
            )
            train_writer.add_summary(summary, t)
            #print  get_wrapper_by_name(env, "Monitor").get_episode_rewards()[-1]
            obs = env.reset()


        last_obs = obs

        ### 3. Perform experience replay and train the network.

        if (t > learning_starts and
                t % learning_freq == 0 and
                replay_buffer.can_sample(batch_size)):
            # Here, you should perform training. Training consists of four steps:
            # 3.a use the replay buffer to sample a batch of transitions
            obs_t_batch, act_batch, rew_batch, obs_tp1_batch, done_mask = replay_buffer.sample(batch_size)
            # 3.b initialize the model if it has not been initialized yet
            if not model_initialized:
                initialize_interdependent_variables(
                    session,
                    tf.global_variables(),
                    {
                        obs_t_ph: obs_t_batch,
                        obs_tp1_ph: obs_tp1_batch,
                    }
                )
                model_initialized = True
            #print 'Trainging in progress Current Step : ',t
            # 3.c train the model.
            session.run(
                train_fn,
                feed_dict={
                    obs_t_ph : obs_t_batch,
                    act_t_ph : act_batch,
                    rew_t_ph : rew_batch,
                    obs_tp1_ph : obs_tp1_batch,
                    done_mask_ph : done_mask,
                    learning_rate: optimizer_spec.lr_schedule.value(t)
                })

            # 3.d periodically update the target network by calling
            num_param_updates += 1
            if num_param_updates % target_update_freq == 0:
                #print 'Tareget Q func update in progress Current Step : ',t
                session.run(update_target_fn)


        ### 4. Log progress
        episode_rewards = get_wrapper_by_name(env, "Monitor").get_episode_rewards()

        if len(episode_rewards) > 0:
            mean_episode_reward = np.mean(episode_rewards[-100:])
        if len(episode_rewards) > 100:
            best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)
        if t % LOG_EVERY_N_STEPS == 0 and model_initialized:
            print("Timestep %d" % (t,))
            print("mean reward (100 episodes) %f" % mean_episode_reward)
            print("best mean reward %f" % best_mean_episode_reward)
            print("episodes %d" % len(episode_rewards))
            print("exploration %f" % exploration.value(t))
            print("learning_rate %f" % optimizer_spec.lr_schedule.value(t))
            sys.stdout.flush()
            # save the model 
            model_saver.save(session, TENSORFLOW_MODEL_DIR, global_step=t)

def run(env, q_func, session, model_path,
    replay_buffer_size=1000000, frame_history_len=4, max_episode_count=500,
    render=False):
    """Run Deep Q-learning algorithm.

    You can specify your own convnet using q_func.

    All schedules are w.r.t. total number of steps taken in the environment.

    Parameters
    ----------
    env: gym.Env
        gym environment to train on.
    q_func: function
        Model to use for computing the q function. It should accept the
        following named arguments:
            img_in: tf.Tensor
                tensorflow tensor representing the input image
            num_actions: int
                number of actions
            scope: str
                scope in which all the model related variables
                should be created
            reuse: bool
                whether previously created variables should be reused.
    session: tf.Session
        tensorflow session to use.
    model_path: path to the trained model
    replay_buffer_size: int
        How many memories to store in the replay buffer.
    frame_history_len: int
        How many past frames to include as input to the model.
    max_episode_count: int
        maximum number of episodes to run
    """
    assert type(env.observation_space) == gym.spaces.Box
    assert type(env.action_space) == gym.spaces.Discrete

    ###############
    # BUILD MODEL #
    ###############

    if len(env.observation_space.shape) == 1:
        # This means we are running on low-dimensional observations (e.g. RAM)
        input_shape = env.observation_space.shape
    else:
        img_h, img_w, img_c = env.observation_space.shape
        input_shape = (img_h, img_w, frame_history_len * img_c)
    num_actions = env.action_space.n

    # set up placeholders
    # placeholder for current observation (or state)
    obs_t_ph = tf.placeholder(tf.uint8, [None] + list(input_shape))

    # casting to float on GPU ensures lower data transfer times.
    obs_t_float = tf.cast(obs_t_ph, tf.float32)/255.0

    prediction = q_func(obs_t_float, num_actions, scope="q_func", reuse=False)

    # greedy exploration
    predicted_action = tf.argmax(prediction, 1)

    # construct the replay buffer
    replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)

    ###############
    # RUN ENV     #

    last_obs = env.reset()

    init = tf.global_variables_initializer()
    session.run(init)

    episode_count = 0

    saver = tf.train.Saver()
    with tf.Session() as session:
        saver.restore(session, model_path)

        for t in itertools.count():

            if episode_count > max_episode_count:
                break

            idx = replay_buffer.store_frame(last_obs)
            action = session.run(
                predicted_action,
                feed_dict={
                    obs_t_ph:replay_buffer.encode_recent_observation().reshape(1, 84, 84, 4)
                }
            )[0]

            if random.random() < 0.01:
                action = env.action_space.sample()

            obs, reward, done, _ = env.step(action)
            replay_buffer.store_effect(idx, action, reward, done)
            if done:
                obs = env.reset()
                episode_count += 1

            if render:
                env.render()
                
            last_obs = obs
    
