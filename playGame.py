# -*- coding: utf-8 -*-
from osim.env import *
from ddpg import *
import math

# np.random.seed(1337)

def playGame():
    train_indicator = my_config.is_training
    env = RunEnv(visualize = True)
    agent = DDPG(env)

    episode_count = my_config.max_eps
    step = 0
    best_reward = 0

    my_config.logger.warn("Opensim Experiment Start.")
    for episode in range(episode_count):
        s_t = env.reset()
        # s_t = np.divide(s_t, env.observation_space.high)
        total_reward = 0.
        step_eps = 0.
        done = False

        while not done:
            #Take noisy actions durinagentg training
            if (train_indicator):
                a_t = agent.noise_action(s_t)
            else:
                a_t = agent.action(s_t)

            s_t1, r_t, done, info = env.step(a_t)
            # my_config.logger.debug("before, s_t1: %s" % (s_t1))
            # s_t1 = np.divide(s_t1, env.observation_space.high)
            # my_config.logger.debug("after, s_t1: %s" % (s_t1))
            shaped_r_t = shape_reward(s_t, s_t1, r_t)

            if (train_indicator):
                # agent.perceive(s_t,a_t,r_t,s_t1,done)
                agent.perceive(s_t, a_t, shaped_r_t, s_t1, done)
                
            #Cheking for nan rewards
            if ( math.isnan( r_t )):
                r_t = 0.0
                for bad_r in range( 50 ):
                    my_config.logger.warn('Bad Reward Found')

            total_reward += r_t
            s_t = s_t1

            # if (np.mod(step, 1000) == 0):
            #     my_config.logger.debug("episode: %s, step: %s, action: %s, reward: %s" % (episode, step_eps, a_t, r_t))

            step += 1
            step_eps += 1

        # my_config.logger.info("episode: %s, step_eps: %s, step: %s, reward: %s, replay buffer: %s" % (episode, step_eps, step, total_reward, agent.replay_buffer.count()))

        #Saving the best model.
        if total_reward >= best_reward :
            if (train_indicator):
                my_config.logger.info("Now we save model with reward %s, previous best reward was %s" % (str(total_reward), str(best_reward)))
                best_reward = total_reward
                agent.saveNetwork()

                total_reward_test = 0
                for i in xrange(my_config.test_eps):
                    state_test = env.reset()
                    done_test = False
                    while not done_test:
                        # env.render()
                        action_test = agent.action(state_test)  # direct action for test
                        state_test, reward_test, done_test, _ = env.step(action_test)
                        total_reward_test += reward_test
                        # my_config.logger.debug("test action: %s, reward: %s, total reward: %s" % (action_test, reward_test, total_reward_test))
                ave_reward = total_reward_test / my_config.test_eps
                my_config.logger.info("Episode: %s, Evaluation Average Reward: %s" % (episode, ave_reward))

    my_config.logger.warn("Finish...")

# state is of 41 dimension:
# [1.3641781712711316, -0.7965535680856243, 0.6399285380684508,     #骨盆位置
# 3.0440460726366663, -2.1398679023503377, -2.018546508259545,      #骨盆速度
# -0.6210047028490362, 0.2962524969799402, -0.7459638431997042, -0.6209773830501356, 0.30311605595508373, -0.7584414311172483,    #角度
# 0.4091856096497938, -0.27004629315680106, 0.06147351142197961, 0.18420892486505097, -0.05981841011323434, 0.10561122627778939,     #角速度
# -0.8982108618405059, 0.5193824863386319, -1.750385255207639, -2.3052817903385145,      #质心
# -1.2879790284749424, 0.9235422323494551, -0.7965535680856243, 0.6399285380684508, -0.8035612663287532, 0.6630821937307102,     #头，盆骨，躯干位置
# 0.031531889786680894, 0.013214694425566626, 0.028878999343764034, 0.010414535295682717,   #脚趾位置
# -0.10301840786663219, 0.04033702473483097, -0.10577346949965855, 0.03702498501700663,     #脚踝位置
# 1.0677326347878326, 1.023293151924051,
# 2.9394444913646733, 0.012830400892693024, 0.20750385713624836]
# 通过分析state前后两帧的距离和速度，两帧间的间隔为0.01秒
GRAVITY_A = 9.8
SEC_PER_FRAME = 0.01
def shape_reward(state0, state, reward):
    x_mc = state[18]
    v_mc = state[20]
    shaped_reward = reward + v_mc
    v_mc0 = state0[20]
    a_mc = (v_mc - v_mc0) / SEC_PER_FRAME
    y_mc = state[19]
    x_zmp = x_mc - y_mc * a_mc / GRAVITY_A
    x_left_toe = state[26]
    x_right_toe = state[28]
    x_left_talus = state[30]
    x_right_talus = state[32]
    x_talus_min = min([x_left_talus, x_right_talus])
    x_toe_max = max([x_left_toe, x_right_toe])

    if x_zmp <= x_talus_min:
        shaped_reward -= (x_talus_min - x_zmp )
    elif x_zmp >= x_toe_max:
        shaped_reward -= (x_zmp - x_toe_max)

    return shaped_reward

if __name__ == "__main__":
    # my_config.init()
    playGame()