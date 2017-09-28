
from osim.env import *
from ddpg import *
import math
import traj
import reward_shaping

# np.random.seed(1337)

def playGame():
    train_indicator = my_config.is_training
    env = RunEnv(visualize = True)
    agent = DDPG(env)

    traj.generate(env, agent)

    episode_count = my_config.max_eps
    step = 0
    best_reward = 0
    best_shaped_reward = 0

    my_config.logger.warn("Opensim Experiment Start.")
    for episode in range(episode_count):
        s_t = env.reset(difficulty=0)
        # s_t = np.divide(s_t, env.observation_space.high)
        total_reward = 0.
        total_shaped_reward = 0.
        step_eps = 0.
        done = False

        while not done:
            #Take noisy actions durinagentg training
            if (train_indicator):
                a_t = agent.noise_action(s_t)
            else:
                a_t = agent.action(s_t)

            s_t1, r_t, done, info = env.step(a_t)
            shaped_r_t = reward_shaping.shape(s_t, s_t1, r_t)

            if (train_indicator):
                # agent.perceive(s_t,a_t,r_t,s_t1,done)
                agent.perceive(s_t, a_t, shaped_r_t, s_t1, done)
                
            #Cheking for nan rewards
            if (math.isnan( r_t )):
                r_t = 0.0
                for bad_r in range( 50 ):
                    my_config.logger.warn('Bad Reward Found')

            total_reward += r_t
            total_shaped_reward += shaped_r_t
            s_t = s_t1

            if (np.mod(step, 100) == 0):
                my_config.logger.debug("episode: %s, step: %s, action: %s, reward: %s, shaped: %s" % (episode, step_eps, a_t, r_t, shaped_r_t))

            step += 1
            step_eps += 1

        # my_config.logger.info("episode: %s, step_eps: %s, step: %s, reward: %s, replay buffer: %s" % (episode, step_eps, step, total_reward, agent.replay_buffer.count()))

        # Saving the best model.
        # if total_reward >= best_reward:
        if total_shaped_reward >= best_shaped_reward:
            if (train_indicator):
                my_config.logger.info("Now we save model with reward %s shaped %s, previous best reward %s shaped %s" % (total_reward, total_shaped_reward, best_reward, best_shaped_reward))
                best_reward = total_reward
                best_shaped_reward = total_shaped_reward
                agent.saveNetwork()

                # total_reward_test = 0
                # for i in xrange(my_config.test_eps):
                #     state_test = env.reset()
                #     done_test = False
                #     while not done_test:
                #         # env.render()
                #         action_test = agent.action(state_test)  # direct action for test
                #         state_test, reward_test, done_test, _ = env.step(action_test)
                #         total_reward_test += reward_test
                #         # my_config.logger.debug("test action: %s, reward: %s, total reward: %s" % (action_test, reward_test, total_reward_test))
                # ave_reward = total_reward_test / my_config.test_eps
                # my_config.logger.info("Episode: %s, Evaluation Average Reward: %s" % (episode, ave_reward))

    my_config.logger.warn("Finish...")

if __name__ == "__main__":
    playGame()