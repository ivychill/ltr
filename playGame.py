from osim.env import *
from ddpg import *
import math

def playGame():
    train_indicator = my_config.is_training
    env = RunEnv(visualize = False)
    agent = DDPG(env)

    episode_count = my_config.max_eps
    step = 0
    best_reward = 0

    my_config.logger.warn("Opensim Experiment Start.")
    for episode in range(episode_count):
        s_t = env.reset()
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
            # my_config.logger.debug("s_t1: %s" % (s_t1))

            if (train_indicator):
                agent.perceive(s_t,a_t,r_t,s_t1,done)
                
            #Cheking for nan rewards
            if ( math.isnan( r_t )):
                r_t = 0.0
                for bad_r in range( 50 ):
                    my_config.logger.warn('Bad Reward Found')

            total_reward += r_t
            s_t = s_t1

            if (np.mod(step, 1000) == 0):
                my_config.logger.debug("episode: %s, step: %s, action: %s, reward: %s" % (episode, step_eps, a_t, r_t))

            step += 1
            step_eps += 1

        my_config.logger.debug("episode: %s, step_eps: %s, step: %s, reward: %s, replay buffer: %s" % (episode, step_eps, step, total_reward, agent.replay_buffer.count()))

        # Testing:
        if (train_indicator):
            if episode % 100 == 0 and episode > 100:
                total_reward_test = 0
                for i in xrange(my_config.test_eps):
                    state = env.reset()
                    while not done:
                        # env.render()
                        action = agent.action(state)  # direct action for test
                        state, reward, done, _ = env.step(action)
                        total_reward_test += reward
                        if done:
                            break
                ave_reward = total_reward_test / my_config.test_eps
                my_config.logger.debug("Episode: %s, Evaluation Average Reward: %s" % (episode, ave_reward))

        #Saving the best model.
        if total_reward >= best_reward :
            if (train_indicator):
                my_config.logger.debug("Now we save model with reward %s, previous best reward was %s" % (str(total_reward), str(best_reward)))
                best_reward = total_reward
                agent.saveNetwork()

    my_config.logger.warn("Finish...")

if __name__ == "__main__":
    my_config.init()
    playGame()