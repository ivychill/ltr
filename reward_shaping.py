# -*- coding: utf-8 -*-

import my_config

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
def shape(state0, state, reward):
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
    # my_config.logger.debug(
    #     "reward: %s, v_mc: %s, x_zmp: %s, x_talus_min: %s, x_toe_max: %s" % (reward, v_mc, x_zmp, x_talus_min, x_toe_max))
    if x_zmp <= x_talus_min:
        shaped_reward -= (x_talus_min - x_zmp ) * 100
    elif x_zmp >= x_toe_max:
        shaped_reward -= (x_zmp - x_toe_max) * 100

    return shaped_reward