from lib import *
import pandas as pd
import numpy as np
import random, os


# by Xiang Gao, 2018


class Market:
    """
    state 			MA of prices, normalized using values at t
                    ndarray of shape (window_state, n_instruments * n_MA), i.e., 2D
                    which is self.state_shape

    action 			three action
                    0:	empty, don't open/close.
                    1:	open a position
                    2: 	keep a position
    """

    def reset(self):
        self.empty = True
        self.max_reward = self.max_sample_RUL - self.window_state
        self.t = self.t0
        self.t_max = len(self.sample_data) - 1
        return self.get_state(), self.valid_actions

    def get_state(self, t=None):
        if t is None:
            t = self.t
        state = self.sample_data[self.sensor_name_choice][t - self.window_state + 1: t + 1].copy()
        norm = state.mean()
        state = (state/norm-1)*100  # not sure why this is being done, maybe normalisation?
        state = state.to_numpy()

        # for i in range(1):
        #     norm = np.mean(state[:, i])
        #     state[:, i] = (state[:, i] / norm - 1.) * 100
        return state

    def get_maintain_reward(self):
        t = self.t
        rul = self.max_sample_RUL - t
        if rul > self.optimum_buffer:
            reward = self.reward_increment
        elif self.optimum_buffer >= rul >= self.window_state:
            reward = self.reward_reduction
        else:
            reward = self.failure_cost
        return reward

    def get_replace_reward(self):
        t = self.t
        midway_point = int(self.max_sample_RUL/2)
        if midway_point > t:
            reward = self.failure_cost
        else:
            reward = self.reward_replace
        return reward

    def step(self, action):
        done = False
        if action == 0:  # maintain
            reward = self.get_maintain_reward()
        elif action == 1:  # replace
            reward = self.get_replace_reward()
        else:
            raise ValueError('no such action: ' + str(action))

        self.t += 1

        if self.t == self.t_max or action == 1:
            done = True
        return self.get_state(), reward, done, self.valid_actions

    def sample(self):
        unit_nr_list = list(self.df["unit_nr"].unique())
        unit_nr_choice = random.choice(unit_nr_list)
        sensor_name_choice = random.choice(self.sensor_names)

        return(unit_nr_choice , sensor_name_choice)

    def __init__(self,
                 csv_name,
                 window_state,
                 reward_increment,
                 reward_replace,
                 reward_reduction,
                 failure_cost,
                 optimum_buffer,
                 n_action=2,
                 n_var=1):

        self.df = pd.read_csv(csv_name)
        self.window_state = window_state
        self.reward_increment = reward_increment
        self.reward_replace = reward_replace
        self.reward_reduction = reward_reduction
        self.failure_cost = failure_cost
        self.optimum_buffer = optimum_buffer

        self.n_action = n_action
        self.n_var = n_var
        self.state_shape = (window_state, self.n_var)
        self.action_labels = ['maintain', 'replace']
        self.valid_actions = [0, 1]  # 0 for maintain, 1 for replace
        self.t0 = window_state - 1
        self.sensor_names = ['s_{}'.format(i) for i in range(1,22)]

        # self.unit_nr_choice, self.sensor_name_choice = self.sample()
        self.unit_nr_choice = 1
        self.sensor_name_choice = "s_4"
        self.sample_data = self.df[self.df["unit_nr"] == self.unit_nr_choice][["unit_nr", self.sensor_name_choice, "RUL"]]
        self.max_sample_RUL = self.sample_data["RUL"].max()


if __name__ == '__main__':
    print(os.getcwd())

    env = Market(csv_name="main_data.csv",
                 window_state=10,
                 reward_increment=10.,
                 reward_replace=100,
                 reward_reduction=-1000.,
                 failure_cost=-200000.,
                 optimum_buffer=30)

    env.reset()
    # print("state: {}".format(state),"\n","reward: {}".format(reward),"\n","t_max: {}".format(t_max))
    # print(len(env.sample_data))
    rewards = []
    for i in range(env.max_sample_RUL):
        action = random.choice([0, 1])
        state, reward, _, _ = env.step(action=action)
        print("reward: {}".format(reward),"\n")
        rewards.append(reward)
    print("total reward: {}".format(sum(rewards)))


