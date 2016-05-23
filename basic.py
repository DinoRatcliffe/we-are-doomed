#!/usr/bin/python2
from __future__ import print_function
from vizdoom import DoomGame
from vizdoom import Mode
from vizdoom import Button
from vizdoom import GameVariable
from vizdoom import ScreenFormat
from vizdoom import ScreenResolution
# Or just use from vizdoom import *

from random import choice
from time import sleep
from time import time

from networks.simple_qnetwork import SimpleQNetwork
import os
import numpy as np

game = DoomGame()
game.load_config("./scenarios/config/health_gathering.cfg")
actions = [[True,False,False],[False,True,False],[False,False,True]]
game.init()

sleep_time = 0
tick = 0

#ANN CODE
network = SimpleQNetwork('models/doom/test', 3, game.get_screen_width(), game.get_screen_height(), observe_ticks = 10000)

not_finished = True
observe = True
observe_sleep_time = 0.05
average = 0
episodes = 0

while True and not_finished:
	game.new_episode()

        action = 0
        episodes += 1

	while not (game.is_episode_finished()):
                tick += 1
                if not observe and tick % 1000 == 0: 
                    print("Tick:\t", tick)
		
                reward = game.make_action(actions[action])

                if (reward != -1):
                    print(reward)

                if not game.is_episode_finished():
                    state = game.get_state()

                if observe:
                    action = network.game_tick({'frame': state.image_buffer[0]})
                else:
                    action = network.game_train_tick(
                            { 'frame': state.image_buffer[0],
                              'terminal': game.is_episode_finished(),
                              'reward': reward })
                if observe:
                    sleep(observe_sleep_time)

        if observe:
            episode_reward = game.get_total_reward()
            average = average * (episodes-1)/episodes + episode_reward/episodes 

            os.system('cls' if os.name == 'nt' else 'clear')
            print("Last Episode Reward:\t", episode_reward)
            print("Number of Episodes:\t", episodes)
            print("Average Reward:\t\t", average)

game.close()
