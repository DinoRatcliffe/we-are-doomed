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
import numpy as np


game = DoomGame()

game.set_vizdoom_path("./bin/vizdoom")
game.set_doom_game_path("./bin/freedoom2.wad")
game.set_doom_scenario_path("./bin/basic.wad")

game.set_doom_map("map01")
game.set_screen_resolution(ScreenResolution.RES_320X240)

# Sets the screen buffer format. Not used here but now you can change it. Defalut is CRCGCB.
game.set_screen_format(ScreenFormat.GRAY8)

game.set_render_hud(False)
game.set_render_crosshair(False)
game.set_render_weapon(True)
game.set_render_decals(False)
game.set_render_particles(False)

game.add_available_button(Button.TURN_LEFT) 
game.add_available_button(Button.TURN_RIGHT)
game.add_available_button(Button.ATTACK)

actions = [[True,False,False],[False,True,False],[False,False,True]]

game.add_available_game_variable(GameVariable.HEALTH)
game.add_available_game_variable(GameVariable.AMMO2)

game.set_episode_start_time(10)
game.set_window_visible(True)
game.set_living_reward(-1)
game.set_mode(Mode.PLAYER)
game.init()

episodes = 10
sleep_time = 0
tick = 0

#ANN CODE
network = SimpleQNetwork('models/doom/test', 3, 240, 320)

for i in range(episodes):
	print("Episode #" + str(i+1))

	# Starts a new episode. It is not needed right after init() but it doesn't cost much. At least the loop is nicer.
	game.new_episode()

        action = 0

	while not (game.is_episode_finished() or tick == 105):

                tick += 1
                print("tick: " + str(tick))
		
		state = game.get_state()
                reward = game.make_action(actions[action])

                action = network.game_train_tick(
                        { 'frame': state.image_buffer[0],
                          'terminal': game.is_episode_finished(),
                          'reward': reward })

	# Check how the episode went.
	print("Episode finished.")
	print("total reward:", game.get_total_reward())
	print("************************")

game.close()
