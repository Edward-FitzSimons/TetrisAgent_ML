# -*- coding: utf-8 -*-
# Agent for playing Tetris
# Using TetrisRL environment by Jay Butera
# Authors:  Edward FitzSimons
#           Matt Alacheff
#           Ryan Novotny
#           Ben Harvey

import curses as cs
import numpy as np
import random as rnd
import os
from engine import TetrisEngine

# This is being modeled based on the
# user engine, where the user plays

#### Constants #####
SPECTATE = False
SPEED = 1

def play_game():
    
    #Grab environment
    
    # store play information
    db = []
    # grab board from engine
    brd = env.board
    
    # initial rendering
    stdscr.addstr(str(env))
    
    done = False
    # Global Action
    action = 6
    
    # NOTE: ACTION NUMBER : ACTION
    #       0 : Shift Left
    #       1 : Shift Right
    #       2 : Hard drop
    #       3 : Soft drop
    #       4 : Rotate Left
    #       5 : Rotate Right
    #       6 : No Action
    
    value = 0
    while not done:
          
            # Get set of states to decide from
            # state: [0] = anchor x, [1] = shape (including rotation)
            states = env.get_states()
            # Select an end states at random
            end = states[rnd.randint(0,len(states) - 1)]
            
            # Game Step
            # Step until we reach our desired states
            new = False
            reward = 0
            done = False
            acts = []
            while not done and not new:
                
                if SPECTATE: stdscr.getch()
                action = 6
                
                if not np.array_equal(end[1], env.shape):
                    action = 5
                elif end[0] != env.anchor[0]:
                    if end[0] < env.anchor[0]: action = 0
                    else: action = 1
                
                state, reward, done, new = env.step(action)
                acts.append(action)
                
                # Render
                stdscr.clear()
                stdscr.addstr(str(env))
                stdscr.addstr('\nReward: ' + str(reward) 
                            + '\nValue: ' + str(value)
                            + '\nCurrent Shape: ' + str(env.shape)
                            + '\nGoal Shape: ' + str(end[1]))
                value += reward
                
            db.append((state, reward, done, acts))
            
    return db

def play_again():
   
    # Shift back to command line for this
    print('Play Again? [Y/n]')
    print('>', end='')
    choice = input()
    
    return True if choice.lower() == 'y' else False
    
def save_game():
    print('Accumulated reward: {0} | {1} moves'.format(sum([i[1] for i in db]), len(db)))
    print('Would you like to store the game info as training data? [y/n]')
    print('> ', end='')
    choice = input()
    
    return True if choice.lower() == 'y' else False

def terminate():
    cs.nocbreak()
    stdscr.keypad(False)
    cs.echo()
    cs.endwin()
    
def init():
    cs.noecho()
    cs.halfdelay(SPEED)
    stdscr.keypad(True)

if __name__ == '__main__':
    # Curses standard screen
    stdscr = cs.initscr()

    # Init environment
    width, height = 10, 20 # standard tetris friends rules
    env = TetrisEngine(width, height)

    # Play games on repeat
    while True:
        init()
        stdscr.clear()
        env.clear()
        db = play_game()

        # Return to terminal
        terminate()
        # Should the game info be saved?
        if save_game():
            try:
                fr = open('training_data.npy', 'rb')
                x = np.load(fr)
                fr.close()
                fw = open('training_data.npy', 'wb')
                x = np.concatenate((x,db))
                #print('Saving {0} moves...'.format(len(db)))
                np.save(fw, x)
                print('{0} data points in the training set'.format(len(x)))
            except Exception as e:
                print('no training file exists. Creating one now...')
                fw = open('training_data.npy', 'wb')
                print('Saving {0} moves...'.format(len(db)))
                np.save(fw, db)
        # Prompt to play again
        if not play_again():
            print('Thanks for contributing!')
            break