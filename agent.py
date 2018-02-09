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

def play_game():
    
    #Grab environment
    
    # store play information
    db = []
    
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
    
    while not done:
            action= 6
            action = rnd.randint(1,5)
            
            # Game Step
            state, reward, done = env.step(action)
            db.append((state, reward, done, action))
            
            # Render
            stdscr.clear()
            stdscr.addstr(str(env))
            stdscr.addstr('reward: ' + str(reward))
            
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
    cs.stdscr.keypad(False)
    cs.echo()
    cs.endwin()
    
def init():
    cs.noecho()
    cs.halfdelay(7)
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