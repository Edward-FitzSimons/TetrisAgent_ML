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
from engine import TetrisEngine

# This is being modeled based on the
# user engine, where the user plays

############ Constants #############
SPEED = 1

# For greedy alg
E = .05

def play_game(spectate):
    
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
    
    value = 0
    while not done:
          
            # Get set of states to decide from
            # state: [0] = anchor x, [1] = shape (including rotation)
            states = env.get_states()
            
            # Select an end states at random
            #if rnd.randint(1, 100) < 100 * E:
            end = states[rnd.randint(0, len(states) - 1)]

            
            # Game Step
            # Step until we reach our desired states
            new = False # whether or not we've dropped a new block
            success = False # whether or not we could get to the end state
            st_anch = end[0] # starting anchor of the shape
            reward = 0 # reward of this set of actions
            done = False # whether or not we've lost
            acts = [] # the set of actions
            while not done and not new:
                
                if spectate: stdscr.getch()
                action = 2
                
                if end[0][0] != env.anchor[0]:
                    if end[0][0] < env.anchor[0]: action = 0
                    else: action = 1
                elif not np.array_equal(end[1], env.shape):
                    action = 5
                
                state, reward, done, new = env.step(action)
                acts.append(action)
                
                # Render
                stdscr.clear()
                stdscr.addstr(str(env))
                stdscr.addstr('\nReward: ' + str(reward) 
                            + '\nValue: ' + str(value)
                            + '\nCurrent Shape: ' + str(env.shape)
                            + '\nGoal Shape: ' + str(end[1])
                            + '\nGoal Anchor: ' + str(end[0]))
                value += reward
                
                if end[0][0] == env.anchor[0] and np.array_equal(end[1], env.shape):
                    success = True
                
            db.append((state, reward, done, st_anch, 
                       env.anchor, end[1], env.height,
                       success, acts))
            
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
    
# Function to get command line agruments
def getopts(argv):
    opts = {}  # Empty dictionary to store key-value pairs.
    while argv:  # While there are arguments left to parse...
        if argv[0][0] == '-':  # Found a "-name value" pair.
            if len(argv) > 1:
                opts[argv[0]] = argv[1]  # Add key and value to the dictionary.
            else:
                opts[argv[0]] = 0
                
        argv = argv[1:]  # Reduce the argument list by copying it starting from index 1.
            
    return opts

# Arguments for agent.py:
# -s: spectate, no variable just flag
# -r n: play n games and record to database
if __name__ == '__main__':
    # Curses standard screen
    stdscr = cs.initscr()
    
    # Grab arguments
    from sys import argv
    myargs = getopts(argv)
    
    spectate = False
    if '-s' in myargs: spectate = True
    runAuto = 0
    if '-r' in myargs: runAuto = int(myargs['-r'])

    # Init environment
    width, height = 10, 20 # standard tetris friends rules
    env = TetrisEngine(width, height)

    # Play games on repeat
    while True:
        init()
        stdscr.clear()
        env.clear()
        db = play_game(spectate)

        # Return to terminal
        terminate()
        
        # Should the game info be saved?
        save = runAuto > 0 or save_game()
        if save:
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
        play = runAuto > 0 or play_again()
        if not play:
            print('Thanks for contributing!')
            break
        
        runAuto = runAuto - 1