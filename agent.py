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
    db = grab_db()

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
                
                state, reward, done, new, b_height = env.step(action)
                acts.append(action)
                
                # Render
                stdscr.clear()
                stdscr.addstr(str(env))
                stdscr.addstr('\nReward: ' + str(reward) 
                            + '\nValue: ' + str(value)
                            + '\nGoal Anchor: ' + str(end[0])
                            + '\nTop: ' + str(b_height))
                value += reward
                
                if end[0][0] == env.anchor[0] and np.array_equal(end[1], env.shape):
                    success = True
       
            db = update_db(db, reward, end[1], env.board, 0, success)
            
    return db

def update_db(db, reward, shape, board, direc, success):
    
    # Update general reward
    db['Reward'][0], db['Reward'][1] = online_mean(db, 'Reward', reward)
    # Update block/reward rations
    db['Board'][0], db['Board'][1] = board_means(db, board, reward)
    
    # Update reward based on whether or not we increase or decrease the block level
    if direc > 0:
        db['R|Up'][0], db['R|Up'][1] = online_mean(db, 'R|Up', reward)
    elif direc < 0:
        db['R|Down'][0], db['R|Down'][1] = online_mean(db, 'R|Down', reward)
        
    # Update reward based on whether or not move was completed
    if success:
        db['R|Success'][0], db['R|Success'][1] = online_mean(db, 'R|Success', reward)
        
    sh_name = find_shape_name(shape)
    if sh_name is not None:
        tag = 'R|' + sh_name
        db[tag][0], db[tag][1] = online_mean(db, tag, reward)
    
    return db

# Finds the name, including number of clockwise rotations, of a shape
def find_shape_name(shape):
    shapes = {
    'T': [(0, 0), (-1, 0), (1, 0), (0, -1)],
    'J': [(0, 0), (-1, 0), (0, -1), (0, -2)],
    'L': [(0, 0), (1, 0), (0, -1), (0, -2)],
    'Z': [(0, 0), (-1, 0), (0, -1), (1, -1)],
    'S': [(0, 0), (-1, -1), (0, -1), (1, 0)],
    'I': [(0, 0), (0, -1), (0, -2), (0, -3)],
    'O': [(0, 0), (0, -1), (-1, 0), (-1, -1)],
    }
    shape_names = ['T', 'J', 'L', 'Z', 'S', 'I', 'O']
    for n in shape_names:
        sh = shapes[n]
        for r in range(1,4):
            if np.array_equal(shape, sh):
                return n + '_' + str(r)
            sh = [(-j, i) for i, j in sh]
            
    return None

def online_mean(db, tag, reward):
    n, r = db[tag][0] + 1, db[tag][1]
    r = r + (1/n) * (reward - r)
    return n, r

def board_means(db, brd_env, r):
    n = db['Board'][0] + 1
    brd = db['Board'][1]
    for i in range(len(brd)):
        for j in range(len(brd[i])):
            if brd_env[i][j] != 0:
                brd[i][j] = brd[i][j] + (1/n) * (r - brd[i][j])
    
    return n, brd

def grab_db():

    try:
        fr = open('training_data.npy', 'rb')
        return np.load(fr).item()
    except Exception as e:
        brd = np.zeros((10,20))
        # 'Key': [n, avg]
        return {'Board':[0,brd], 'Reward':[0,0], 'R|Up':(0,0), 'R|Down':[0,0], 'R|Success':[0,0],
                'R|T_1':[0,0],'R|T_2':[0,0],'R|T_3':[0,0],'R|T_4':[0,0],
                'R|J_1':[0,0],'R|J_2':[0,0],'R|J_3':[0,0],'R|J_4':[0,0],
                'R|L_1':[0,0],'R|L_2':[0,0],'R|L_3':[0,0],'R|L_4':[0,0],
                'R|S_1':[0,0],'R|S_2':[0,0],'R|S_3':[0,0],'R|S_4':[0,0],
                'R|Z_1':[0,0],'R|Z_2':[0,0],'R|Z_3':[0,0],'R|Z_4':[0,0],
                'R|I_1':[0,0],'R|I_2':[0,0],'R|I_3':[0,0],'R|I_4':[0,0],
                'R|O_1':[0,0],'R|O_2':[0,0],'R|O_3':[0,0],'R|O_4':[0,0],}

def play_again():
   
    # Shift back to command line for this
    print('Play Again? [Y/n]')
    print('>', end='')
    choice = input()
    
    return True if choice.lower() == 'y' else False
    
def save_game():
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
                fw = open('training_data.npy', 'wb')
                #print('Saving {0} moves...'.format(len(db)))
                np.save(fw, db)
                print('Training set updated.')
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