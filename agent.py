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
            end = states[0]
            
            # Select an end states at random
            if rnd.randint(1, 100) < 100 * E:
                end = states[rnd.randint(0, len(states) - 1)]
            else:
                end = pick_action(db, states)

            
            # Game Step
            # Step until we reach our desired states
            new = False # whether or not we've dropped a new block
            success = False # whether or not we could get to the end state
            reward = 0 # reward of this set of actions
            done = False # whether or not we've lost
            acts = [] # the set of actions
            prev_height = env.block_height
            b_height = prev_height
            
            while not done and not new:
                
                if spectate: stdscr.getch()
                action = 2
                
                if end[0][0] != env.anchor[0]:
                    if end[0][0] < env.anchor[0]: action = 0
                    else: action = 1
                elif not np.array_equal(end[1], env.shape):
                    action = 5
                else:
                    action = 2
                
                state, reward, done, new, b_height, cleared, cover = env.step(action)
                acts.append(action)
                
                # Render
                stdscr.clear()
                stdscr.addstr(str(env))
                stdscr.addstr('\nReward: ' + str(reward) 
                            + '\nValue: ' + str(value)
                            + '\nGoal Anchor: ' + str(end[0])
                            + '\nBlock Height: ' + str(env.block_height))
                value += reward
                
                if end[0][0] == env.anchor[0] and np.array_equal(end[1], env.shape):
                    success = True
       
            db = update_db(db, reward, end[1], env.board,
                           b_height - prev_height, cleared, cover, success)
            
    db['Value'].append(value)
    return db

# Pick an end state from all possible end states
def pick_action(db, states):
    
    # Get dummy board from environment
    dummy_brd = np.copy(env.board)
            
    # Get Value board to check against anchors
    reward_brd = db['Board']
    b_height = env.block_height
    
    # Cycle and compare
    end = states[0]
    end_rew = -10
    for state in states:        
        rew = 0
        fin_brd, height = apply_shape(state[0], state[1], dummy_brd)
        
        #Number of lines that can be cleared and reward
        clear = can_clear(fin_brd)
        for i in range(clear, 0, -1):
            tag_clear = 'R|Lines_Cleared_' + str(clear)
            if tag_clear in db:
                rew = rew + db[tag_clear][1]
                break
            
        # Did we reaise or lower the block height?
        if height - clear > b_height:
            rew = rew + db['R|Up'][1] * (height - clear - b_height)
        else:
            rew = rew + db['R|NotUp'][1] * (b_height - clear - height)
            
        # Any blocks open below - If so, add to expected reward
        op = open_below(state[0], state[1], fin_brd, 20)
        rew = rew + op * db['R|Cover'][1]
        
        # Add all average rewards and return    
        rew = rew + get_reward_avg(state[0], state[1], reward_brd)
        
        # If end reward is greater than previous reward, set equal to the state and current reward    
        if rew > end_rew:
            end, end_rew = state, rew
    
    return end
    
# gets the average reward from group of blocks
def get_reward_avg(anchor, shape, board):
    x , y = anchor[0], anchor[1]
    rw = 0
    
    for i,j in shape:
        if x + i >= 0 and x + i < 10 and y + j >= 0 and y + j < 20:
            rw = rw + board[x + i][y + j][1]

    return rw/4

# make sure we use a copy for this
def can_clear(board):
    clear = 0
    for row in board:
        if 0 not in row:
            clear = clear + 1
            
    return clear

def open_below(anchor, shape, board, height):

    open = 0
    for i, j in shape:
        x, y = anchor[0] + i, anchor[1] + j
        z = 1
        while y + z < height and y >= 0 and board[x, y+z] == 0:
            open = open + 1
            z = z + 1
                
    return open

def apply_shape(anchor, shape, board):
    brd = np.copy(board)
    x, y, top = anchor[0], anchor[1], anchor[1]
    for i,j in shape:
        if x + i >= 0 and x + i < 10 and y + j >= 0 and y + j < 20:
            brd[x + i][y + j] = 1
            if top > y + j: top = y + j
    
    return brd, 20 - top

def update_db(db, reward, shape, board, direc, l_clear, cover, success):
    # Slot 0 = Number of entries, Slot 1 = the actual entries
    
    # Update general reward
    db['Reward'][0], db['Reward'][1] = online_mean(db, 'Reward', reward)

    # Update block/reward rations
    if reward > 0:
        db['Board'] = board_means(db, board, reward)
    
    # Update reward based on whether or not we increase or decrease the block level
    if direc > 0:
        db['R|Up'][0], db['R|Up'][1] = online_mean(db, 'R|Up', reward)
    elif direc < 0:
        db['R|NotUp'][0], db['R|NotUp'][1] = online_mean(db, 'R|NotUp', reward)
        
    # Update reward based on whether or not move was completed
    if success:
        db['R|Success'][0], db['R|Success'][1] = online_mean(db, 'R|Success', reward)
        
    # Update reward based on whether or not we've covered any open spaces
    if cover:
        db['R|Cover'][0], db['R|Cover'][1] = online_mean(db, 'R|Cover', reward)
    
    # Update reward based on lines cleared    
    sh_name, rot = find_shape_name(shape)
    if sh_name is not None:
        tag = 'R|' + sh_name + '_' + str(rot)
        db[tag][0], db[tag][1] = online_mean(db, tag, reward)

    # Update reward based on lines cleared    
    if l_clear > 0:
        tag = 'R|Lines_Cleared_' + str(l_clear)
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
                return n, r
            sh = [(-j, i) for i, j in sh]
            
    return None, 0

def online_mean(db, tag, reward):
    if tag not in db:
        db[tag] = [0,0]
    
    n, r = db[tag][0] + 1, db[tag][1]
    r = r + (1/n) * (reward - r)
    return n, r

def board_means(db, brd_env, r):
    brd = db['Board']
    for i in range(10):
        for j in range(20):
            if brd_env[i][j] != 0:
                n = brd[i][j][0] + 1
                brd[i][j][1] = brd[i][j][1] + (1/n) * (r - brd[i][j][1])
                brd[i][j][0] = n
    
    return brd

def grab_db():
    # Initialize the database
    
    try:
        fr = open('training_data.npy', 'rb')
        return np.load(fr).item()
    except Exception as e:
        brd = np.zeros((10,20,2))
        # 'Key': [n, avg]
        return {'Board': brd, 'Reward':[0,0], 'Value':[],
                'R|Up':[0,0], 'R|NotUp':[0,0], 'R|Success':[0,0], 'R|Cover':[0,0],
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