# Creating an Agent to Intelligently Play Tetris
The purpose of this project is to create an agent that can play the game Tetris without the need for human interaction or intervention. This projects environment is based largely off the contributions of Jay Butera and his "Tetris RL" project. Installation instructions for the environment can be found on his Git Repo page [here.](https://github.com/jaybutera/tetrisRL)

## Methods
We intend to utilize Butera's implemented methods for Reinforcement Learning as a basis for our own modified implementation. Additionally, a modified Q-Learning algorithm will be implemented later on to help increase the efficiency of the learning and the diminishment of failed states.

## Layout
* dqn_agent.py - DQN reinforcement learning agent trains on tetris
* supervised_agent.py - The same convolutional model as DQN trains on a dataset of user playthroughs
* user_engine.py - Play tetris and accumulate information as a training set
* run_model.py - Evaluate a saved agent model on a visual game of tetris (i.e.)
```bash
$ python run_model.py checkpoint.pth.tar
```

## Play Tetris for Training Data
Play games and accumulate a data set for a supervised learning algorithm to
trian on. An element of data stores a
(state, reward, done, action) tuple for each frame of the game.

You may notice the rules are slightly different than normal Tetris.
Specifically, each action you take will result in a corresponding soft drop
This is how the AI will play and therefore how the training data must be taken.

To play Tetris:
```bash
$ python user_engine.py
```

Controls:  
W: Hard drop (piece falls to the bottom)  
A: Shift left  
S: Soft drop (piece falls one tile)  
D: Shift right  
Q: Rotate left  
E: Rotate right  

At the end of each game, choose whether you want to store the information of
that game in the data set.
