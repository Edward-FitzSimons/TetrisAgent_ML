# Creating an Agent to Intelligently Play Tetris
The purpose of this project is to create an agent that can play the game Tetris without the need for human interaction or intervention. This projects environment is based largely off the contributions of Jay Butera and his "Tetris RL" project. Installation instructions for the environment can be found on his Git Repo page [here.](https://github.com/jaybutera/tetrisRL)

## Methods
We intend to utilize Butera's implemented methods for Reinforcement Learning as a basis for our own modified implementation. Additionally, a modified Q-Learning algorithm will be implemented later on to help increase the efficiency of the learning and the diminishment of failed states.

## Layout
* agent.py - Agent plays Tetris by itself.
* data_view.ipynb - Jupyter Notebook for the calculation and visualizations of rewards.
* engine.py - The underlying engine for the game.
* training_data.npy - Data stored from each game played as a Numpy Array.
* user_engine.py - Manually play tetris and accumulate information as a training set.

## Play Tetris
Play games and accumulate a data set for a supervised learning algorithm to train on. The database stores a dictionary of tuples containing the mean average of the rewards and other values.

Each action you take will result in a corresponding soft drop. This is how the AI will play and therefore how the training data must be taken.

### Play Tetris - Agent:
```
$ python3 agent.py
```

__Flags:__
* -s: Visualize the game currently being played
* -r: Play a select number of games

For example,
```
$ python3 agent.py -s -r 5
```
will visualize the agent playing 5 separate games of Tetris.

At the end of each game, choose whether you want to store the information of that game in the data set.

### Play Tetris - Manually (CURRENTLY BROKEN):
```bash
$ python3 user_engine.py
```

__Controls:__
* W: Hard drop (piece falls to the bottom)
* A: Shift left
* S: Soft drop (piece falls one tile)
* D: Shift right
* Q: Rotate left
* E: Rotate right