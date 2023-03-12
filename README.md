# Project 2: Dijkstra algorithm for a point robot

Implementing Dijkstra planning algorithm for a point robot, to find optimal path between the starting position and goal posiiton.

## Installation

Libraries used in this project, can be installed via package manager PIP on your system. Following libraries were used:

* NumPy
* OpenCV
* Math
* Time
* Argparse
* Imageio
* Queue


## Usage
* Open the code on your IDE and write the following command on the terminal:

* python test.py -src "starting position" -goal "goal position"

* The algorithm will start and when it will be done, output would be saved as images and gif files on yours system.

## Test Case
1. start position: (5,5); goal position: (55,65); time: 9.66 sec
2. start position: (5,5); goal position: (175,128); time: 49.28 sec

## Output files
* backtracked.png
* Map.png
* explorednodes.png
* closedlist.gif
* path.gif