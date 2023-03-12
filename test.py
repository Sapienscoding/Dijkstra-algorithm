import numpy as np
import time
from queue import PriorityQueue
import cv2
import math
import imageio as ig
import argparse


class nodes:
    def __init__(self, explored=False, cost2come=math.inf, parent_idx=None):
        self.explored = explored
        self.parent_idx = parent_idx
        self.cost2come = cost2come

# creating obstacle shapes

# Adding clearance to map walls


def map_clearance(map_canvas):
    cv2.rectangle(map_canvas, (0, 0), (600, 250), (255, 255, 255), 5)

# rectangle obstacle with clearance


def rectangle1(map_canvas):
    cv2.rectangle(map_canvas, (100, 0), (150, 100), (150, 100, 100), -1)
    cv2.rectangle(map_canvas, (100, 0), (150, 100), (255, 255, 255), 5)


def rectangle2(map_canvas):
    cv2.rectangle(map_canvas, (100, 150), (150, 250), (150, 100, 100), -1)
    cv2.rectangle(map_canvas, (100, 150), (150, 250), (255, 255, 255), 5)

# hexagon obstacle with clearance


def hexagon(map_canvas):
    pts = np.array([[300, 50], [225, 82.5], [225, 157.5],
                    [300, 200], [375, 157.5], [375, 82.5]])
    cv2.fillPoly(map_canvas, np.int32([pts]), (150, 100, 100))
    cv2.polylines(map_canvas, np.int32([pts]), True, (255, 255, 255), 5)

# triangle obstacle with clearance


def triangle(map_canvas):
    pts = np.array([[460, 25], [460, 225], [510, 125]])
    cv2.fillPoly(map_canvas, [pts], (150, 100, 100))
    cv2.polylines(map_canvas, [pts], True, (255, 255, 255), 5)

# function to create map with obstacles in it


def Map():
    image = np.zeros((250, 600, 3))
    map_clearance(image)
    rectangle1(image)
    rectangle2(image)
    hexagon(image)
    triangle(image)
    cv2.imwrite('Map.jpg', image)
    return image
# Map()

# taking start and goal positions from the user


def getInput():
    inp = argparse.ArgumentParser()
    inp.add_argument('-src', '--starting_position',
                     required=True, nargs='+', type=int)
    inp.add_argument('-goal', '--goal_position',
                     required=True, nargs='+', type=int)
    positions = vars(inp.parse_args())
    return positions
# to retrieve starting pos of bot


def start_pos(c_space, pos):
    x, y = c_space.shape[:2]
    s_x = x-pos[1]-1
    s_y = pos[0]
    if not Isvalid(c_space, s_x, s_y):
        print('Invalid input!, starting position OUT-OF-RANGE!!!')
        return None
    return s_x, s_y

# to retreive goal pos of bot


def goal_pos(c_space, pos):
    x, y = c_space.shape[:2]
    # print(x)
    g_x = x-pos[1]-1
    g_y = pos[0]
    if not Isvalid(c_space, g_x, g_y):
        print('Invalid input!, goal position OUT-OF-RANGE!!!')
        return None
    return g_x, g_y

# to check if the current node is in the correct space


def Isvalid(c_space, x, y):
    x_c, y_c = c_space.shape[:2]
    # current node pos within map and not in obstacle space
    if (0 <= x < x_c) and (0 <= y < y_c) and (c_space[x][y] == (0, 0, 0)).all():
        return True
    else:
        return False

# Defining action set to get new nodes from current nodes


def action_set(t_node, c_space, c_node, openlist):
    x_cur, y_cur = c_node

    # Move up
    if Isvalid(c_space, x_cur-1, y_cur):
        # comparing the cost of new child node and current node
        if (t_node[x_cur][y_cur].cost2come + 1) < t_node[x_cur-1][y_cur].cost2come:
            t_node[x_cur-1][y_cur].cost2come = t_node[x_cur][y_cur].cost2come + 1
            t_node[x_cur-1][y_cur].parent_idx = c_node
            openlist.put((t_node[x_cur-1][y_cur].cost2come, (x_cur-1, y_cur)))

    # Move down
    if Isvalid(c_space, x_cur+1, y_cur):
        if (t_node[x_cur][y_cur].cost2come + 1) < t_node[x_cur+1][y_cur].cost2come:
            t_node[x_cur+1][y_cur].cost2come = t_node[x_cur][y_cur].cost2come + 1
            t_node[x_cur+1][y_cur].parent_idx = c_node
            openlist.put((t_node[x_cur+1][y_cur].cost2come, (x_cur+1, y_cur)))

    # Move left
    if Isvalid(c_space, x_cur, y_cur-1):
        if (t_node[x_cur][y_cur].cost2come + 1) < t_node[x_cur][y_cur-1].cost2come:
            t_node[x_cur][y_cur-1].cost2come = t_node[x_cur][y_cur].cost2come + 1
            t_node[x_cur][y_cur-1].parent_idx = c_node
            openlist.put((t_node[x_cur][y_cur-1].cost2come, (x_cur, y_cur-1)))

    # Move right
    if Isvalid(c_space, x_cur, y_cur+1):
        if (t_node[x_cur][y_cur].cost2come + 1) < t_node[x_cur][y_cur+1].cost2come:
            t_node[x_cur][y_cur+1].cost2come = t_node[x_cur][y_cur].cost2come + 1
            t_node[x_cur][y_cur+1].parent_idx = c_node
            openlist.put((t_node[x_cur][y_cur+1].cost2come, (x_cur, y_cur+1)))

    # Move up-left
    if Isvalid(c_space, x_cur-1, y_cur-1):
        if (t_node[x_cur][y_cur].cost2come + 1.4) < t_node[x_cur-1][y_cur-1].cost2come:
            t_node[x_cur-1][y_cur-1].cost2come = t_node[x_cur][y_cur].cost2come + 1.4
            t_node[x_cur-1][y_cur-1].parent_idx = c_node
            openlist.put(
                (t_node[x_cur-1][y_cur-1].cost2come, (x_cur-1, y_cur-1)))

    # Move up-right
    if Isvalid(c_space, x_cur-1, y_cur+1):
        if (t_node[x_cur][y_cur].cost2come + 1.4) < t_node[x_cur-1][y_cur+1].cost2come:
            t_node[x_cur-1][y_cur+1].cost2come = t_node[x_cur][y_cur].cost2come + 1.4
            t_node[x_cur-1][y_cur+1].parent_idx = c_node
            openlist.put(
                (t_node[x_cur-1][y_cur+1].cost2come, (x_cur-1, y_cur+1)))

    # Move down-left
    if Isvalid(c_space, x_cur+1, y_cur-1):
        if (t_node[x_cur][y_cur].cost2come + 1.4) < t_node[x_cur+1][y_cur-1].cost2come:
            t_node[x_cur+1][y_cur-1].cost2come = t_node[x_cur][y_cur].cost2come + 1.4
            t_node[x_cur+1][y_cur-1].parent_idx = c_node
            openlist.put(
                (t_node[x_cur+1][y_cur-1].cost2come, (x_cur+1, y_cur-1)))

    # Move down-right
    if Isvalid(c_space, x_cur+1, y_cur+1):
        if (t_node[x_cur][y_cur].cost2come + 1.4) < t_node[x_cur+1][y_cur+1].cost2come:
            t_node[x_cur+1][y_cur+1].cost2come = t_node[x_cur][y_cur].cost2come + 1.4
            t_node[x_cur+1][y_cur+1].parent_idx = c_node
            openlist.put(
                (t_node[x_cur+1][y_cur+1].cost2come, (x_cur+1, y_cur+1)))

    return t_node

# To compute minimum cost2come of popped node from the open list


def lowestC2C(node):
    # giving a random position to node
    c_node = (-1, -1)
    c2c_min = math.inf
    # updating the cost
    for i in range(len(node)):
        for j in range(len(node[i])):
            if node[i][j].cost2come < c2c_min and not node[i][j].explored:
                c_node = (i, j)
                c2c_min = node[i][j].cost2come
    return c_node

# backtracking from goal to start node, to get the shortest path


def backtrack(node, g_node, image):
    # taking current node as goal node
    c_node = g_node
    # to store the next nodes
    queue = []
    # storing the path moves
    s2g_images = []
    while c_node is not None:
        queue.append(c_node)
        c_node = node[c_node[0]][c_node[1]].parent_idx
    queue = queue[::-1]
    for n in queue:
        image[n[0]][n[1]] = (150, 100, 100)
        s2g_images.append(np.uint8(image.copy()))
        image[n[0]][n[1]] = (0, 0, 0)

    """Saving path video, error with fourcc tag"""

    # fourcc = cv2.VideoWriter_fourcc(*'mp4s')
    # v = cv2.VideoWriter('path.mp4', fourcc, 55, (250, 600), True)
    # for i in range(len(s2g_images)):
    #     v.write(s2g_images[i])
    # cv2.destroyAllWindows()
    # v.release()
    ig.mimsave('path.gif', s2g_images, fps=50)
    cv2.imwrite('backtracked.png', s2g_images[-1])

# dijkstra algorithm to find the path


def Dijkstra(pos):
    c_space = Map()
    closedlist_node = []
    openlist_node = PriorityQueue()
    r, c = c_space.shape[:2]

    src = start_pos(c_space, pos['starting_position'])
    dst = goal_pos(c_space, pos['goal_position'])

    # if the starting and goal positions are empty
    if (src is None and dst is None):
        exit(1)

    node_info = [[nodes() for n in range(c)] for m in range(r)]
    node_info = np.array(node_info)
    node_info[src[0]][src[1]].explored = True
    node_info[src[0]][src[1]].cost2come = 0
    # putting the starting node in the open list
    openlist_node.put((node_info[src[0]][src[1]].cost2come, src))
    # time when the algorithm begins
    time_initial = time.time()
    # Creating local copy of the map
    img = c_space.copy()
    # Marking the goal point
    img[dst[0]][dst[1]] = (0, 255, 0)
    while openlist_node:
        cur_node = openlist_node.get()[1]
        # comparing current node to goal
        if cur_node == dst:
            # time when goal state is reached
            time_final = time.time()
            print('time to goal {} sec'.format(time_final-time_initial))
            print('Backtracking!, visualisation from goal to start')
            backtrack(node_info, dst, img)
            break
        # nodes info getting updated, by finding child nodes in all directions
        node_info[cur_node[0]][cur_node[1]].explored = True
        node_info = action_set(node_info, c_space, cur_node, openlist_node)
        img[cur_node[0]][cur_node[1]] = (255, 0, 0)
        closedlist_node.append(np.uint8(img.copy()))

    """Saving explored nodes video, error with fourcc tag"""
    # fourcc = cv2.VideoWriter_fourcc(*'mp4s')
    # v = cv2.VideoWriter('explored.mp4', fourcc, 55, (250, 600), True)
    # for i in range(len(closedlist_node)):
    #     v.write(closedlist_node[i])
    # cv2.destroyAllWindows()
    # v.release()
    ig.mimsave('closedlist.gif', closedlist_node, fps=50)
    cv2.imwrite('explorednodes.png', closedlist_node[-1])


if __name__ == '__main__':
    coordinates = getInput()
    Dijkstra(coordinates)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
