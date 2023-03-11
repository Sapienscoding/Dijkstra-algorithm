import numpy as np
import time
import heapq
import cv2
import math


class node:
    def __init__(self, explored=False, cost2come=math.inf, parent_idx=None):
        self.explored = explored
        self.cost = cost2come
        self.parent_idx = parent_idx

# creating obstacle shapes


def rectangle1(map_canvas):
    cv2.rectangle(map_canvas, (100, 0), (150, 100), (150, 100, 100), -1)
    cv2.rectangle(map_canvas, (100, 0), (150, 100), (255, 255, 255), 3)


def rectangle2(map_canvas):
    cv2.rectangle(map_canvas, (100, 150), (150, 250), (150, 100, 100), -1)
    cv2.rectangle(map_canvas, (100, 0), (150, 100), (255, 255, 255), 3)


def hexagon(map_canvas):
    pts = np.array([[300, 50], [225, 82.5], [225, 157.5],
                    [300, 200], [375, 157.5], [375, 82.5]])
    cv2.fillPoly(map_canvas, np.int32([pts]), (150, 100, 100))
    cv2.polylines(map_canvas, np.int32([pts]), True, (255, 255, 255), 3)


def triangle(map_canvas):
    pts = np.array([[460, 25], [460, 225], [510, 125]])
    cv2.fillPoly(map_canvas, pts, (150, 100, 100))
    cv2.polylines(map_canvas, pts, True, (255, 255, 255), 3)

# function to create map with obstacles in it


def Map():
    image = np.zeros((250, 600, 3))
    rectangle1(image)
    rectangle2(image)
    hexagon(image)
    triangle(image)
    cv2.imwrite('Map.jpg', image)
    cv2.imshow('Dijkstra Map', image)
    return image


Map()

# to retrieve starting pos of bot


def start_pos(c_space, pos):
    x, y = c_space.shape
    s_x = x-pos[1]-1
    s_y = pos[0]
    if not Isvalid(c_space, s_x, s_y):
        print('Invalid input!, starting position OUT-OF-RANGE!!!')
        return None
    return s_x, s_y

# to retreive goal pos of bot


def goal_pos(c_space, pos):
    x, y = c_space.shape
    g_x = x-pos[1]-1
    g_y = pos[0]
    if not Isvalid(c_space, g_x, g_y):
        print('Invalid input!, goal position OUT-OF-RANGE!!!')
        return None
    return g_x, g_y

# to check if the current node is in the correct space


def Isvalid(c_space, x, y):
    x_c, y_c = c_space.shape
    if (0 <= x < x_c) and (0 <= y < y_c) and (c_space[x][y] == (0, 0, 0)).all():
        return True
    else:
        return False

# Defining action set to get new nodes


def action_set(t_node, c_space, c_node, openlist):
    x_cur, y_cur = c_node

    # Move up
    if Isvalid(c_space, x_cur-1, y_cur):
        if (t_node[x_cur][y_cur].cost2come + 1) < t_node[x_cur-1][y_cur].cost2come:
            t_node[x_cur-1][y_cur].cost2come = t_node[x_cur][y_cur].cost2come + 1
            t_node[x_cur-1][y_cur].parent_idx = c_node
            openlist.put(((x_cur-1, y_cur), t_node[x_cur-1][y_cur].cost2come))

    # Move down
    if Isvalid(c_space, x_cur+1, y_cur):
        if (t_node[x_cur][y_cur].cost2come + 1) < t_node[x_cur+1][y_cur].cost2come:
            t_node[x_cur+1][y_cur].cost2come = t_node[x_cur][y_cur].cost2come + 1
            t_node[x_cur+1][y_cur].parent_idx = c_node
            openlist.put(((x_cur+1, y_cur), t_node[x_cur+1][y_cur].cost2come))

    # Move left
    if Isvalid(c_space, x_cur, y_cur-1):
        if (t_node[x_cur][y_cur].cost2come + 1) < t_node[x_cur][y_cur-1].cost2come:
            t_node[x_cur][y_cur-1].cost2come = t_node[x_cur][y_cur].cost2come + 1
            t_node[x_cur][y_cur-1].parent_idx = c_node
            openlist.put(((x_cur, y_cur-1), t_node[x_cur][y_cur-1].cost2come))

    # Move right
    if Isvalid(c_space, x_cur, y_cur+1):
        if (t_node[x_cur][y_cur].cost2come + 1) < t_node[x_cur][y_cur+1].cost2come:
            t_node[x_cur][y_cur+1].cost2come = t_node[x_cur][y_cur].cost2come + 1
            t_node[x_cur][y_cur+1].parent_idx = c_node
            openlist.put(((x_cur, y_cur+1), t_node[x_cur][y_cur+1].cost2come))

    # Move up-left
    if Isvalid(c_space, x_cur-1, y_cur-1):
        if (t_node[x_cur][y_cur].cost2come + 1.4) < t_node[x_cur-1][y_cur-1].cost2come:
            t_node[x_cur-1][y_cur-1].cost2come = t_node[x_cur][y_cur].cost2come + 1.4
            t_node[x_cur-1][y_cur-1].parent_idx = c_node
            openlist.put(
                ((x_cur-1, y_cur-1), t_node[x_cur-1][y_cur-1].cost2come))

    # Move up-right
    if Isvalid(c_space, x_cur-1, y_cur+1):
        if (t_node[x_cur][y_cur].cost2come + 1.4) < t_node[x_cur-1][y_cur+1].cost2come:
            t_node[x_cur-1][y_cur+1].cost2come = t_node[x_cur][y_cur].cost2come + 1.4
            t_node[x_cur-1][y_cur+1].parent_idx = c_node
            openlist.put(
                ((x_cur-1, y_cur+1), t_node[x_cur-1][y_cur+1].cost2come))

    # Move down-left
    if Isvalid(c_space, x_cur+1, y_cur-1):
        if (t_node[x_cur][y_cur].cost2come + 1.4) < t_node[x_cur+1][y_cur-1].cost2come:
            t_node[x_cur+1][y_cur-1].cost2come = t_node[x_cur][y_cur].cost2come + 1.4
            t_node[x_cur+1][y_cur-1].parent_idx = c_node
            openlist.put(((x_cur+1, y_cur-1), t_node[x_cur+1][y_cur-1]))

    # Move down-right
    if Isvalid(c_space, x_cur+1, y_cur+1):
        if (t_node[x_cur][y_cur].cost2come + 1.4) < t_node[x_cur+1][y_cur+1].cost2come:
            t_node[x_cur+1][y_cur+1].cost2come = t_node[x_cur][y_cur].cost2come + 1.4
            t_node[x_cur+1][y_cur+1].parent_idx = c_node
            openlist.put(
                ((x_cur+1, y_cur+1), t_node[x_cur+1][y_cur+1].cost2come))

    return t_node

# To compute minimum cost2come of popped node from the open list


def lowestC2C(node):
    c_node = (-1, -1)
    c2c_min = math.inf
    for i in range(len(node)):
        for j in range(len(node[i])):
            if node[i][j].cost2come < c2c_min and not node[i][j].explored:
                c_node = (i, j)
                c2c_min = node[i][j].cost2come
    return c_node

# backtracking from goal to start node, to get the shortest path


def backtrack(node, g_node, image):
    c_node = g_node
    x_c = c_node[0]
    y_c = c_node[1]
    queue = []
    s2g_images = []
    while c_node is not None:
        queue.append(c_node)
        c_node = node[x_c][y_c].parent_idx
    queue = queue[::-1]
    for c_node in queue:
        image[x_c][y_c] = (100, 80, 80)
        s2g_images.append(np.uint8(image.copy()))
        image[x_c][y_c] = (0, 0, 0)

    # saving the output video
    h, w = s2g_images[1].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    v = cv2.VideoWriter('path.mp4', fourcc, 55, (w, h), True)
    for i in range(len(s2g_images)):
        v.write(s2g_images[i])
    cv2.destroyAllWindows()
    v.release()
