import numpy as np
import time
import heapq
import cv2
import math


class node:
    def __init__(self, x, y, cost=math.inf, parent_idx=None):
        self.x = x
        self.y = y
        self.cost = cost
        self.parent_idx = parent_idx

# Defining action set to get new nodes


def moveUp(x, y, c2c):
    y += 1
    c2c += 1
    return x, y, c2c


def moveDown(x, y, c2c):
    y -= 1
    c2c += 1
    return x, y, c2c


def moveLeft(x, y, c2c):
    x -= 1
    c2c += 1
    return x, y, c2c


def moveRight(x, y, c2c):
    x += 1
    c2c += 1
    return x, y, c2c


def moveUp_Left(x, y, c2c):
    x -= 1
    y += 1
    c2c += math.sqrt(2)
    return x, y, c2c


def moveUp_Right(x, y, c2c):
    x += 1
    y += 1
    c2c += math.sqrt(2)
    return x, y, c2c


def moveDown_Left(x, y, c2c):
    x -= 1
    y -= 1
    c2c += math.sqrt(2)
    return x, y, c2c


def moveDown_Right(x, y, c2c):
    x += 1
    y -= 1
    c2c += math.sqrt(2)
    return x, y, c2c
