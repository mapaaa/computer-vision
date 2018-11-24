#!/usr/bin/python3

import numpy as np
import random
import sys

from matplotlib import pyplot as plt
from scipy.ndimage import sobel
from skimage import img_as_ubyte, io
from skimage.color import rgb2gray
from skimage.transform import rescale

INF = 2 ** 62
get_best_column = None

def compute_energy(img):
    grayscale = rgb2gray(img)
    dy = sobel(grayscale, 1)
    dx = sobel(grayscale, 0)
    energy = np.hypot(dx, dy)
    energy *= 255.0 / np.max(energy)
    return energy


def get_best_column_dynamicprogramming(energy):
    (n, m) = energy.shape
    dp = np.ndarray(energy.shape, dtype=energy.dtype)
    dp[0, :] = energy[0, :]

    for i in range(1, n):
        neighbours_left = np.append(INF, dp[i - 1][0:m - 1])
        neighbours_right = np.append(dp[i - 1][1:], INF)
        dp[i] = energy[i] + np.minimum(dp[i - 1], np.minimum(neighbours_left, neighbours_right))

    column = np.zeros(n, dtype = np.uint32)
    column[n - 1] = np.argmin(dp[n - 1])
    for i in range(n - 2, -1, -1):
        min_neighbour = dp[i][column[i + 1]]
        column[i] = column[i + 1]
        if column[i + 1] - 1 >= 0 and min_neighbour > dp[i][column[i + 1] - 1]:
            min_neighbour = dp[i][column[i + 1] - 1]
            column[i] = column[i + 1] - 1
        if column[i + 1] + 1 < m and min_neighbour > dp[i][column[i + 1] + 1]:
            min_neighbour = dp[i][column[i + 1] + 1]
            column[i] = column[i + 1] + 1
    return column


def get_best_column_random(energy):
    (n, m) = energy.shape;
    column = np.zeros(n, dtype = np.uint32)
    column[0] = random.randrange(0, m)
    for i in range(1, n):
        prev_column = column[i - 1]
        neighbours = [prev_column]
        if prev_column - 1 >= 0:
            neighbours.append(prev_column - 1)
        if prev_column + 1 < m:
            neighbours.append(prev_column + 1)
        column[i] = random.choice(neighbours)
    return column


def get_best_column_greedy(energy):
    (n, m) = energy.shape;
    column = np.zeros(n, dtype = np.uint32)
    column[0] = np.argmax(energy[0])
    for i in range(1, n):
        prev_column = column[i - 1]
        column[i] = prev_column
        if prev_column - 1 >= 0 and energy[i][prev_column - 1] < energy[i][column[i]]:
            column[i] = prev_column - 1;
        if prev_column + 1 < m and energy[i][prev_column + 1] < energy[i][column[i]]:
            column[i] = prev_column + 1;
    return column


def remove_column(img, column):
    (n, m, c) = img.shape
    new_img = np.zeros((n, m - 1, c), dtype = np.uint8)
    for i in range(n):
        new_img[i] = np.concatenate((img[i][0:column[i]], img[i][column[i] + 1:m]), axis = 0)
    return new_img

    
def remove_columns(img, cnt):
    for i in range(cnt):
        energy = compute_energy(img)
        column = get_best_column(energy)
        img = remove_column(img, column)
    return img


def remove_lines(img, cnt):
    img = np.rot90(img, 3)
    img = np.rot90(remove_columns(img, cnt))
    return img


def add_columns(img, cnt):
    (n, m, c) = img.shape
    cp_img = img
    columns_to_be_added = np.array([], dtype = np.uint32).reshape(n, 0)
    for i in range(cnt):
        energy = compute_energy(cp_img)
        column = get_best_column(energy)
        columns_to_be_added = np.hstack((columns_to_be_added, np.reshape(column, (n, 1))))
        cp_img = remove_column(cp_img, column)

    new_img = np.ndarray((n, m + cnt, 3), dtype = img.dtype) 
    for i in range(n):
        new_row = img[i]

        for j in range(cnt): 
            col = columns_to_be_added[i][j]
            if col - 1 >= 0:
                new_col1 = np.mean(np.array([new_row[col - 1], new_row[col]]), axis = 0)
            else:
                new_col1 = new_row[col]

            if col + 1 < len(new_row):
                new_col2 = np.mean(np.array([new_row[col], new_row[col + 1]]), axis = 0)
            else:
                new_col2 = new_row[col]

            new_col1 = np.reshape(new_col1, (1, c))
            new_col2 = np.reshape(new_col1, (1, c))
            new_row = np.concatenate((new_row[:col], new_col1, new_col2, new_row[col + 1:])) 

            for k in range(j + 1, cnt):
                if columns_to_be_added[i][k] >= col:
                    columns_to_be_added[i][k] += 2
        new_img[i] = new_row
    return new_img


def add_lines(img, cnt):
    img = np.rot90(img, 3)
    img = np.rot90(add_columns(img, cnt))
    return img


def amplify_content(img, factor):
    (n, m, c) = img.shape
    img = img_as_ubyte(rescale(img, factor, mode = 'reflect', multichannel = True, anti_aliasing = True))
    (new_n, new_m, new_c) = img.shape;

    if new_n > n:
        img = remove_lines(img, new_n - n)
    if new_m > m:
        img = remove_columns(img, new_m - m)
    
    return img
    


# for dev only
fig = plt.figure(figsize=(1, 2))
image = io.imread('castel.jpg')
fig.add_subplot(1, 2, 1);

plt.imshow(image)
algorithm = 'dynamicprogramming'
algorithms = {'dynamicprogramming': get_best_column_dynamicprogramming,
              'greedy': get_best_column_greedy,
              'random': get_best_column_random}
get_best_column = algorithms[algorithm]

image = add_columns(image, 50)
fig.add_subplot(1, 2, 2);
plt.imshow(image)
plt.show()
