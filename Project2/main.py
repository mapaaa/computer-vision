#!/usr/bin/python3

import numpy as np
import sys

from matplotlib import pyplot as plt
from scipy.ndimage import sobel
from skimage import io
from skimage.color import rgb2gray

INF = 2 ** 62

def compute_energy(img):
    grayscale = rgb2gray(img)
    dy = sobel(grayscale, 1)
    dx = sobel(grayscale, 0)
    energy = np.hypot(dx, dy)
    energy *= 255.0 / np.max(energy)
    return energy


def get_best_column(energy):
    (n, m) = energy.shape
    dp = np.ndarray(energy.shape, dtype=energy.dtype)
    dp[0, :] = energy[0, :]

    for i in range(1, n):
        neighbours_left = np.append(INF, dp[i - 1][0:m - 1])
        neighbours_right = np.append(dp[i - 1][1:], INF)
        dp[i] = energy[i] + np.minimum(dp[i - 1], np.minimum(neighbours_left, neighbours_right))

    column = np.zeros(n).astype('int32')
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


def remove_column(img, column):
    (n, m, c) = img.shape
    new_img = np.zeros((n, m - 1, c)).astype('uint8')
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


    
# for dev only
fig = plt.figure(figsize=(1, 2))
image = io.imread('castel.jpg')
print(image.shape)
fig.add_subplot(1, 2, 1);
plt.imshow(image)
image = remove_lines(image, 20)
print(image.shape)
fig.add_subplot(1, 2, 2);
plt.imshow(image)
plt.show()
