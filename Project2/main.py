#!/usr/bin/python3

import argparse
import random
import sys
import numpy as np

from matplotlib import pyplot as plt
from scipy.ndimage import sobel
from skimage import img_as_ubyte, io
from skimage.color import rgb2gray
from skimage.viewer import ImageViewer
from skimage.viewer.canvastools import RectangleTool
from skimage.transform import rescale

INF = 2 ** 62
get_best_column = None


def plot_column(img, energy, column):
    img_with_seam = img
    (n, m, c) = img_with_seam.shape
    global seam_color
    for i in range(n):
        img_with_seam[i][column[i]] = seam_color
    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    plt.imshow(img_with_seam)
    fig.add_subplot(1, 2, 2)
    plt.imshow(energy, cmap='gray')
    plt.pause(1.5)
    plt.close()


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

    column = np.zeros(n, dtype=np.uint32)
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
    (n, m) = energy.shape
    column = np.zeros(n, dtype=np.uint32)
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
    (n, m) = energy.shape
    column = np.zeros(n, dtype=np.uint32)
    column[0] = np.argmax(energy[0])
    for i in range(1, n):
        prev_column = column[i - 1]
        column[i] = prev_column
        if prev_column - 1 >= 0 and energy[i][prev_column - 1] < energy[i][column[i]]:
            column[i] = prev_column - 1
        if prev_column + 1 < m and energy[i][prev_column + 1] < energy[i][column[i]]:
            column[i] = prev_column + 1
    return column


def remove_column(img, column):
    (n, m, c) = img.shape
    new_img = np.zeros((n, m - 1, c), dtype=np.uint8)
    for i in range(n):
        new_img[i] = np.concatenate((img[i][0:column[i]], img[i][column[i] + 1:m]), axis=0)
    return new_img


def remove_columns(img, cnt):
    global plot_seam
    for i in range(cnt):
        energy = compute_energy(img)
        column = get_best_column(energy)
        if plot_seam:
            plot_column(img, energy, column)
        img = remove_column(img, column)
    return img


def remove_lines(img, cnt):
    img = np.rot90(img, 3)
    img = np.rot90(remove_columns(img, cnt))
    return img


def add_columns(img, cnt):
    (n, m, c) = img.shape
    cp_img = img
    columns_to_be_added = np.array([], dtype=np.uint32).reshape(n, 0)
    global plot_seam
    for i in range(cnt):
        energy = compute_energy(cp_img)
        column = get_best_column(energy)
        if plot_seam:
            plot_column(cp_img, energy, column)
        columns_to_be_added = np.hstack((columns_to_be_added, np.reshape(column, (n, 1))))
        cp_img = remove_column(cp_img, column)

    new_img = np.ndarray((n, m + cnt, 3), dtype=img.dtype)
    for i in range(n):
        new_row = img[i]

        for j in range(cnt):
            col = columns_to_be_added[i][j]
            if col - 1 >= 0:
                new_col1 = np.mean(np.array([new_row[col - 1], new_row[col]]), axis=0)
            else:
                new_col1 = new_row[col]

            if col + 1 < len(new_row):
                new_col2 = np.mean(np.array([new_row[col], new_row[col + 1]]), axis=0)
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
    img = img_as_ubyte(rescale(img, factor, mode='reflect', multichannel=True, anti_aliasing=True))
    (new_n, new_m, new_c) = img.shape

    if new_n > n:
        img = remove_lines(img, new_n - n)
    if new_m > m:
        img = remove_columns(img, new_m - m)

    return img


def get_coord(extents):
    global viewer
    img = viewer.image
    global obj_coord
    obj_coord = np.int64(extents)


def remove_object(img, coord):
    (n, m, c) = img.shape
    imgRotated = False
    x1 = coord[2]
    x2 = coord[3]
    y1 = coord[0]
    y2 = coord[1]

    if x2 - x1 < y2 - y1:
        # Rotate the image because it's faster to remove lines than columns
        img = np.rot90(img, 3)
        imgRotated = True
        y1, x1 = n - x1 + 1, y1
        y2, x2 = n - x2 + 1, y2
        y1, y2 = y2, y1

    # Object's coordinates are considered [x1, x2) and [y1, y2)
    x2 += 1
    y2 += 1

    cnt = y2 - y1
    global plot_seam
    while (cnt):
        energy = compute_energy(img)
        if plot_seam:
            cp_energy = energy
        energy[x1:x2, :y1].fill(INF)
        energy[x1:x2, y2:].fill(INF)
        column = get_best_column(energy)
        if plot_seam:
            plot_column(img, cp_energy, column)
        img = remove_column(img, column)
        y2 -= 1
        cnt -= 1

    if imgRotated:
        img = np.rot90(img)

    return img


def main():
    algorithms = {'dynamicprogramming': get_best_column_dynamicprogramming,
                  'greedy': get_best_column_greedy,
                  'random': get_best_column_random}
    default_algorithm = 'dynamicprogramming'

    parser = argparse.ArgumentParser(description='Content aware image resizing using seam carving.')
    parser.add_argument('source', type=str, help='Input picture to be resized.')
    parser.add_argument('output', type=str, help='Output with the resized picture.')
    parser.add_argument('--width', type=int, help='Change in width.')
    parser.add_argument('--height', type=int, help='Change in height.')
    parser.add_argument('--amplify-content',
                        type=float,
                        help='Amplify content with a desired factor.' +
                        'For example, use 1.2 to amplify the content with 20%.')
    parser.add_argument('--remove-rectangle',
                        action='store_true',
                        help='Select rectangle objects to be removed.')
    parser.add_argument('--algorithm',
                        type=str,
                        default=default_algorithm,
                        choices=algorithms.keys(),
                        help='Strategy to be used for seam selection.')
    parser.add_argument('--plot-result',
                        action='store_true',
                        help='Plots the original image and the resized one side by side')
    parser.add_argument('--plot-seam',
                        action='store_true',
                        help='Select a color in RGB format and plots the seam at each step.')

    args = parser.parse_args()

    img = io.imread(args.source)
    original_image = img

    global get_best_column
    get_best_column = algorithms.get(args.algorithm, default_algorithm)

    global plot_seam
    global seam_color
    plot_seam = args.plot_seam
    seam_color = (255, 0, 0)

    if args.remove_rectangle:
        global viewer
        viewer = ImageViewer(img)
        global obj_coord
        obj_coord = []
        rect_tool = RectangleTool(viewer, on_enter=get_coord)
        viewer.show()
        img = remove_object(img, obj_coord)

    if args.width:
        if args.width < 0:
            img = remove_column(img, -args.width)
        else:
            img = add_columns(img, args.width)

    if args.height:
        if args.height < 0:
            img = remove_lines(img, -args.height)
        else:
            img = add_lines(img, args.height)

    io.imsave(args.output, img)

    if args.plot_result:
        fig = plt.figure(figsize=(1, 2))
        fig.add_subplot(1, 2, 1)
        plt.imshow(original_image)
        fig.add_subplot(1, 2, 2,)
        plt.imshow(img)
        plt.show()


if __name__ == '__main__':
    sys.exit(main())
