import cv2
import numpy as np
from openslide import OpenSlide
import os
import torch
import time
from multiprocessing import Pool
import multiprocessing

grid_size = 224
ratio = 0.1

def simple_tile(image_path, resolution):
    # just use grids to tile the slide, return a image list
    slide = OpenSlide(image_path)
    if resolution + 6 < slide.level_count:
        resolution = resolution + 3
    print('resolution = ', resolution)
    slide_width = slide.level_dimensions[resolution][0]
    slide_height = slide.level_dimensions[resolution][1]
    print('slide_width = ', slide_width)
    print('slide_height = ', slide_height)
    grid_number_x = slide_width // grid_size + 1
    grid_number_y = slide_height // grid_size + 1
    image_list = []
    for i in range(grid_number_y):
        for j in range(grid_number_x):
            startx = j * grid_size
            starty = i * grid_size
            tile_width = grid_size
            tile_height = grid_size
            if startx + tile_width > slide_width:
                startx = slide_width - tile_width - 1
            if starty + tile_height > slide_height:
                starty = slide_height - tile_height - 1
            if i * grid_size != slide_height and j * grid_size != slide_width:
                tile = np.array(slide.read_region((startx * 2 ** resolution, starty * 2 ** resolution), 
                                                   resolution, (tile_width, tile_height)))
                image_list.append(tile)
    return image_list

def plot_grids_in_image(image, grid, grid_width, grid_height):
    startx = grid[0]
    starty = grid[1]
    endx = startx + grid_width
    endy = starty + grid_height
    image[starty, startx: endx] = (0, 0, 0, 255)
    image[endy, startx: endx] = (0, 0, 0, 255)
    image[starty: endy, startx] = (0, 0, 0, 255)
    image[starty: endy, endx] = (0, 0, 0, 255)
    return image

def convert_gray(image_list):
    # convert a colour image to a gray one, because original image is too large
    # so we covert every tiles instead of covert the whole slide
    for image in image_list:
        if image.shape[-1] == 4:
            image = image[:, :, :-1]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image_list

def cal_var(histogram, threshold):
    back = histogram[histogram < threshold]
    fore = histogram[histogram >= threshold]
    var_back = back.var()
    if back.shape[0] == 0:
        var_back = 0
    var_fore = fore.var()
    if fore.shape[0] == 0:
        var_fore = 0
    var = (var_back * back.shape[0] + var_fore * fore.shape[0]) / (back.shape[0] + fore.shape[0])
    end = time.time()
    return var

def select_thre(image_list):
    # selcet the best threshold to get a minimum variance
    var = []
    histogram = np.array(image_list)
    for i in range(120, 220, 1):
        print('now thre', i)
        if i != 0:
            var.append(cal_var(histogram, i))
    minvar = min(var)
    thre = var.index(minvar) + 120
    return thre

def very_simple_tile(image_path):
    slide = OpenSlide(image_path)
    res = slide.level_count - 1
    whole_image = np.array(slide.read_region((0, 0), res, slide.level_dimensions[res]))
    whole_image = cv2.cvtColor(whole_image, cv2.COLOR_BGR2GRAY)
    histogram = whole_image.flatten()
    var = []
    for i in range(0, 256):
        if i != 0:
            var.append(cal_var(histogram, i))
    minvar = min(var)
    thre = var.index(minvar)
    return thre

def select_grids(threshold, image_path, resolution):
    # use the original image to get the grid contain tissues
    now_grids = []
    slide = OpenSlide(image_path)
    if resolution >= slide.level_count:
        resolution = slide.level_count - 1
    slide_width = slide.level_dimensions[resolution][0]
    slide_height = slide.level_dimensions[resolution][1]
    grid_number_x = slide_width // grid_size + 1
    grid_number_y = slide_height // grid_size + 1
    for i in range(grid_number_y):
        for j in range(grid_number_x):
            startx = j * grid_size
            starty = i * grid_size
            if starty + grid_size > slide_height:
                starty = slide_height - grid_size - 1
            if startx + grid_size > slide_width:
                startx = slide_width - grid_size - 1
            if i * grid_size != slide_height and j * grid_size != slide_width:
                tile = np.array(slide.read_region((startx * 2 ** resolution, starty * 2 ** resolution),
                                                   resolution, (grid_size, grid_size)))
                gray_tile = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
                no_background = gray_tile < threshold
            if sum(sum(no_background)) / (grid_size * grid_size) > 0.1:
                    grid = (startx, starty)
                    now_grids.append(grid)
    return now_grids

def create_lib(image_directory, category, resolution):
    slides = []
    grids = []
    targets = []
    mult = []
    level = []
    for image_path in os.listdir(image_directory):
        image_path = image_directory  + '/' + image_path
        print(image_path)
        slides.append(image_path)
        threshold = very_simple_tile(image_path)
        now_grids = select_grids(threshold, image_path, resolution)
        grids.append(now_grids)
        if 'tumor' in image_path:
            targets.append(1)
        if 'normal' in image_path:
            targets.append(0)
        mult.append(1)
        level.append(resolution)
        print(image_path + ' has been processed')
    dictory = {'slides': slides, 'grid': grids, 'targets': targets, 'mult': mult, 'level': level}
    torch.save(dictory, category + '_' + str(resolution) + '.pki')
train_image_directory = '/notebooks/19_ZZQ/CAMELYON16_data/train_data'
val_image_directory = '/notebooks/19_ZZQ/CAMELYON16_data/val_data'
test_image_directory = '/notebooks/19_ZZQ/CAMELYON16_data/test_data'
create_lib(train_image_directory, 'train', 3)
create_lib(val_image_directory, 'val', 3)
create_lib(test_image_directory, 'test', 3)
print('completed')
