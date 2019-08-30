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
data_directory = '/Users/xiaoying/Downloads/CMELYON16'

def simple_tile(image_path, resolution):
    # just use grids to tile the slide, return a image list
    slide = OpenSlide(image_path)
    if resolution + 6 < slide.level_count:
        resolution = resolution + 3
    print('resolution = ', resolution)
    slide_width = slide.level_dimensions[resolution][0]
    slide_height = slide.level_dimensions[resolution][1]
    grid_number_x = slide_width // grid_size + 1
    grid_number_y = slide_height // grid_size + 1
    image_list = []
    for i in range(grid_number_y):
        for j in range(grid_number_x):
            startx = j * grid_size
            starty = i * grid_size
            tile_width = grid_size
            tile_height = grid_size
            if (i + 1) * grid_size > slide_height:
                startx = slide_height - grid_size
            if (j + 1) * grid_size > slide_width:
                starty = slide_width - grid_size
            if i * grid_size != slide_height and j * grid_size != slide_width:
                tile = np.array(slide.read_region((startx * 2 ** resolution, starty * 2 ** resolution), 
                                                   resolution, (tile_width, tile_height)))
                image_list.append(tile)
    return image_list

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
    var_fore = fore.var()
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

def select_grids(threshold, image_path, resolution):
    # use the original image to get the grid contain tissues
    grids = []
    slide = OpenSlide(image_path)
    slide_width = slide.level_dimensions[resolution][0]
    slide_height = slide.level_dimensions[resolution][1]
    grid_number_x = slide_width // grid_size + 1
    grid_number_y = slide_height // grid_size + 1
    for i in range(grid_number_y):
        for j in range(grid_number_x):
            startx = j * grid_size
            starty = i * grid_size
            tile_width = grid_size
            tile_height = grid_size
            if (i + 1) * grid_size > slide_height:
                starty = slide_height - grid_size
            if (j + 1) * grid_size > slide_width:
                startx = slide_width - grid_size
            if i * grid_size != slide_height and j * grid_size != slide_width:
                tile = np.array(slide.read_region((startx * 2 ** resolution, starty * 2 ** resolution),
                                                   resolution, (tile_width, tile_height)))
                gray_tile = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
                no_background = gray_tile < threshold
                if sum(sum(no_background)) / (grid_size * grid_size) > 0.1:
                    grid = (startx, starty)
                    grids.append(grid)
    return grids

def read_and_process(list_of_image_path, slides, grids, targets, mult, level):
    for image_path in list_of_image_path:
        image_list = simple_tile(image_path, 7)
        gray_image_list = convert_gray(image_list)
        threshold = select_thre(gray_image_list)
        grid = select_grids(threshold, image_path, 7)
        grids.append(grid)
        slides.append(image_path)
        if 'tumor' in image_path:
            targets.append(1)
        else:
            targets.append(0)
        mult.append(0)
        level.append(0)
        print(image_path + ' has precessed')

def create_trainlib(image_directory):
    slides = multiprocessing.Manager().list()
    grids = multiprocessing.Manager().list()
    targets = multiprocessing.Manager().list()
    mult = multiprocessing.Manager().list()
    level = multiprocessing.Manager().list()
    file_list = os.listdir(image_directory)
    list_of_image_path = []
    every_process_num = len(file_list) // 48
    for i in range(47):
        every_list = file_list[i * every_process_num : (i + 1) * ever_process_num]
        list_of_image_path.append(every_list)
    last_list = [46 * every_process_num : ]
    p = Pool(48)
    for i in range(48):
        p.apply_async(read_and_process, args=(list_of_image_path[i], slides, grids, targets, mult, level))
    p.close()
    p.join()
    dictory = {'slides': slides, 'grid': grids, 'targets': targets, 'mult': mult, 'level': level}
    torch.save(dictory, 'train_lib.pki')

def create_vallib(image_director):
    slides = multiprocessing.Manager().list()
    grids = multiprocessing.Manager().list()
    targets = multiprocessing.Manager().list()
    mult = multiprocessing.Manager().list()
    level = multiprocessing.Manager().list()
    file_list = os.listdir(image_directory)
    list_of_image_path = []
    every_process_num = len(file_list) // 48
    for i in range(47):
        every_list = file_list[i * every_process_num : (i + 1) * ever_process_num]
        list_of_image_path.append(every_list)
    last_list = [46 * every_process_num : ]
    p = Pool(48)
    for i in range(48):
        p.apply_async(read_and_process, args=(list_of_image_path[i], slides, grids, targets, mult, level))
    p.close()
    p.join()
    dictory = {'slides': slides, 'grid': grids, 'targets': targets, 'mult': mult, 'level': level}
    torch.save(dictory, 'val_lib.pki')

create_trainlib('/notebooks/19_ZZQ/CAMELYON16_data/train_data')
create_vallib('/notebooks/19_ZZQ/CAMELYON16_data/val_data')
