import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from skimage.filters import sobel
from skimage.segmentation import watershed
from scipy import ndimage as ndi
from multiprocessing import Pool
from itertools import repeat
import random
    

def scheduler(epoch, lr):
    if (epoch+1) % 200 == 0:
        return lr * 3/4
    
    else:
        return lr
    
    
def load_img_paths(split_path):
    
    img_paths = []
    with open(split_path) as file:
        for line in file:
            img_paths.append(line.strip())
            
    return img_paths


def load_imgs(root_dir, img_paths, img_size):
    
    imgs = []
    for path in img_paths:
        path = os.path.join(root_dir, path)
        img = Image.open(path).resize((img_size, img_size))
        imgs.append(img)
    
    return imgs

    
def standardize(x):
    return np.array(x, dtype=np.float32)/127.5 - 1.


def destandardize(x, PIL=False):
    # convert an array for an image to PIL object
    
    assert len(x.shape) <= 3
    
    y = ((x + 1.)*127.5).astype(np.int32)
    
    if PIL:
        return Image.fromarray(y)
        
    else:
        return y

    
def get_input_pool(x, num_processors=2):
    y = []
    size_per_proc = len(x)//num_processors
                         
    for i in range(num_processors - 1):
        y.append(x[i*size_per_proc:(i+1)*size_per_proc])
    y.append(x[(num_processors-1)*size_per_proc:])
    
    return y


def get_xy_in_parallel(p, imgs, num_processors=1, mode=None, in_size=(64,64,3), augmentations=None, jitter=None, delta=1.): 
    
    x_in = get_input_pool(imgs, num_processors)
    outs = p.starmap(get_xy, zip(x_in, repeat(mode), repeat(in_size), repeat(augmentations), repeat(jitter), repeat(delta)))

    # Collect from multi processes
    x_out = np.concatenate([out[0] for out in outs], axis=0)
    y_out = np.concatenate([out[1] for out in outs], axis=0)
    
    return x_out, y_out

        
def get_xy(imgs, mode=None, in_size=(64,64), augmentations=None, jitter=None, delta=1.):
    
    ag_idx = np.random.choice(len(imgs), size=len(imgs)//2, replace=False)
    x_ag = np.empty((len(imgs),)+in_size)
    y_ag = np.zeros((len(imgs),), dtype=np.float32)

    for i in range(len(imgs)):

        x = imgs[i]

        # Traditional augmentation: jitter, flip
        if augmentations is not None:
            for augmentation in augmentations:
                x = augmentation(x)
        
        if i in ag_idx:
            x, _ = segmentation_ch_shuffle(x, rand_pixels=True, mode=mode, sobel_app=False, delta=delta)

        x_ag[i] = standardize(x)
    
    y_ag[ag_idx] = 1. # augmented
    
    return x_ag, y_ag


def segmentation_ch_shuffle(x, sobel_app=False, rand_pixels=False, mode='CH-Rand', delta=1., verbose=False):
    img = np.array(x)
    gray_img = x.convert('L')
    gray_img = np.asarray(gray_img)
    mask = np.zeros_like(gray_img)

    ########### Where? ###########
    if sobel_app:
        delta = .5
        q_low = np.random.random()*.5
        q_high = q_low + delta
        
        seg_mask = np.zeros_like(gray_img)
        seg_mask[gray_img < np.quantile(gray_img, q_low)] = 1
        seg_mask[gray_img > np.quantile(gray_img, q_high)] = 2
        
        edges = sobel(gray_img)
        segmentation = watershed(edges, seg_mask)
        segmentation = ndi.binary_fill_holes(segmentation - 1)
        labeled, _ = ndi.label(segmentation)
        
        # sample from top region
        max_l = np.max(labeled)
        counts = np.bincount(labeled.flatten())
        top_counts = sorted(counts)[-1:]
        weights = np.zeros(len(counts))
        for i, count in enumerate(counts):
            if count >= top_counts[0]:
#                 weights[i] = count/np.sum(top_counts)
                weights[i] = 1 #len(top_counts)
        weights = weights/np.sum(weights)
        shuffle_l = np.random.choice(np.arange(max_l+1), p=weights)
        mask[labeled == shuffle_l] = 1

    # n pixels in range (a, b) where n is delta*N
    else:
        lower_ths = np.random.random() * (1 - delta) 
        upper_ths = lower_ths + delta
        lower = np.quantile(gray_img, lower_ths)
        upper = np.quantile(gray_img, upper_ths)
        mask[(gray_img >= lower) & (gray_img <= upper)] = 1
    mask = mask.astype(bool)

    ########## How? ##########
    if mode == 'BLANK':
        img[mask] = 0
        chs = None
    
    elif mode == 'UNIFORM':
        img[mask] = np.random.uniform(low=0., high=255., size=img[mask].shape)
        chs = None

    # channel randomisation
    elif mode == 'CH-Rand':
        while True:
            chs = np.random.choice(3, 3, replace=True)
            if not np.all(chs == np.arange(3)):
                break
        img[mask] = img[mask][..., chs]

    # channel permutation
    elif mode == 'CH-Perm': 
        chs = [[0,2,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0]]
        ch = random.choice(range(len(chs)))
        img[mask] = img[mask][..., chs[ch]]
        chs = ch # test for 6-way cls
    
    # channel splitting
    elif mode == 'CH-Split':
        chs = np.random.choice(3)
        for i in range(3):
            img[mask, i] = img[mask][..., chs]

    elif mode == 'AGN':
        img[mask] = img[mask] + np.random.normal(scale=255*.1, size=img[mask].shape)
        chs = None
                     
    else:
        print('No {} mode'.format(mode))

    return img, chs   
