import numpy as np
from PIL import Image
import os
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


def get_xy_in_parallel(p, imgs, num_processors=1, mode=None, in_size=(64,64,3), augmentations=None, 
                       jitter=None, delta=1., ch_label=False): 
    
    x_in = get_input_pool(imgs, num_processors)
    outs = p.starmap(get_xy, zip(x_in, repeat(mode), repeat(in_size), repeat(augmentations), 
                     repeat(jitter), repeat(delta), repeat(ch_label)))

    # Collect from multi processes
    x_out = np.concatenate([out[0] for out in outs], axis=0)
    y_out = np.concatenate([out[1] for out in outs], axis=0)
    
    return x_out, y_out

        
def get_xy(imgs, mode=None, in_size=(64,64), augmentations=None, jitter=None, delta=1., ch_label=False):

    x_ag = np.empty((len(imgs),)+in_size)
    y_ag = np.zeros((len(imgs),), dtype=np.float32)

    if ch_label:
        if mode == 'CH-Rand':
            ag_idx = random.sample(range(len(imgs)), int(len(imgs)*26/27))
        elif mode == 'CH-Perm':
            ag_idx = random.sample(range(len(imgs)), int(len(imgs)*5/6))
        
    else:
        ag_idx = random.sample(range(len(imgs)), len(imgs)//2)
    
    for i in range(len(imgs)):

        x = imgs[i]
        
        # Traditional augmentation: jitter, flip
        if augmentations is not None:
            for augmentation in augmentations:
                x = augmentation(x)
        
        if i in ag_idx:
            if mode == 'CH-Rand' or mode == 'CH-Perm':
                x, c = segmentation_ch_shuffle(x, rand_pixels=True, mode=mode, sobel_app=False, delta=delta)

            elif mode == 'CutPaste':
                x = cut_paste(x, area_ratios=(.02,.15), aspect_widths=(.3, 1.), aspect_heights=(1.,3.3),
                              jitter=augmentations[0], verbose=False)

            if ch_label:
                y_ag[i] = c + 1 # 1 ~ n_channels 

            else:
                y_ag[i] = 1.

        x_ag[i] = standardize(x)
    
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
        shuffle_l = random.choices(np.arange(max_l+1), weights=weights)
        mask[labeled == shuffle_l] = 1

    # n pixels in range (a, b) where n is delta*N
    else:
        lower_ths = random.random() * (1 - delta) 
        upper_ths = lower_ths + delta
        lower = np.quantile(gray_img, lower_ths)
        upper = np.quantile(gray_img, upper_ths)
        mask[(gray_img >= lower) & (gray_img <= upper)] = 1
    mask = mask.astype(bool)

    ########## How? ##########
    if mode == 'BLANK':
        img[mask] = 0
        ch_choice = 0

    # channel randomisation
    elif mode == 'CH-Rand':
        while True:
            chs = np.asarray(random.choices([0,1,2], k=3))
            if not np.all(chs == np.arange(3)):
                break
        chs = [[0,0,0], [0,0,1], [0,0,2], [0,1,0], [0,1,1], 
               [0,2,0], [0,2,1], [0,2,2], [1,0,0], [1,0,1], [1,0,2],
               [1,1,0], [1,1,1], [1,1,2], [1,2,0], [1,2,1], [1,2,2],
               [2,0,0], [2,0,1], [2,0,2], [2,1,0], [2,1,1], [2,1,2],
               [2,2,0], [2,2,1], [2,2,2]]
        ch_choice = random.choice(range(len(chs)))
        img[mask] = img[mask][..., chs[ch_choice]]

    # channel permutation
    elif mode == 'CH-Perm': 
        chs = [[0,2,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0]]
        ch_choice = random.choice(range(len(chs)))
        img[mask] = img[mask][..., chs[ch_choice]]
        
    # channel splitting
    elif mode == 'CH-Split':
        ch_choice = random.choice([0,1,2])
        for i in range(3):
            img[mask, i] = img[mask][..., ch_choice]
                     
    else:
        print('No {} mode'.format(mode))
        ch_choice = -1

    return img, ch_choice   


def cut_paste(img, area_ratios=(.02, .15), aspect_widths=(.3, 1.), aspect_heights=(1.,3.3), 
              verbose=False, jitter=None):

    (width, height) = img.size
    area_ratio = random.uniform(area_ratios[0], area_ratios[1])
    max_area = height * width * area_ratio
    
    # sample width and height ratios
    aspect_width = random.uniform(aspect_widths[0], aspect_widths[1])
    aspect_height = random.uniform(aspect_heights[0], aspect_heights[1])
    
    # determine width and height 
    unit = np.sqrt(max_area/(aspect_width*aspect_height))
    patch_width = np.minimum(int(unit * aspect_width), width)
    patch_height = np.minimum(int(unit * aspect_height), height)
    
    # cut 
    rand_x = random.randint(0, width-patch_width) if width-patch_width >= 1 else 0
    rand_y = random.randint(0, height-patch_height) if height-patch_height >= 1 else 0
    
    # vars for paste
    while True:
        rand_x_p = random.randint(0, width-patch_width) if width-patch_width >= 1 else 0
        rand_y_p = random.randint(0, height-patch_height) if height-patch_height >= 1 else 0

        if rand_x != rand_x_p or rand_y != rand_y_p: 
            break
            
    # extract patch
    patch = img.crop((rand_x, rand_y, rand_x+patch_width, rand_y+patch_height))
    
    # jitter in patch
    patch = patch if jitter is None else jitter(patch)
    
    if verbose:
        print('area ratio={:.03f}'.format(area_ratio))
        print('aspect width={:.03f}, aspect height={:.03f}'.format(aspect_width, aspect_height))
        print('patch width={:.03f}, patch height={:.03f}'.format(patch_width, patch_height))
        print('cut from ({}, {})'.format(rand_x, rand_y))
        print('paste at ({}, {})\n'.format(rand_x_p, rand_y_p))
    
    # make numpy array
    img, patch = np.array(img), np.asarray(patch)

    # paste
    img[rand_y_p:rand_y_p+patch_height, rand_x_p:rand_x_p+patch_width] = patch
    
    # if random.random() < .01: 
    #     img2 = Image.fromarray(img)
    #     img2.save('cutpaste_ex.png')

    return img