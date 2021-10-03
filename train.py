import argparse
import yaml
import numpy as np
import glob
import os
import gc 
from multiprocessing import Pool
import random
import time
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
import torchvision
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from models import Cls
import utils

'''
Access config file 
'''
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', help='Path to config file (.YAML)')
options = parser.parse_args()

'''
Load config
'''
config_path = options.config
config_file = open(config_path)
config = yaml.safe_load(config_file)

'''
Load image files: train, val, test_norm, test_anom 
'''
print('Image data loaded ================================')

train_img_paths = utils.load_img_paths(config['train_split'])
train_x = utils.load_imgs(config['normal_dir'], train_img_paths, config['img_size'])

val_img_paths = utils.load_img_paths(config['val_split'])
val_x = utils.load_imgs(config['normal_dir'], val_img_paths, config['img_size'])

test_norm_img_paths = utils.load_img_paths(config['test_split'])
test_norm_x = utils.load_imgs(config['normal_dir'], test_norm_img_paths, config['img_size'])

test_anom_img_paths = glob.glob(os.path.join(config['anomalous_dir'], '**/*.{}'.format(config['anomalous_ext'])), 
                           recursive=True)
test_anom_x = utils.load_imgs('', test_anom_img_paths, config['img_size'])

'''
Prep for model to train
'''
print('Model created ================================')
in_size = (config['img_size'], config['img_size'], 3)

model = Cls(n_cls=1)
model.build(input_shape=(None,)+in_size)
model.compile(loss=['binary_crossentropy'], metrics=['accuracy'], 
              optimizer=keras.optimizers.Adam(learning_rate=config['lr']))
print(model.summary())

'''
Prep for train settings and params 
'''
lr_callback = keras.callbacks.LearningRateScheduler(utils.scheduler, verbose=0)

# Logs
model_id = datetime.now().strftime("%Y%m%d-%H%M%S")
tb_dir = "./{}/{}".format(config['tensorboard_dir'], model_id)
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=tb_dir, write_graph=False, profile_batch=0)
file_writer = tf.summary.create_file_writer(os.path.join(tb_dir, 'test'))
file_writer.set_as_default()
saved_path = './{}/{}.h5'.format(config['model_dir'], model_id)

if not os.path.isdir(config['model_dir']):
    os.makedirs(config['model_dir'], exist_ok=True)

# Traditional augmentations
color_jitter = torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
h_flip = torchvision.transforms.RandomHorizontalFlip()
v_flip = torchvision.transforms.RandomVerticalFlip()
augmentations = [color_jitter, h_flip, v_flip]

'''
Train
'''
p = Pool(processes=config['n_processors'])
val_accs, max_acc = [], -1

for e in range(config['n_epochs']):
    gc.collect()
    
    # Augment Train set
    train_x_ag = random.choices(train_x, k=config['batch_size']*config['epoch_size'])
    train_x_ag, train_y_ag = utils.get_xy_in_parallel(p, train_x_ag, config['n_processors'], 
                                    mode=config['aug_mode'], in_size=in_size, augmentations=augmentations, 
                                    jitter=color_jitter, delta=config['portion'])

    # Augment Val set
    if e % config['val_interval'] == 0:
        val_x_ag = random.choices(val_x, k=config['batch_size']*config['epoch_size'])
        val_x_ag, val_y_ag = utils.get_xy_in_parallel(p, val_x_ag, config['n_processors'], 
                                    mode=config['aug_mode'], in_size=in_size, augmentations=augmentations, 
                                    jitter=color_jitter, delta=config['portion'])

    # Fit
    print('\nFit ================================')
    start_time = time.time()
    history = model.fit(train_x_ag, train_y_ag, validation_data=(val_x_ag, val_y_ag), 
                batch_size=config['batch_size'], initial_epoch=e, epochs=e+1, 
                callbacks=[tensorboard_callback, lr_callback], 
                validation_freq=config['val_interval'], verbose=2)
    print('{:.03f} seconds to fit the model with lr={}'.format(time.time()-start_time, model.optimizer.lr.numpy()))

    # Test 
    if (e+1) % config['test_interval'] == 0:
        print('\nTest ================================')
        train_outs, test_norm_outs, test_anom_outs = [], [], []
        
        # Feat from train
        for i in range(len(train_x)):
            img = np.expand_dims(utils.standardize(train_x[i]), axis=0)
            train_outs.append(model.feature(img, fc_out=config['fc_feat']))
        train_outs = np.concatenate(train_outs, axis=0)

        # Feat from normal from test
        for i in range(len(test_norm_x)):
            img = np.expand_dims(utils.standardize(test_norm_x[i]), axis=0)
            test_norm_outs.append(model.feature(img, fc_out=config['fc_feat']))
        test_norm_outs = np.concatenate(test_norm_outs, axis=0)

        # Feat from anomalous from test
        for i in range(len(test_anom_x)):
            img = np.expand_dims(utils.standardize(test_anom_x[i]), axis=0)
            test_anom_outs.append(model.feature(img, fc_out=config['fc_feat']))
        test_anom_outs = np.concatenate(test_anom_outs, axis=0)

        # Evaluate
        rocs, prs = [], []
        for n in [1, 3, 5]:
            neigh = NearestNeighbors(n_neighbors=n, algorithm='auto')
            neigh.fit(train_outs)
            test_norm_dists = np.mean(neigh.kneighbors(test_norm_outs)[0], -1)
            test_anom_dists = np.mean(neigh.kneighbors(test_anom_outs)[0], -1)

            y_hats = np.concatenate([test_norm_dists, test_anom_dists])
            true_labels = [0] * len(test_norm_x) + [1] * len(test_anom_x)

            # Measure AUC in ROC and PR curves
            fpr, tpr, _ = roc_curve(true_labels, y_hats)
            pre, rec, _ = precision_recall_curve(true_labels, y_hats)
            rocs.append(auc(fpr, tpr))
            prs.append(auc(rec, pre))
            print('# neighbors={}, AUC-ROC = {:.3f}, AUC-PR = {:.3f}'.format(n, rocs[-1], prs[-1]))

        # Log best performance into Tensorboard
        roc_score = np.max(rocs)
        pr_score = np.max(prs)
        print('{:.3f}: AUC-ROC = {:.3f}, AUC-PR = {:.3f}'.format(e, roc_score, pr_score))
        tf.summary.scalar('AUC-ROC', data=roc_score, step=e)
        tf.summary.scalar('AUC-PR', data=pr_score, step=e)
        tf.summary.flush()
        print('{:.03f} seconds to test the model'.format(time.time()-start_time))
        
    # Check whether to stop training
    if (e+1) % config['val_interval'] == 0:
        print('\nValidate ================================')
        val_accs.append(history.history['val_accuracy'][0])

        if val_accs[-1] >= max_acc: 
            max_acc = val_accs[-1] 
            model.save_weights(saved_path)
            print('model saved at {} with max val_accuracy={:.3f}!'.format(saved_path, max_acc))
            
        else:
            print('no change to the max val_accuracy={:.3f}'.format(max_acc))

        if len(val_accs) >= config['stop_criterion'] \
            and np.mean(val_accs[-config['stop_criterion']:]) > config['val_acc_threshold']:
            print('val_accuracy reached {} so train ends'.format(config['val_acc_threshold']))
            break
