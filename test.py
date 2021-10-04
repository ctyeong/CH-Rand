import argparse
import yaml
import glob
import os
import numpy as np
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
Load image files: train, test_norm, test_anom 
'''
print('Image data loaded ================================')

train_img_paths = utils.load_img_paths(config['train_split'])
train_x = utils.load_imgs(config['normal_dir'], train_img_paths, config['img_size'])

test_norm_img_paths = utils.load_img_paths(config['test_split'])
test_norm_x = utils.load_imgs(config['normal_dir'], test_norm_img_paths, config['img_size'])

test_anom_img_paths = glob.glob(os.path.join(config['anomalous_dir'], '**/*.{}'.format(config['anomalous_ext'])), 
                           recursive=True)
test_anom_x = utils.load_imgs('', test_anom_img_paths, config['img_size'])

'''
Load model 
'''

print('Model loaded ================================')
in_size = (config['img_size'], config['img_size'], 3)

model = Cls(n_cls=1)
model.build(input_shape=(None,)+in_size)
model.load_weights(config['test']['model_path'])
print(model.summary())


'''
Extract features 
'''
print('\nFeatures extracted ================================')
train_outs, test_norm_outs, test_anom_outs = [], [], []
for i in range(len(train_x)):
    img = np.expand_dims(utils.standardize(train_x[i]), axis=0)
    train_outs.append(model.feature(img, fc_out=config['fc_feat']))
train_outs = np.concatenate(train_outs, axis=0)

test_norm_x_arr = np.asarray([np.array(x) for x in test_norm_x], dtype=np.float32)
for i in range(len(test_norm_x_arr)):
    img = np.expand_dims(utils.standardize(test_norm_x_arr[i]), axis=0)
    test_norm_outs.append(model.feature(img, fc_out=config['fc_feat']))
test_norm_outs = np.concatenate(test_norm_outs, axis=0)

test_anom_x_arr = np.asarray([np.array(x) for x in test_anom_x], dtype=np.float32)
for i in range(len(test_anom_x_arr)):
    img = np.expand_dims(utils.standardize(test_anom_x_arr[i]), axis=0)
    test_anom_outs.append(model.feature(img, fc_out=config['fc_feat']))
test_anom_outs = np.concatenate(test_anom_outs, axis=0)

'''
Distance calculation with nearest neighbors in Train set
'''
print('\nEvaluation results ================================')
for i in config['n_neighbors']:
    neigh = NearestNeighbors(n_neighbors=i, algorithm='auto')
    neigh.fit(train_outs)
    test_norm_dists = np.mean(neigh.kneighbors(test_norm_outs)[0], -1)
    test_anom_dists = np.mean(neigh.kneighbors(test_anom_outs)[0], -1)

    y_hats = np.concatenate([test_norm_dists, test_anom_dists])
    true_labels = [0] * len(test_norm_x) + [1] * len(test_anom_x)

    fpr, tpr, _ = roc_curve(true_labels, y_hats)
    pre, rec, _ = precision_recall_curve(true_labels, y_hats)
    
    roc_score = auc(fpr, tpr)
    pr_score = auc(rec, pre)

    print('{} neighbor(s): AUC-ROC = {:.3f}, AUC-PR = {:.3f}'.format(i, roc_score, pr_score))