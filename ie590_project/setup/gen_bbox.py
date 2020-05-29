"""generate and save bounding boxes"""

import argparse
import os
import logging

import pandas as pd
import cv2
import time
from scipy.io import savemat
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--annot_dir', default='../../data/train_annot/', help='the directory where keypoints data are')
parser.add_argument('--depimg_dir', default='../../../hand-graph-cnn/data/custom_dataset/IsoGD_phase_1_K/', help='the directory where depth images are')
parser.add_argument('--out_dir', default='../../../hand-graph-cnn/data/custom_dataset/bboxes/', help='the directory where you want to save bboxes mat file')
parser.add_argument('--trainlist_file', default='../../data/data_descriptors_mean.csv', help='csv file path which has list of videos paths')
parser.add_argument('--LR', default='L', help='L for left hand, R for right hand')


def set_logger(log_path): # TODO: do research about logging package
    """
    function; to set a logger to log model learning informations
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)
        
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)

# TODO included: 1
def get_MK_paths(annot_path, 
                 Mdata_dir='../../../hand-graph-cnn/data/custom_dataset/IsoGD_phase_1/train/', 
                 Kdata_dir='../../hand-graph-cnn/data/custom_dataset/IsoGD_phase_1_K/train/'):
    """return paths of RGB and D videos (in order by frames) given a video path"""
    num_frames = len(pd.read_csv(annot_path, header=None)) # TODO: is there a faster way to get the number of lines?
    dirname, vname = annot_path.split('.txt')[0].split('/')[-2:]
    vname = vname[2:]
    
    Mname = lambda i: 'M_' + vname + '_' + str(i) + '.png'
    Kname = lambda i: 'K_' + vname + '_' + str(i) + '.png'
    
    Mdata_paths = [os.path.join(Mdata_dir, dirname, Mname(i)) for i in range(num_frames)]
    Kdata_paths = [os.path.join(Kdata_dir, dirname, Kname(i)) for i in range(num_frames)]
    
    return Mdata_paths, Kdata_paths

def calculate_bboxes(hw_kpts, hs_kpts, kv_path, lr): # TODO included: 1
    """
    Args:
    - hw_kpts: (tuple) hand wrist keypoints of one hand (2,)
    - hs_kpts: (tuple) hand shoulder keypoints of both shoulder (4,)
    - kv_path: (str) K-video (depth) path
    """
    from math import ceil
    
    hw_x, hw_y = int(round(hw_kpts[0], 0)), int(round(hw_kpts[1]))
    rhs_x, rhs_y = int(round(hs_kpts[0])), int(round(hs_kpts[1]))
    lhs_x, lhs_y = int(round(hs_kpts[2])), int(round(hs_kpts[3]))
    
    k_img = cv2.imread(kv_path)
    
    try:
        hw_dep = k_img[hw_x, hw_y, 0] # I just took the firsth channel
        rhs_dep, lhs_dep = k_img[rhs_x, rhs_y, 0], k_img[lhs_x, lhs_y, 0]
    except IndexError:
        logging.info("--IndexError in {}".format(kv_path))
        return np.array([-1,-1,-1,-1])
        
    hw_kpts = np.array([hw_x, hw_y, hw_dep])
    lhs_kpts = np.array([lhs_x, lhs_y, lhs_dep])
    rhs_kpts = np.array([rhs_x, rhs_y, rhs_dep])
    
    s_len = np.linalg.norm(lhs_kpts - rhs_kpts) # shoulder length
#     print("DEBUGGING.", kv_path)
#     print("DEBUGGING... s_len {}, hw_dep {}, lhs_dep {}".format(s_len, hw_dep, lhs_dep))
    if lr == 'L':
        sw_dist = abs(lhs_dep - hw_dep) # left shoulder-wrist 'depth' distance (assuming hand comes to the front)
        
        lhw_dep_adj = min(hw_dep, lhs_dep) # assume hand comes to the front
        lhs_dep_adj = max(hw_dep, lhs_dep) # assume hand comes to the front
        try:
            scale = ceil((lhw_dep_adj / lhs_dep_adj) * s_len) # (rel) shoulder length at the depth of (left) hand wrist
        except ValueError:
            logging.info("--ValueError in {}".format(kv_path))
            scale = s_len
            
        bbox = [int(hw_y-0.8*scale), int(hw_x-1.5*scale), int(1.6*scale), int(2.1*scale)] # note that xy order should be reversed
        bbox = np.array(bbox)
    else:
        sw_dist = abs(rhs_dep - hw_dep) # right shoulder-wrist 'depth' distance (assuming hand comes to the front)
        
        rhw_dep_adj = min(hw_dep, rhs_dep) # assume hand comes to the front
        rhs_dep_adj = max(hw_dep, rhs_dep) # assume hand comes to the front
        try:
            scale = ceil((rhw_dep_adj / rhs_dep_adj) * s_len) # (rel) shoulder length at the depth of (right) hand wrist
        except ValueError:
            logging.info("--ValueError in {}".format(kv_path))
            scale = s_len
            
        bbox = [int(hw_y-0.8*scale), int(hw_x-1.5*scale), int(1.6*scale), int(2.1*scale)]
        bbox = np.array(bbox)
    
    return bbox


if __name__ == '__main__': #TODO included: 1
    start_time = time.time()
    
    set_logger(os.path.join('.', 'gen_bbox.log'))
    
    args = parser.parse_args()
    
    annot_dir = args.annot_dir
    depimg_dir = args.depimg_dir
    out_dir = args.out_dir
    trainlist_file = args.trainlist_file
    LR = args.LR
    
    out_dir = os.path.join(out_dir, LR)
    
    assert os.path.isfile(trainlist_file), "trainlist csv does not exists: {}".format(trainlist_file)
    assert os.path.isdir(annot_dir), "data driectory does not exists: {}".format(data_dir)
    assert os.path.isdir(depimg_dir), "data driectory does not exists: {}".format(data_dir)
    assert LR in ['L', 'R'], "wrong value for LR argument: {}".format(LR)
    if not os.path.isdir(out_dir):
        logging.info("Create output directory {}".format(out_dir))
        os.makedirs(out_dir)
        
    annot_path_fn = lambda p: os.path.join(annot_dir, p.split('.avi')[0] + '.txt')
    trainlist = pd.read_csv(trainlist_file).iloc[:, 0] 
    annotlist = [annot_path_fn(p) for p in trainlist] # annotation list
    
    train_length = len(trainlist)
    
    for i, annot_path in enumerate(annotlist):
        df = pd.read_csv(annot_path, header=None)
        df.columns = ['vname', *["N"+str(i) for i in np.arange(1, 29)]]
        
        if LR == 'L':
            hw_kpts = df[['N9','N10']] # L/R hand wrist keypoints
            hs_kpts = df[['N5','N6','N11','N12']] # L/R hand shoulder keypoints
        else:
            hw_kpts = df[['N15','N16']] # L/R hand wrist keypoints
            hs_kpts = df[['N5','N6','N11','N12']] # L/R hand shoulder keypoints
        
        _, kv_paths = get_MK_paths(annot_path, Kdata_dir=os.path.join(depimg_dir+'train'))
        
        bboxes = np.empty((len(df),4))
        for j, kv_path in enumerate(kv_paths): # TODO: delete this nested for loop
            bbox = calculate_bboxes(hw_kpts.iloc[j,:], hs_kpts.iloc[j,:], kv_path, lr=LR)
            bboxes[j] = bbox
        
        save_path = os.path.join(out_dir, *annot_path.split('.txt')[0].split('/')[-3:]) + '.mat'
        save_dir = os.path.join('.', *save_path.split('/')[:-1])
        if not os.path.isdir(save_dir):
            logging.info("--create output sub-directory {}".format(save_dir))
            os.makedirs(save_dir)
        savemat(save_path, {'bboxes': bboxes})
        
        if i % 1000 == 0:
            print("{:.2f}% processed".format(100 * i / train_length))
