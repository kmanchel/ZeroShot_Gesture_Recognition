"""convert videos into images(frames)"""

import argparse
import os
import logging

import pandas as pd
import cv2
import time

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='../../data/IsoGD_phase_1/', help='the directory where videos are')
parser.add_argument('--out_dir', default='../../../hand-graph-cnn/data/custom_dataset/IsoGD_phase_1/', help='the directory where you want to save frame images')
parser.add_argument('--trainlist_file', default='../../data/data_descriptors_mean.csv', help='csv file path which has list of videos paths')
parser.add_argument('--MK', default='M', help='M for RGB-videos, K for D-videos')

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

if __name__ == '__main__':
    start_time = time.time()
    
    set_logger(os.path.join('.', 'v2f.log'))
    
    args = parser.parse_args()
    
    data_dir = args.data_dir
    out_dir = args.out_dir
    trainlist_file = args.trainlist_file
    MK = args.MK
    
    assert os.path.isfile(trainlist_file), "train list csv does not exists: {}".format(trainlist_file)
    assert os.path.isdir(data_dir), "data driectory does not exists: {}".format(data_dir)
    assert MK in ['M', 'K'], "Wrong value for MK argument: {}".format(MK)
    if not os.path.isdir(out_dir):
        logging.info("Create output directory {}".format(out_dir))
        os.mkdir(out_dir)
        
    if MK == 'M':
        trainlist = pd.read_csv(trainlist_file).iloc[:, 0]
    else:
        trainlist = pd.read_csv(trainlist_file).iloc[:, 1]
    train_length = len(trainlist)
    
    logging.info("Extracting frames from video list")
    for i, vname in enumerate(trainlist):
        base_dir = os.path.join(*os.path.join(out_dir, vname).split('/')[:-1])
        if not os.path.isdir(base_dir):
            os.makedirs(base_dir)
            
        vpath = os.path.join(data_dir, vname)
        vidcap = cv2.VideoCapture(vpath)
        success, image = vidcap.read()
        count = 0
        while success:
            os.path.join(out_dir, vname).split('/')[:-1]
            fpath = os.path.join(out_dir, vname).split('.avi')[0]
            fpath = fpath + '_' + str(count) + '.png' # image file path (per frame)

            cv2.imwrite(fpath, image)
            success, image = vidcap.read()
            count += 1
            
        if i % 1000 == 0:
            print("{:.2f}% processed".format(100 * i / train_length))
    logging.info("Done!")
    
    end_time = time.time()
    logging.info("Elapsed time: {:.2f} secs".format(end_time - start_time))