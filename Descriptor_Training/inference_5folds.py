""" Train with 5 folds """


# from __future__ import absolute_import, division, print_function, unicode_literals
import sys

sys.path.append('../..')
sys.path

import warnings
warnings.simplefilter("ignore", UserWarning)

import argparse
import os
import logging

import pandas as pd
import cv2
import time
import numpy as np

import tensorflow as tf
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

import numpy as np
import tqdm
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from skimage.transform import resize
import math
import pandas as pd
from ie590_project.model.input_fn import input_fn_features
from ie590_project.utils.utils import Params, build_train_validate


def set_logger(log_path): 
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
        
class VGG16_FC(tf.keras.Model):
    def __init__(self, hidden_size_1, hidden_size_2, num_classes=2, channel='rgb'):
        super(VGG16_FC, self).__init__()        
        #self.input_layer = tf.keras.layers.InputLayer(input_shape=num_features*25),
        self.fc1 = tf.keras.layers.Dense(hidden_size_1
                                         , activation=tf.nn.leaky_relu)
        self.fc2 = tf.keras.layers.Dense(hidden_size_2
                                         , activation=tf.nn.leaky_relu)
        self.fc3 = tf.keras.layers.Dense(num_classes
                                         , activation='softmax')
        if channel == 'rgb':
            self.ix = 0
        else:
            self.ix = 1
    
    def call(self, x, training=False):
        x = x[:,self.ix,:,:]
        frame_features = []
        for i in range(x.shape[1]):
            frame_features.append(x[:,i,:])
        x = np.concatenate(frame_features,axis=1)
        #x = self.input_layer(x)
        x = self.fc1(x)      
        x = self.fc2(x)
        x = self.fc3(x)
        
        return x

class VGG16_LSTM(tf.keras.Model):
    def __init__(self, num_features=512, hidden_size_1=128, num_classes=2, channel='rgb', dropout=0.2):
        super(VGG16_LSTM, self).__init__()        
        #self.input_layer = tf.keras.layers.InputLayer(input_shape=num_features*25),
        self.lstm = tf.keras.layers.LSTM(num_features, return_sequences=False,
                                               input_shape=(25,),
                                               dropout=dropout)
        self.fc1 = tf.keras.layers.Dense(hidden_size_1,activation=tf.nn.leaky_relu)
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.fc2 = tf.keras.layers.Dense(num_classes,activation='softmax')
        if channel == 'rgb':
            self.ix = 0
        else:
            self.ix = 1

    def call(self, x, training=False):
        x = x[:,self.ix,:,:]
        #x = self.input_layer(x)
        x = self.lstm(x)      
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x        

def predict_descriptor(model, inp, descriptor, csv_file, fold):
    """
    Args:
    - model: (tf.keras.Model) loaded model
    - inp: (tf.data.Dataset) 
    - descriptor: (str) descriptor name to predict
    - csv_file: (str) path for csv file to read
    - fold: (int) fold index of test set
    """
    
    inputs = iter(inp)
    predictions = []
    auc = tf.keras.metrics.AUC(name='AUC')
    recall = tf.keras.metrics.Recall(name='recall')
    accuracy = tf.keras.metrics.BinaryAccuracy(name='accuracy')
    
    for X, y in inputs:
        preds = model(X, training=False)
        preds = np.argmax(preds, axis=1)
        auc.update_state(y, preds)
        recall.update_state(y, preds)
        accuracy.update_state(y, preds)
        
        for i in preds:
            predictions.append(i)
    logging.info("--> AUC: {:.4f}, Recall: {:.4f}, Accuracy: {:.4f}".format(auc.result(), recall.result(), accuracy.result()))
                 
    df = pd.read_csv(csv_file)
    dist = df[descriptor].value_counts(normalize=True).sort_index()
    if len(dist) == 2:
        dist0 = dist[0]
        dist1 = dist[1]
    else:
        if dist.index[0]:
            dist0 = 0.0
            dist1 = 1.0
        else:
            dist0 = 1.0
            dist1 = 0.0
    logging.info("---> (original distribution: 0: {:.4f}%, 1: {:.4f}%)".format(dist0, dist1))
    df[descriptor] = predictions
    dist = df[descriptor].value_counts(normalize=True).sort_index()
    if len(dist) == 2:
        dist0 = dist[0]
        dist1 = dist[1]
    else:
        if dist.index[0]:
            dist0 = 0.0
            dist1 = 1.0
        else:
            dist0 = 1.0
            dist1 = 0.0
    logging.info("---> (prediction distribution: 0: {:.4f}%, 1: {:.4f}%)".format(dist0, dist1))
    
    df.to_csv(csv_file, index=False)
    logging.info("---> Updated descriptor '{}' predictions on {}".format(descriptor, csv_file))
    
      
if __name__ == "__main__":
    tf.random.set_seed(1234)

    USE_GPU = True

    if USE_GPU:
        device = '/device:GPU:0'
    else:
        device = '/cpu:0'
        
    desc_list = ['Both_Hands', 'F_Index', 'F_Middle', 'F_Pinky', 'F_Ring', 'F_Thumb', 'M_Back', 'M_Down', 'M_Front', 
                 'M_In', 'M_Iterative', 'M_Out', 'M_Up', 'O_Back', 'O_Down', 'O_Front', 'O_In', 'O_Out', 'O_Up']
        
    set_logger(os.path.join('.', 'inference_5folds.log')) # logger
    
    start_time = time.time()
    
    # train 5-folds
    for i in range(5):
        build_train_validate('annotated_videos_w_folds.csv', test_fold=i, model_type='INF_FC')
        build_train_validate('annotated_videos_w_folds.csv', test_fold=i, model_type='INF_LSTM')
        logging.info("=>> START INFERENCING.. FOLD-{}".format(i))
        # for each descriptors
        for desc in desc_list:
            logging.info("==>> DESCRIPTOR {}".format(desc))
            ######################
            # FC #################
            ######################
            params_json_path = 'params_' + desc + '.json'
            params = Params(params_json_path)
            params.descriptors = desc
            params.train_path = 'train_INF_FC_fold{}.csv'.format(i)
            params.validation_path = 'validation_INF_FC_fold{}.csv'.format(i)
            params.prediction_path = 'test_INF_FC_fold{}.csv'.format(i)
            
            # Load Datasets 
            params.validation_path = 'train_INF_FC_fold{}.csv'.format(i)
            train = input_fn_features(is_training=False, params=params)
            params.validation_path = 'validation_INF_FC_fold{}.csv'.format(i)
            val = input_fn_features(is_training=False, params=params)
            test = input_fn_features(is_training=False, params=params, prediction_mode=True)
            
            # Load Models
            model_loaded = VGG16_FC(2000,200)
            weight_path = '/scratch/gilbreth/choi502/ie590/models/FOLD-{}_DESC-{}_FC'.format(i, desc)
            model_loaded.load_weights(weight_path)
            
            # predict
            logging.info("-> FC train:")
            predict_descriptor(model_loaded, train, desc, params.train_path, i)
            logging.info("-> FC validation:")
            predict_descriptor(model_loaded, val, desc, params.validation_path, i)
            logging.info("-> FC test:")
            predict_descriptor(model_loaded, test, desc, params.prediction_path, i)
            
            logging.info("---> Elapsed time: {}".format(time.time() - start_time))
            ##########################
            # FC END #################
            ##########################
            
            ######################
            # LSTM ###############
            ######################
            params_json_path = 'params_' + desc + '.json'
            params = Params(params_json_path)
            params.descriptors = desc
            params.train_path = 'train_INF_LSTM_fold{}.csv'.format(i)
            params.validation_path = 'validation_INF_LSTM_fold{}.csv'.format(i)
            params.prediction_path = 'test_INF_LSTM_fold{}.csv'.format(i)
            
            # Loading datasets 
            params.validation_path = 'train_INF_LSTM_fold{}.csv'.format(i)
            train = input_fn_features(is_training=False, params=params)
            params.validation_path = 'validation_INF_LSTM_fold{}.csv'.format(i)
            val = input_fn_features(is_training=False, params=params)
            test = input_fn_features(is_training=False, params=params, prediction_mode=True)
            
            # Load Models
            model_loaded = VGG16_LSTM()
            weight_path = '/scratch/gilbreth/choi502/ie590/models/FOLD-{}_DESC-{}_LSTM'.format(i, desc)
            model_loaded.load_weights(weight_path)
            
            # predict
            logging.info("-> LSTM train:")
            predict_descriptor(model_loaded, train, desc, params.train_path, i)
            logging.info("-> LSTM validation:")
            predict_descriptor(model_loaded, val, desc, params.validation_path, i)
            logging.info("-> LSTM test:")
            predict_descriptor(model_loaded, test, desc, params.prediction_path, i)
            
            logging.info("--> Elapsed time: {}".format(time.time() - start_time))
            ##########################
            # LSTM END ###############
            ##########################
    
    end_time = time.time()
    

