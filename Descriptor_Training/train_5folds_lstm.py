""" Train with 5 folds """


# from __future__ import absolute_import, division, print_function, unicode_literals
import sys

sys.path.append('../..')
sys.path

import argparse
import os
import logging

import pandas as pd
import cv2
import time
import numpy as np

import tensorflow as tf

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
    
def model_init_fn():
    model = VGG16_FC(2000,200)
    return model

def model_init_fn_lstm():
    model = VGG16_LSTM()
    return model        
        
def train_part34(model_init_fn, compile_params):
    """
    Simple training loop for use with models defined using tf.keras. It trains
    a model for one epoch on the CIFAR-10 training set and periodically checks
    accuracy on the CIFAR-10 validation set.
    
    Inputs:
    - model_init_fn: A function that takes no parameters; when called it
      constructs the model we want to train: model = model_init_fn()
    - optimizer_init_fn: A function which takes no parameters; when called it
      constructs the Optimizer object we will use to optimize the model:
      optimizer = optimizer_init_fn()
    - num_epochs: The number of epochs to train for
    
    Returns: Model, History (Loss, AUC, Recall, Accuracy)
    """    
    with tf.device(compile_params['device']):

        # Compute the loss like we did in Part II
        loss_fn = tf.keras.losses.BinaryCrossentropy()
        
        model = model_init_fn()
        optimizer = tf.keras.optimizers.Adam(learning_rate=compile_params["learning_rate"])
        #Calculate Loss
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        val_loss = tf.keras.metrics.Mean(name='val_loss')
        #Area Under Curve 
        train_auc = tf.keras.metrics.AUC(name='train_auc')
        val_auc = tf.keras.metrics.AUC(name='val_auc')
        #Recall
        train_recall = tf.keras.metrics.Recall(name='train_recall')
        val_recall = tf.keras.metrics.Recall(name='val_recall')
        #Accuracy
        train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')
        val_accuracy = tf.keras.metrics.BinaryAccuracy(name='val_accuracy')
        
        #Logging all the above for plots later on
        train_loss_log = []
        val_loss_log = []
        train_auc_log = []
        val_auc_log = []
        train_recall_log = []
        val_recall_log = []
        train_accuracy_log = []
        val_accuracy_log = []
        
        t = 0
        best_val_loss = 1000
        best_model = None
        num_watch = 0 # for early stopping
        
        for epoch in range(compile_params["num_epochs"]):
            train_dset = iter(compile_params['train_data'])
            
            # Reset the metrics - https://www.tensorflow.org/alpha/guide/migration_guide#new-style_metrics
            train_loss.reset_states()
            train_accuracy.reset_states()
            train_recall.reset_states()
            train_auc.reset_states()
    
            for x_np, y_np in train_dset:
                y_np = tf.keras.utils.to_categorical(y_np, num_classes=2, dtype='int32')
                with tf.GradientTape() as tape:
                    
                    # Use the model function to build the forward pass.
                    scores = model(x_np, training=True)
                    loss = loss_fn(y_np, scores)
      
                    gradients = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                    
                    # Update the metrics
                    train_loss.update_state(loss)
                    train_auc.update_state(y_np, scores)
                    train_recall.update_state(y_np, scores)
                    train_accuracy.update_state(y_np, scores)
                    
                    if t % compile_params["print_every"] == 0:
                        val_loss.reset_states()
                        val_accuracy.reset_states()
                        val_recall.reset_states()
                        val_auc.reset_states()
                        
                        val_dset = iter(compile_params['val_data'])
                        
                        for test_x, test_y in val_dset:
                            # During validation at end of epoch, training set to False
                            test_y = tf.keras.utils.to_categorical(test_y, num_classes=2)
                            prediction = model(test_x, training=False)
                            t_loss = loss_fn(test_y, prediction)
                            
                            val_loss.update_state(t_loss)
                            sparse_prediction = tf.keras.utils.to_categorical(np.argmax(prediction,axis=1), num_classes=2)
                            val_auc.update_state(test_y, prediction)
                            val_recall.update_state(test_y, sparse_prediction)
                            val_accuracy.update_state(test_y, prediction)
                            #print("Debug:",val_accuracy.result())

                        
                        template = '--> Iteration {}, Epoch {}, Train Loss: {:.4f}, Train {}: {:.4f}, Val Loss: {:.4f}, Val {}: {:.4f}'
                        logging.info(template.format(t, epoch+1, 
                                             train_loss.result(),
                                            compile_params['metric'],
                                             train_accuracy.result(),
                                             val_loss.result(),
                                            compile_params['metric'],
                                             val_accuracy.result()))
                    t += 1
            train_loss_log.append(train_loss.result())
            val_loss_log.append(val_loss.result())
            train_auc_log.append(train_auc.result())
            val_auc_log.append(val_auc.result())
            train_recall_log.append(train_recall.result())
            val_recall_log.append(val_recall.result())
            train_accuracy_log.append(train_accuracy.result())
            val_accuracy_log.append(val_accuracy.result())   
            
            # early stopping
            val_loss_eval = val_loss.result()
            if (best_val_loss < val_loss_eval) and (num_watch >= 5):
                logging.info("--> Early Stopped at {}th Epoch.".format(epoch+1))
                history = {"loss":{"train":train_loss_log,"validation":val_loss_log},
                      "auc":{"train":train_auc_log,"validation":val_auc_log},
                      "recall":{"train":train_recall_log,"validation":val_recall_log},
                      "accuracy":{"train":train_accuracy_log,"validation":val_accuracy_log}}
                return(best_model, history)
            elif best_val_loss < val_loss_eval:
                num_watch += 1
            else:
                best_val_loss = val_loss_eval
                best_model = model
                num_watch = 0
            
        history = {"loss":{"train":train_loss_log,"validation":val_loss_log},
                      "auc":{"train":train_auc_log,"validation":val_auc_log},
                      "recall":{"train":train_recall_log,"validation":val_recall_log},
                      "accuracy":{"train":train_accuracy_log,"validation":val_accuracy_log}}
        
        return(best_model, history)
        
def save_plots(hist, desc, modelname):
    title = 'Descriptor:' + desc + ', ' + 'Model:' + 'VGG16-' + modelname
    
    fig, axs = plt.subplots(2, 2)
    fig.set_size_inches(15, 10)
    axs[0, 0].plot(hist['loss']['train'])
    axs[0, 0].plot(hist['loss']['validation'])
    axs[0, 0].set_title('Loss')
    axs[0, 0].set(ylabel='Loss')
    axs[0, 1].plot(hist['accuracy']['train'])
    axs[0, 1].plot(hist['accuracy']['validation'])
    axs[0, 1].set_title('Accuracy')
    axs[0, 1].set(ylabel='Accuracy')
    axs[1, 0].plot(hist['auc']['train'])
    axs[1, 0].plot(hist['auc']['validation'])
    axs[1, 0].set_title('AUC')
    axs[1, 0].set(ylabel='AUC')
    axs[1, 1].plot(hist['recall']['train'])
    axs[1, 1].plot(hist['recall']['validation'])
    axs[1, 1].set_title('Recall')
    axs[1, 1].set(ylabel='Recall')

    for ax in axs.flat:
        ax.set(xlabel='Epochs')

    fig.suptitle(title)
    fig.legend(['train', 'test'], loc='upper left')
    
    plt.savefig('plots/'+title+'.png')
    logging.info("==> {} is saved".format('plots/'+title+'.png'))
    
      
if __name__ == "__main__":
    tf.random.set_seed(1234)

    USE_GPU = True

    if USE_GPU:
        device = '/device:GPU:0'
    else:
        device = '/cpu:0'
        
    desc_list = ['Both_Hands', 'F_Index', 'F_Middle', 'F_Pinky', 'F_Ring', 'F_Thumb', 'M_Back', 'M_Down', 'M_Front', 
                 'M_In', 'M_Iterative', 'M_Out', 'M_Up', 'O_Back', 'O_Down', 'O_Front', 'O_In', 'O_Out', 'O_Up']
        
    set_logger(os.path.join('.', 'train_5folds_lstm.log')) # logger
    
    start_time = time.time()
    
    # train 5-folds
    for i in range(5):
        build_train_validate('annotated_videos_w_folds.csv', test_fold=i, model_type='LSTM')
        
        # for each descriptors
        for desc in desc_list:
            logging.info("START FITTING.. FOLD-{}, DESCRIPTOR-{}".format(i, desc, time.time()-start_time))
            params_json_path = 'params_' + desc + '.json'
            params = Params(params_json_path)
            params.descriptors = desc
            params.train_path = 'train_LSTM.csv'
            params.validation_path = 'validation_LSTM.csv'
            params.prediction_path = 'test_LSTM.csv'
            
            train = input_fn_features(is_training=True,params=params)
            val = input_fn_features(is_training=False,params=params)
            
            # FC
#             compile_params = {
#                 "device": device,
#                 "train_data": train,
#                 "val_data": val,
#                 "learning_rate": params.learning_rate_fc,
#                 "num_epochs": params.num_epochs,
#                 "metric": "Accuracy",
#                 "descriptor_type": "mode",
#                 "print_every": 40,
#             }
#             both_hands_model_fc, both_hands_history_fc = train_part34(model_init_fn, compile_params)
#             save_plots(both_hands_history_fc, desc, 'FC')
#             ckpt_path = os.path.join('/scratch/gilbreth/choi502/ie590/models/', 'FOLD-{}_DESC-{}_FC'.format(i, desc))
#             both_hands_model_fc.save_weights(ckpt_path)
#             logging.info("==> {} is saved".format(ckpt_path))
#             logging.info("==> Elapsed time: {}".format(time.time() - start_time))
            
            # LSTM
            compile_params = {
                "device": device,
                "train_data": train,
                "val_data": val,
                "learning_rate": params.learning_rate_lstm,
                "num_epochs": params.num_epochs,
                "metric": "Accuracy",
                "descriptor_type": "mode",
                "print_every": 40,
            }
            both_hands_model_lstm, both_hands_history_lstm = train_part34(model_init_fn_lstm, compile_params)
            save_plots(both_hands_history_lstm, desc, 'LSTM')
            ckpt_path = os.path.join('/scratch/gilbreth/choi502/ie590/models/', 'FOLD-{}_DESC-{}_LSTM'.format(i, desc))
            both_hands_model_lstm.save_weights(ckpt_path)
            logging.info("==> {} is saved".format(ckpt_path)) 
            logging.info("==> Elapsed time: {}".format(time.time() - start_time))
    
    end_time = time.time()
    
