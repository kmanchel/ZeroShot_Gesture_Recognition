"""Train the model"""

import argparse
import os
import logging
import time

import tensorflow as tf

from model.input_fn import input_fn
from model.model_fn import model_fn
from utils.utils import Params, set_logger, save_dict_to_json, get_train_test_split
from fit_and_predict import fit

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='./ie590_project/experiments/ex1', help="the directory where params.json is located")
parser.add_argument('--data_dir', default='./data/', help="the directory where data(videos' hdf5) is located")
parser.add_argument('--json_path', default='./data/rawdata.json', help="the raw data json file path")
parser.add_argument('--model_save_dir', default='./ie590_project/experiments/ex1/model_save/', help="the directory for saving ckpt and saver")
parser.add_argument('--restore_from', default=None, help="(Optional) ckpt or the directory where ckpts are located for restore saved parameters")
parser.add_argument('--train_size', default=None, help="(Optional) the number of dates of the training sample you want", type=int)

if __name__ == '__main__':
    start_time = time.time()
    
    #for reproducibility
    tf.compat.v1.set_random_seed(123)
    
    args = parser.parse_args()
    params_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(params_path), "params.json does not exits at {}".format(params_path)
    params = Params(params_path)
    
    #TODO: check and load if there's the best weights so far for retraining case
#     model_dir_has_best_weights = os.path.isdir(os.path.join(args.model_dir, "best_weights"))
    
    #set logger
    set_logger(os.path.join(args.model_dir, 'train.log'))
    
    #train/test split
    train_fpaths, test_fpaths, train_targets, test_targets = get_train_test_split(args.json_path, args.data_dir, train_size=args.train_size) 
    
    params.train_size = len(train_fpaths)
    params.test_size = len(test_fpaths)
    
    logging.info("Creating the dataset...")
    train_inputs = input_fn(True, train_fpaths, train_targets, params)
    test_inputs = input_fn(False, test_fpaths, test_targets, params)
    
    logging.info("Creating the model...")
    train_model_spec = model_fn(True, train_inputs, params)
    test_model_spec = model_fn(False, test_inputs, params, reuse=True)
    
    logging.info("Start training for {} epoch(s)".format(params.num_epochs))
    fit(train_model_spec, test_model_spec, args.model_save_dir, params, args.restore_from)
    
    end_time = time.time()
    logging.info("Elapsed training time is {:.2f} secs".format(end_time - start_time))