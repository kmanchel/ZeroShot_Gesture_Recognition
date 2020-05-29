"""Train the model """
"""You can run this .py as a main python file or import predict() function only depedning on your preference"""

import argparse
import os
import logging
import time
import numpy as np
from tqdm import tqdm_notebook as tqdm

import tensorflow as tf

from model.input_fn import input_fn
from model.model_fn import model_fn
from utils.utils import Params, set_logger, save_dict_to_json, get_train_test_split
from fit_and_predict import fit

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='./ie590_project/experiments/ex1', help="the directory where params.json is located")
parser.add_argument('--data_dir', default='./data/', help="the directory where data(videos' hdf5) is located")
parser.add_argument('--json_path', default='./data/rawdata.json.json', help="the json file path")
parser.add_argument('--model_save_dir', default='./ie590_project/experiments/ex1/model_save/', help="the directory for saving ckpt and saver")
parser.add_argument('--restore_from', default=None, help="(Optional) ckpt or the directory where ckpts are located for restore saved parameters")
parser.add_argument('--train_size', default=None, help="(Optional) the number of dates of the training sample you want", type=int)


def predict(inp, target, params, restore_from, config=None,\
            model_dir='./ie590_project/experiments/ex1', model_save_dir='./ie590_project/experiments/ex1/model_save/1'):
    """predict target values given input file paths
    Args:
        inp: (list) a string list of image files paths; 2D -> [sample_size, number_of_channels]
        model_spec: (dict) model specifications of tf Ops
        params: (Params or str) Params object or params.joson path
        tar: (list) a float list of target values
        restore_from: (str) ckpt or directory name where ckpts are located for restoring
        ...
    Return:
        out: (list) a list of precicted target values; have exactly same dimension as target
    """
    
    assert len(inp) == len(target)
    
    iterator_init_op = model_spec['iterator_init_op']
    update_metrics_op = model_spec['update_metrics_op']
    metrics = model_spec['metrics']
    metrics_init_op = model_spec['metrics_init_op']
    predictions = model_spec['predictions']
    
    saver = tf.compat.v1.train.Saver()
    
    if type(params) is str:
        assert os.path.isfile(params), "params.json does not exits at {}".format(params)
        params = Params(params)
        params.load(params.update) # load parameters
    params.inp_size = len(inp)
    
    set_logger(os.path.join(model_dir, 'train.log'))
    
    logging.info("Creating the dataset...")
    inputs = input_fn(False, inp, target, params)
    
    logging.info("Creating the model...")
    model_spec = model_fn(False, inputs, params)
    
    logging.info("Calculating predictions...")
    with tf.compat.v1.Session(config=config) as sess:
        sess.run(model_spec['variable_init_op'])
        
        save_path = os.path.join(model_save_dir, restore_from)
        if os.path.isdir(save_path):
            save_path = tf.train.latest_checkpoint(save_path) # If restore_from is a directory, get the latest ckpt
        saver.restore(sess, save_path)
        
        num_steps = (params.inp_size + params.batch_size - 1) // params.batch_size
        
        sess.run([iterator_init_op, metrics_init_op])
    
        if len(np.shape(target)) == 1:
            out = np.empty(np.shape(target))[:, np.newaxis]
        else:
            out = np.empty(np.shape(target))
        for i in tqdm(range(num_steps)):
            _, predictions_eval = sess.run([update_metrics_op, predictions])
            if i < num_steps - 1:
                out[i*params.batch_size:(i+1)*params.batch_size, :] = predictions_eval
            else:
                out[i*params.batch_size:, :] = predictions_eval
    
    return out

def _predict_sess(sess, model_spec, num_steps):
    """
    Args:
        sess: (tf.Session) current session
        model_spec: (dict)
        num_stpes: (int)
    """
    iterator_init_op = model_spec['iterator_init_op']
    update_metrics_op = model_spec['update_metrics_op']
    metrics = model_spec['metrics']
    predictions = model_spec['predictions']
    
    sess.run(iterator_init_op)
    
    for _ in range(num_steps):
        _, predictions_eval = sess.run([update_metrics_op, predictions])

        
        
if __name__ == '__main__':
    start_time = time.time()
    
    #for reproducibility
    tf.compat.v1.set_random_seed(123)
    
    args = parser.parse_args()
    params_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(params_path), "params.json does not exits at {}".format(params_path)
    params = Params(params_path)
    params.load(params.update)
    
    #TODO: check and load if there's the best weights so far
#     model_dir_has_best_weights = os.path.isdir(os.path.join(args.model_dir, "best_weights"))
    
    #set logger
    set_logger(os.path.join(args.model_dir, 'train.log'))
    
    #train/test split
    train_fpaths, test_fpaths, train_targets, test_targets = \
        get_train_test_split(args.json_path, args.data_dir, train_size=args.train_size) 
    
    params.train_size = len(train_fpaths)
    params.test_size = len(test_fpaths)
    
    logging.info("Creating the dataset...")
    train_inputs = input_fn(True, train_fpaths, train_targets, params)
    test_inputs = input_fn(False, test_fpaths, test_targets, params)
    
    logging.info("Creating the model...")
    train_model_spec = model_fn(True, train_inputs, params)
    test_model_spec = model_fn(False, test_inputs, params, reuse=True)
    
    logging.info("train set predict...")
    predict(train_model_spec, args.model_save_dir, params, args.restore_from)
    
    logging.info("test set predict...")
    predict(test_model_spec, args.model_save_dir, params, args.restore_from)
    
    end_time = time.time()
    logging.info("Elapsed training time is {:.2f} secs".format(end_time - start_time))