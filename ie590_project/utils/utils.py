import json
import logging
import numpy as np
import pandas as pd

class Params():
    """A class to load hyperparameters from a json file"""
    
    def __init__(self, json_path):
        self.update = json_path
        
        #load parameters onto members
        self.load(self.update)
    
    def save(self, json_path):
        """Saves parameters to json file"""
        with open(json_path, 'w') as jf:
            json.dump(self.__dict__, jf, indent=4)
            
    def load(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as jf:
            params = json.load(jf)
            self.__dict__.update(params)
            
    @property
    def dict(self):
        """give the class dictionary-like access (e.g. params.dict['learning_rate']) """
        return self.__dict__
    
def _train_validate_test_split(df, train_percent=.8, validate_percent=.2, seed=243):
    """Helper function for build_train_test()"""
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.iloc[perm[:train_end]]
    validate = df.iloc[perm[train_end:validate_end]]
    test = df.iloc[perm[validate_end:]]
    return train, validate, test


def build_train_test(main_csv_path, dataset_subset_ratio, seed=243):
    """This function creates the train, validation, and test csv files"""
    """The created csvs are be what should be passed into input_fn() to create the tf.data.dataset object"""
    df = pd.read_csv(main_csv_path)
    #Shrink dataset by the ratio provided
    df = df.sample(frac=dataset_subset_ratio, random_state=seed).reset_index(drop=True)
    train, val, test = _train_validate_test_split(df, train_percent=.8, validate_percent=.2)
    train.to_csv("train.csv", index=False)
    print("Created train.csv in current directory")
    val.to_csv("validation.csv", index=False)
    print("Created validation.csv in current directory")
    test.to_csv("test.csv", index=False)
    print("Created test.csv in current directory")



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
        
        
def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file
    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)
        
        
def get_train_test_split(json_path, data_dir, train_size=None, test_ratio=0.2, seed=123, hash_fn=None): #TODO included:1
    """split a json file into 'raw' train/test sets
    Args:
        json_path: (str) the json file path which stores the image file names and corresponding keypoints' information
        train_size: (int) train size; the number of images in input
        test_ratio: (float) test_size = train_size * test_ratio
        hash_fn: #TODO: hash function for splitting train/test sets
    Returns:
        train_fnames, test_fnames: (np.array) a list of filenames (string)
        train_targets, test_targets: (np.array) a list of keypoints' locations
    """
    #set-ups
    import math
    import random
    import os
    random.seed(seed)
    
    assert os.path.isfile(json_path), "{} does not exist.".format(json_path)
    
    with open(json_path, 'r') as jf:
        raw_data = json.load(jf)
        
    if (train_size is None) or (train_size*(1 + test_ratio) > len(list(raw_data))):
        train_size = int(len(list(raw_data)) * 1 / (1 + test_ratio))
    test_size = math.ceil(test_ratio * train_size)
        
    #train/test split by primary dict keys: date 
    keys = random.sample(list(raw_data), train_size + test_size)
    train_fnames = keys[:train_size]
    test_fnames = keys[train_size:]
    
    #extract train/test targets w.r.t. fnames
    target_map_fn = lambda x: raw_data[x]['annotations']
    train_targets = list(map(target_map_fn, train_fnames))
    test_targets = list(map(target_map_fn, test_fnames))
    
    #clean data
    train_fnames, train_targets = _clean_input(train_fnames, train_targets, raw_data)
    test_fnames, test_targets = _clean_input(test_fnames, test_targets, raw_data)
    
    train_fpaths = _get_fpaths(train_fnames, data_dir)
    test_fpaths = _get_fpaths(test_fnames, data_dir)
    
    return train_fpaths, test_fpaths, train_targets, test_targets

def _clean_input(fnames, targets, raw_data, mode=None): #TODO included:1
    """pre-clean the input data
    Args:
        fnames: (list) a list of strings
        targets: (list) a list of 'annotations' info corresponding to fnames
        raw_data: (dict) raw data from json
        mode: #TODO: the option or function of how to clean the data
    Returns:
        fnames: (np.array) a list of strings
        targets: (np.array) a list of keypoints
    """
    import numpy as np
    
    assert len(fnames) == len(targets), "The length of fnames and targets have to be same."
    
    subset_target_fn = lambda x: x['keypoints'] #get keypoints's locations
    targets = list(map(subset_target_fn, targets))
    
    fnames = np.array(fnames)
    targets = np.array(targets)
    
    return fnames, targets

def _get_fpaths(fnames, data_dir):
    """get full paths for fnames
    Args:
        fnames: (np.array) image file names
        data_dir: (str) relative or absolute path of the data directory
    """
    import numpy as np
    import os
    
    fpaths = np.array([os.path.join(data_dir, x) for x in fnames])
    
    return fpaths

def build_train_validate(path, seed=1234, test_fold=0, model_type='FC'):
    """split full data into train/validation/test (test is unseen labels) and save them as csv files"""
    np.random.seed(seed)
    
    df = pd.read_csv(path)
    
    test = df[df.fold == test_fold]
    df = df[df.fold != test_fold]
    
    mask = np.random.rand(len(df)) < 0.8
    train = df[mask]
    validation = df[~mask]
    
    train.to_csv('train_{}_fold{}.csv'.format(model_type, str(test_fold)), index=False)
    validation.to_csv('validation_{}_fold{}.csv'.format(model_type, str(test_fold)), index=False)
    test.to_csv('test_{}_fold{}.csv'.format(model_type, str(test_fold)), index=False)
    
    # No return
    
def f1_score():
    """calculate F1-scores for descriptor predictions"""
    pass