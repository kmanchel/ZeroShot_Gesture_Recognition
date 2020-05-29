"""include functions to build tf.data.Dataset"""
import os
import numpy as np
import tensorflow as tf
import cv2
import math
import pandas as pd

def _py_fn(fn):
    def apply_tf_pyfunc(*args, **kwargs):
        return tf.py_function(fn, list(args), **kwargs)
    return apply_tf_pyfunc

@_py_fn
def _parse_fn(inp, target): #TODO included
    """parse frames from video
    Args:
        inp: (tf.Tensor) input tensor slice: video names
        target: (tf.Tensor) target tensor slice: a motion descriptor values
        params: (Params) set of parameters. extracts max_frames, height, width.
    Return:
        image: (tf.Tensor) 3D tensor for an combined image(s)
        target: (tf.Tensor) 1D tensor for a binary descriptor value(s)
    """
    #max_frames = params.max_frames
    #height, width = params.height, params.width 
    max_frames = 25
    height, width = 240, 320
    
    rgb_v, d_v = inp.numpy() # RGB and D video paths
    rgb_v, d_v = rgb_v.decode('utf-8'), d_v.decode('utf-8')
    
    out = np.zeros((max_frames, height, width, 6)) # RGBD frames: for some reason, K-video is not 1 channel

    cap_rgb = cv2.VideoCapture(rgb_v)
    cap_d = cv2.VideoCapture(d_v)

    frame_count_rgb = cap_rgb.get(7)
    frame_count_d = cap_d.get(7)
    
    assert frame_count_rgb == frame_count_d, "Total Frames of RGB-video and D-video do not match"
    

    count = 0
    ret = True
    if frame_count_rgb<max_frames:
        #Take all the frames and add zero padding to remaining:
        while ret:
            frameId = cap_rgb.get(1) # get current frame ID

            ret, vals_rgb = cap_rgb.read()
            _, vals_d = cap_d.read()

            out[count,:,:,:3] = vals_rgb
            out[count,:,:,3:] = vals_d
            count += 1
            if count==frame_count_rgb:
                break
    else:
        #Take the middle max_frames
        length_diff = frame_count_rgb-max_frames
        #define range. Start taking frames at difference/2 and end at
        diff = length_diff//2
        while ret:
            frameId = cap_rgb.get(1) # get current frame ID

            ret, vals_rgb = cap_rgb.read()
            _, vals_d = cap_d.read()
            
            if frameId>diff:
                out[count,:,:,:3] = vals_rgb
                out[count,:,:,3:] = vals_d
                count += 1
            if count==max_frames:
                break
                
    cap_rgb.release()
    cap_d.release()
    
    return out, target

@_py_fn
def _load_features(inp,target):
    m_path, k_path = inp.numpy()
    m_features = np.load(m_path)
    k_features = np.load(k_path)
    out = np.zeros((2,m_features.shape[0],m_features.shape[1]))
    out[0,:,:] = m_features
    out[1,:,:] = k_features
    return out,target

def _image_resize(inps, target, params):
    """
    Args:
        inp: (tf.Tensor)
        target: (tf.Tensor)
        params: (Params) Params instance
    """
    IMG_SIZE = params.image_resize
    inps = tf.cast(inps, tf.float32)
    inps = (inps/127.5) - 1
    inps = tf.image.resize(inps, (IMG_SIZE, IMG_SIZE))
    return inps, target

def _preproc_fn(inp, target, params): 
    """
    Args:
        inp: (tf.Tensor)
        target: (tf.Tensor)
        params: (Params) Params instance
    """
    F, H, W = params.max_frames, params.image_height, params.image_width
    
    inp = tf.reshape(inp, (F,H,W,6))
    
    return inp, target

    
def _internal_input_fn(is_training, inps, targets, params):
    """
    Args:
        is_training: (bool) whether it's training or not
        inps: (list)(str) full paths of videos e.g. [data/train/M_00001.avi, data/train/K_00001.avi],...
        targets: (list)(list)(float) list of descriptor vector to be used as training targets 
                e.g. [[0,1,0.3,0.8,0],[0,1,0.3,0.8,0],..] or [[1],[0],[1]]
        params: (Params) Params instance (utils.params)
    Returns:
        inputs: (dict) tf.Dataset ops & iterator op
    """
    assert len(inps) == len(targets), "Input length and target length should be same."
    num_samples = len(inps)
    
    if (is_training) and (len(inps) < params.batch_size): 
        params.batch_size = len(inps)
    
    if params.descriptor_type=='mean':
        parse_fn = lambda inp, target: _parse_fn(inp, target, Tout=[tf.float64, tf.float64])
    else:
        parse_fn = lambda inp, target: _parse_fn(inp, target, Tout=[tf.int64, tf.int64])
    preproc_fn = lambda inp, target: _preproc_fn(inp, target, params)  
    image_resize = lambda inp,target: _image_resize(inp, target, params)
    
    if is_training: 
        dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(inps), tf.constant(targets)))
                   .shuffle(num_samples) 
                   .map(parse_fn)
                   .map(preproc_fn)
                   .map(image_resize)
                   .batch(params.batch_size)
                   .prefetch(1)
                  )
    else:
        dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(inps), tf.constant(targets)))
                   .map(parse_fn)
                   .map(preproc_fn)
                   .map(image_resize)
                   .batch(params.batch_size)
                   .prefetch(1)
                  )
    
    #iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
    #images, targets = iterator.get_next()
    #iter_init_op = iterator.initializer
    
    iterator = iter(dataset)
    images, targets = iterator.next()
    
    inputs = {'images': images, 'targets': targets, 'iterator': iterator}
    
    return inputs

def input_fn(is_training,params):
    """
    Args:
        is_training: (bool) whether it's training or not
        params: (Params) Params instance (utils.params)
    Returns:
        inps: (list)(str) full paths of videos e.g. [data/train/M_00001.avi, data/train/K_00001.avi],...
        targets: (np.array)(list)(float) list of descriptor vector to be used as training targets 
                e.g. [[0,1,0.3,0.8,0],[0,1,0.3,0.8,0],..] or [[1],[0],[1]]
    """    
    df = pd.read_csv(params.csv_path)
    descriptors = params.descriptors.split(',')
    descriptor_ix = [df.columns.get_loc(name) for name in descriptors] #Finding column indexes of descriptors
    inps = []
    targets = []
    for i in range(len(df)):
        m_path = params.dataset_path+df.M.iloc[i]
        k_path = params.dataset_path+df.K.iloc[i]
        inps.append([m_path,k_path])
        descriptor_vector = list(df.iloc[:,descriptor_ix].iloc[i])
        #targets.append(descriptor_vector) TEMPORARY REMOVAL
        targets.append(descriptor_vector[0])
    targets = np.array(targets)
    return(_internal_input_fn(is_training, inps, targets, params))
        



def _internal_input_fn_features(is_training, inps, targets, params):
    """
    Args:
        is_training: (bool) whether it's training or not
        inps: (list)(str) full paths of videos e.g. [data/train/M_00001.avi, data/train/K_00001.avi],...
        targets: (list)(list)(float) list of descriptor vector to be used as training targets 
                e.g. [[0,1,0.3,0.8,0],[0,1,0.3,0.8,0],..] or [[1],[0],[1]]
        params: (Params) Params instance (utils.params)
    Returns:
        inputs: (dict) tf.Dataset ops & iterator op
    """
    assert len(inps) == len(targets), "Input length and target length should be same."
    num_samples = len(inps)
    
    if (is_training) and (len(inps) < params.batch_size): 
        params.batch_size = len(inps)
    
    if params.descriptor_type=='mean':
        load_features = lambda inp, target: _load_features(inp, target, Tout=[tf.float64, tf.float64])
    else:
        load_features = lambda inp, target: _load_features(inp, target, Tout=[tf.float64, tf.int64])
    
    if is_training: 
        dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(inps), tf.constant(targets)))
                   .shuffle(num_samples) 
                   .map(load_features)
                   .batch(params.batch_size)
                   .prefetch(1)
                  )
    else:
        dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(inps), tf.constant(targets)))
                   .map(load_features)
                   .batch(params.batch_size)
                   .prefetch(1)
                  )

    #iterator = iter(dataset)
    #features, targets = iterator.next()
    
    #inputs = {'features': features, 'targets': targets, 'iterator': iterator}
    
    return dataset
    

def input_fn_features(is_training,params,prediction_mode=False):
    """
    Args:
        is_training: (bool) whether it's training or not
        params: (Params) Params instance (utils.params)
    Returns:
        inps: (list)(str) full paths of videos e.g. [data/train/M_00001.avi, data/train/K_00001.avi],...
        targets: (np.array)(list)(float) list of descriptor vector to be used as training targets 
                e.g. [[0,1,0.3,0.8,0],[0,1,0.3,0.8,0],..] or [[1],[0],[1]]
    """    
    if is_training: 
        df = pd.read_csv(params.train_path)
        print("Loading Training Data from:",params.train_path)
    else:
        if prediction_mode==False:
            df = pd.read_csv(params.validation_path)
            print("Loading Validation Data from:",params.validation_path)
        else: 
            df = pd.read_csv(params.prediction_path)
            print("Loading Test Data from:",params.prediction_path)
    descriptors = params.descriptors.split(',')
    descriptor_ix = [df.columns.get_loc(name) for name in descriptors] #Finding column indexes of descriptors
    inps = []
    targets = []
    for i in range(len(df)):
        m_path = df.feature_path_m.iloc[i]
        k_path = df.feature_path_k.iloc[i]
        inps.append([m_path,k_path])
        descriptor_vector = list(df.iloc[:,descriptor_ix].iloc[i])
        #targets.append(descriptor_vector) TEMPORARY REMOVAL
        targets.append(descriptor_vector[0])
    targets = np.array(targets)
    return(_internal_input_fn_features(is_training, inps, targets, params))
        
    
    
