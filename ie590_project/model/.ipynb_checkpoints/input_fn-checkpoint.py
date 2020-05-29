"""include functions to build tf.data.Dataset"""
import os
import numpy as np
import tensorflow as tf
import cv2
import math
import pandas as pd

def _parse_fullpath(paths, base_dir="../data/", is_training=True):
    """parse full path given video name
    Args:
        paths: (str) a list of video directory e.g. train/001/00001.avi, train/001/00002, ...
    Return:
        full_paths: (str) a list of full paths of videos e.g. [data/train/M_00001.avi, data/train/K_00001.avi], ...
    """
    if not isinstance(paths, list):
        paths = [paths]
        
    video_type = lambda path, vtype: os.path.join(base_dir, path.split('/')[:-1], vtype+'_'+path.split('/')[-1])
    full_paths = [[video_type('M'), video_type('K')] for path in paths]
    
    return full_paths

def _py_fn(fn):
    def apply_tf_pyfunc(*args, **kwargs):
        return tf.py_function(fn, list(args), **kwargs)
    return apply_tf_pyfunc

@_py_fn
def _parse_fn(inp, target, max_frames = 40): #TODO included
    """parse frames from video
    Args:
        inp: (tf.Tensor) input tensor slice: video names
        target: (tf.Tensor) target tensor slice: a motion descriptor values
        max_frames: (int) number of frames to limit the sequence to
    Return:
        image: (tf.Tensor) 3D tensor for an combined image(s)
        target: (tf.Tensor) 1D tensor for a binary descriptor value(s)
    """

    height, width = 240, 320 #TODO: get from params instead
    #max_frames = 5 #TODO: get from params instead. REMOVED
    
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
        print("Zero Padding at the end")
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
        
    #while cap_rgb.isOpened():
    #    frameId = cap_rgb.get(1) # get current frame ID
    #    ret, frame_rgb = cap_rgb.read() # read the next one frame
    #    _, frame_d = cap_d.read()
    #    
    #    if ret != True:
    #        break

    #    if (frameId // lag) % math.floor(fps_rgb) == 0: #TODO: seems like there's logical error in the formula
    #        out[count,:,:,:3] = frame_rgb
    #        out[count,:,:,3:] = frame_d
    #        count += 1
            
    #    if count == max_frames:
    #        break
            
    #assert count == max_frames, "Stopped at {}-f, Video length is too short for max_frames {} and lag {}".format(count, max_frames, lag)
    
    cap_rgb.release()
    cap_d.release()
    
    return out, target

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

    #inps = _parse_fullpath(inps)
    
    parse_fn = lambda inp, target: _parse_fn(inp, target, Tout=[tf.int64, tf.int64])
    preproc_fn = lambda inp, target: _preproc_fn(inp, target, params)                                       
    
    if is_training: 
        print("DEBUG.. TRAINING")
        dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(inps), tf.constant(targets)))
                   .shuffle(num_samples) 
                   .map(parse_fn)
                   .map(preproc_fn)
                   .batch(params.batch_size)
                   .prefetch(1)
                  )
    else:
        print("DEBUG.. NOT TRAINING")
        dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(inps), tf.constant(targets)))
                   .map(parse_fn)
                   .map(preproc_fn)
                   .batch(params.batch_size)
                   .prefetch(1)
                  )
    
    iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
    images, targets = iterator.get_next()
    iter_init_op = iterator.initializer
    
    inputs = {'images': images, 'targets': targets, 'iter_init_op': iter_init_op}
    
    return inputs
    
def input_fn(csv_path,data_path,is_training,params,descriptors,vid_ix,labels=None):
    """
    Args:
        csv_path: (str) path of data file which needs to read
        data_path: (str) path to the folder where the data is kept e.g /home/jupyter/CV_Project/ie590_project/data/IsoGD_phase_1/
        is_training: (bool) whether it's training or not
        params: (Params) Params instance (utils.params)
        descriptors: (list)(str) descriptors to retrieve target from 
        vid_ix: (list)(str) video index numbers to retrieve links of. NEEDS TO BE 5 digits!
        labels: (list) subset by specific labels. Default=None. CURRENTLY NOT FUNCTIONAL. WILL BE WORKED ON IF NEEDED.
    Returns:
        inps: (list)(str) full paths of videos e.g. [data/train/M_00001.avi, data/train/K_00001.avi],...
        targets: (list)(list)(float) list of descriptor vector to be used as training targets 
                e.g. [[0,1,0.3,0.8,0],[0,1,0.3,0.8,0],..] or [[1],[0],[1]]
    """    
    df = pd.read_csv(csv_path)
    descriptor_ix = [df.columns.get_loc(name) for name in descriptors]
    inps = []
    targets = []
    for i in vid_ix:
        #Checking Length of vid_ix
        assert len(i)==5, "Length of Video Index should be EXACTLY 5 characters long!"
        #Appending Input paths
        try:
            m_path = df.M[df.M.str.contains(i)][1]
            k_path = df.K[df.K.str.contains(i)][1]
        except:
            m_path = df.M[df.M.str.contains(i)][0]
            k_path = df.K[df.K.str.contains(i)][0]
        m_path = data_path+m_path
        k_path = data_path+k_path
        print("Debug", m_path)
        inps.append([m_path,k_path])
        #Appending appropriate target descriptor vector
        descriptor_vector = df.iloc[:,descriptor_ix][df.M.str.contains(i)].values[0]
        targets.append(descriptor_vector) 
    targets = np.array(targets)
    return(_internal_input_fn(is_training, inps, targets, params))
        
        
    
    
    
    
