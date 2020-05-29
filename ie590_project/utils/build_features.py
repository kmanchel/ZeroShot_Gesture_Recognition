"""Feature Extractor"""
import sys

sys.path.append("..")
sys.path

import tensorflow as tf

import os
import numpy as np
import cv2
import pandas as pd
from utils import Params

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--USE_GPU", default='False', choices=['True','False'])
args = parser.parse_args()


def _get_frames(inp):
    """parse frames from video
        Args:
            inp: (tf.Tensor) input tensor slice: video names
        Return:
            image: (tf.Tensor) 3D tensor for an combined image(s)
        """
    # max_frames = params.max_frames
    # height, width = params.height, params.width
    max_frames = 25
    height, width = 240, 320

    rgb_v = inp  # RGB and D video paths
    # rgb_v = rgb_v.decode('utf-8')

    out = np.zeros(
        (max_frames, height, width, 3)
    )  # RGBD frames: for some reason, K-video is not 1 channel

    cap_rgb = cv2.VideoCapture(rgb_v)

    frame_count_rgb = cap_rgb.get(7)

    def __sample_frames(array, numElems=max_frames):
        out = array[
            np.round(np.linspace(0, len(array) - 1, numElems)).astype(int)
        ]
        return out

    count = 0
    ret = True
    if frame_count_rgb < max_frames:
        # Take all the frames and add zero padding to remaining:
        while ret:
            frameId = cap_rgb.get(1)  # get current frame ID

            ret, vals_rgb = cap_rgb.read()

            out[count, :, :, :3] = vals_rgb
            count += 1
            if count == frame_count_rgb:
                break
    else:
        # Sample frames
        sampled_frames = __sample_frames(np.arange(frame_count_rgb), max_frames)
        # define range. Start taking frames at difference/2 and end at
        while ret:
            frameId = cap_rgb.get(1)  # get current frame ID
            ret, vals_rgb = cap_rgb.read()
            if frameId in sampled_frames:
                out[count, :, :, :3] = vals_rgb
                count += 1

            if count == max_frames:
                break

    cap_rgb.release()

    return out


def _image_resize(image, IMG_SIZE):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image


class VGG16:
    def __init__(self, params, USE_GPU=False):
        self.path = params.features_path
        if USE_GPU:
            self.device = "/device:GPU:0"
        else:
            self.device = "/cpu:0"
        print("Using device: ", self.device)
        self.params = params
        self.is_training = True
        self.IMG_SIZE = params.image_resize
        with tf.device(self.device):
            self.base_model = tf.keras.applications.VGG16(
                input_shape=(self.IMG_SIZE, self.IMG_SIZE, 3),
                include_top=False,
                weights="imagenet",
            )
            self.base_model.trainable = False

            self.global_avg_pooling = tf.keras.layers.GlobalAveragePooling2D()

    def parse_video_paths(self):
        """
        Args:
            is_training: (bool) whether it's training or not
            params: (Params) Params instance (utils.params)
        Returns:
            m: (str) m-path
            k: (str) k-path
        """
        self.df = pd.read_csv(self.params.csv_path)
        self.m = []
        self.k = []
        for i in range(len(self.df)):
            m_path = params.dataset_path + self.df.M.iloc[i]
            k_path = params.dataset_path + self.df.K.iloc[i]
            self.m.append(m_path)
            self.k.append(k_path)

    def extract_features(self):
        print("\nProcessing %d videos" % len(self.m))
        self.m_names = [file_name[-11:-4] for file_name in self.m]
        self.k_names = [file_name[-11:-4] for file_name in self.k]

        for video in range(len(self.m)):
            f = self.m_names[video]
            f_name_features = self.path + f
            print("Working on:", self.m[video])
            im_original = np.zeros((1, 25, 240, 320, 3))
            im_original[0, :, :, :] = _get_frames(self.m[video])
            im_resized = np.zeros((1, 25, self.IMG_SIZE, self.IMG_SIZE, 3))
            im_resized[0, :, :, :] = _image_resize(im_original[0, :, :, :],self.params.image_resize)
            feature_maps = np.zeros((25, 512))
            for image in range(25):
                with tf.device(self.device):
                    feature_batch = self.base_model(im_resized[:1, image, :, :, :])
                    feature_batch_average = self.global_avg_pooling(feature_batch)
                feature_maps[image] = feature_batch_average.numpy()[0]
            print("Feature map saved: '%s.npy'" % f_name_features)
            np.save(f_name_features, feature_maps)
            self.m_names[video] = f_name_features+".npy"

        for video in range(len(self.k)):
            f = self.k_names[video]
            f_name_features = self.path + f
            print("Working on:", self.k[video])
            im_original = np.zeros((1, 25, 240, 320, 3))
            im_original[0, :, :, :] = _get_frames(self.k[video])
            im_resized = np.zeros((1, 25, self.IMG_SIZE, self.IMG_SIZE, 3))
            im_resized[0, :, :, :] = _image_resize(im_original[0, :, :, :],self.params.image_resize)
            feature_maps = np.zeros((25, 512))
            for image in range(25):
                with tf.device(self.device):
                    feature_batch = self.base_model(im_resized[:1, image, :, :, :])
                    feature_batch_average = self.global_avg_pooling(feature_batch)
                feature_maps[image] = feature_batch_average.numpy()[0]
            print("Feature map saved: '%s.npy'" % f_name_features)
            np.save(f_name_features, feature_maps)
            self.k_names[video] = f_name_features+".npy"

    def add_filenames_toCSV(self):
        assert len(self.m_names) == len(self.df)
        assert len(self.k_names) == len(self.df)
        self.df["feature_path_m"] = self.m_names
        self.df["feature_path_k"] = self.k_names
        self.df.to_csv(self.params.csv_path, index=False)


if __name__ == "__main__":
    params = Params("feature_params.json")
    if args.USE_GPU=='False':
        print("USING CPU")
        USE_GPU = False
    else:
        print("USING GPU")
        USE_GPU = True
    extractor = VGG16(
        params,
        USE_GPU
    )
    extractor.parse_video_paths()
    extractor.extract_features()
    extractor.add_filenames_toCSV()
