from __future__ import print_function

import argparse
import os
import pickle
import json
import math
import shutil
import subprocess
import numpy as np
import torch
from scipy import misc
import time
import cv2
from tqdm import tqdm

from src.i3dpt import I3D

FPS = 25
MAX_INTERVAL = 64
OVERLAP = 8
rgb_pt_checkpoint = 'model/model_rgb.pth'

def extract_frames(args, target_height=224, target_width=224):
    dir = os.path.join(args.video_dir, "MSR-VTT/Videos")
    video_list = os.listdir(dir)
    for video in tqdm(video_list):
        name_clip = video.split('.')[0]
        path_clip = os.path.join(dir, video)
        frame_path = os.path.join(args.video_dir, 'MSR-VTT_frames')
        frame_path = os.path.join(frame_path, name_clip)
        with open(os.devnull, "w") as ffmpeg_log:
            if os.path.exists(frame_path):
                print(" cleanup: " + frame_path + "/")
                shutil.rmtree(frame_path)

            os.makedirs(frame_path)
            video_to_frames_command = ["ffmpeg",
                                       '-y',  # (optional) overwrite output file if it exists
                                       '-i', path_clip,  # input file
                                       '-r', '25',
                                       '-qscale:v', "2",  # quality for JPEG
                                       '{0}/%06d.jpg'.format(frame_path)]
            subprocess.call(video_to_frames_command, stdout=ffmpeg_log, stderr=ffmpeg_log)

        # resize and crop images
        images = os.listdir(frame_path)
        for image in images:
            image_path = os.path.join(frame_path, image)
            image = cv2.imread(image_path)
            height, width, channels = image.shape
            if height == width:
                resized_image = cv2.resize(image, (target_height, target_width))
            elif height < width:
                resized_image = cv2.resize(image, (int(width * target_height / height), target_height))
                # print(resized_image.shape)
                cropping_length = int((resized_image.shape[1] - target_width) / 2)
                # print(cropping_length)
                resized_image = resized_image[:, cropping_length: target_width + cropping_length]
                # print(resized_image.shape)
            else:
                resized_image = cv2.resize(image, (target_width, int(height * target_width / width)))
                cropping_length = int((resized_image.shape[0] - target_height) / 2)
                resized_image = resized_image[cropping_length: target_height + cropping_length, :]
            cv2.imwrite(image_path, resized_image)

def get_features(sample, model):
    sample = sample.transpose(0, 4, 1, 2, 3)
    with torch.no_grad():
        sample_var = torch.autograd.Variable(torch.from_numpy(sample).cuda())
        # sample_var = torch.autograd.Variable(torch.from_numpy(sample))
    out_var = model.extract(sample_var)
    out_tensor = out_var.data.cpu()
    return out_tensor.numpy()


def read_video(video_dir):
    # start = time.time()
    frames = [f for f in os.listdir(video_dir) if os.path.isfile(os.path.join(video_dir, f))]
    data = []
    for i, frame in enumerate(sorted(frames)):
        I = misc.imread(os.path.join(video_dir, frame), mode='RGB')
        if len(I.shape) == 2:
            I = I[:, :, np.newaxis]
            I = np.concatenate((I, I, I), axis=2)
        I = (I.astype('float32') / 255.0 - 0.5) * 2
        data.append(I)
    if len(data) <= 0:
        return None
    res = np.asarray(data)[np.newaxis, :, :, :, :]
    # print("load time: ", time.time() - start)
    return res


def run(args):
    # Run RGB model
    i3d_rgb = I3D(num_classes=400, modality='rgb')
    i3d_rgb.eval()
    i3d_rgb.load_state_dict(torch.load(args.rgb_weights_path))
    i3d_rgb.cuda()

    # read the video list which records the readable videos
    dir = os.path.join(args.video_dir, "MSR-VTT_frames")

    video_list = os.listdir(dir)
    # print(video_list)

    for vid in tqdm(video_list):
        video = os.path.join(dir, vid)
        clip = read_video(video)
        if clip is None:
            continue
        clip_len = clip.shape[1]
        if clip_len <= MAX_INTERVAL:
            features = get_features(clip, i3d_rgb)
        else:
            tmp_1 = 0
            features = []
            while True:
                tmp_2 = tmp_1 + MAX_INTERVAL
                tmp_2 = min(tmp_2, clip_len)
                feat = get_features(clip[:, tmp_1:tmp_2], i3d_rgb)
                features.append(feat)
                if tmp_2 == clip_len:
                    break
                tmp_1 = max(0, tmp_2 - OVERLAP)
            features = np.concatenate(features, axis=1)
            # print(features.shape)
        np.save(os.path.join(args.out_dir, vid), features)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Runs inflated inception v1 network on\
    cricket sample from tensorflow demo (generate the network weights with\
    i3d_tf_to_pt.py first)')

    # RGB arguments
    parser.add_argument(
        '--rgb', action='store_true', help='Evaluate RGB pretrained network')
    parser.add_argument(
        '--rgb_weights_path',
        type=str,
        default='model/model_rgb.pth',
        help='Path to rgb model state_dict')
    parser.add_argument(
        '--rgb_sample_path',
        type=str,
        default='data/kinetic-samples/v_CricketShot_g04_c01_rgb.npy',
        help='Path to kinetics rgb numpy sample')

    # parser.add_argument('--id_list', type=str)
    parser.add_argument('--video_dir', type=str, default='../DATA')
    parser.add_argument('--out_dir', type=str, default="../DATA/MSRVTT_I3D_features")
    parser.add_argument('--split', type=str, default="train_val", help="train_val | test")
    args = parser.parse_args()
    extract_frames(args)
    run(args)
