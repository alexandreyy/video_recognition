# Copyright 2018-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from caffe2.python import workspace, cnn, core, data_parallel_model
import models.model_builder as model_builder
import utils.model_helper as model_helper
import utils.model_loader as model_loader

import numpy as np
import json
import logging
import argparse
import os.path
import pickle
import sys
import cv2

from caffe2.proto import caffe2_pb2

os.environ['QT_X11_NO_MITSHM'] = '1'


logging.basicConfig()
log = logging.getLogger("feature_extractor")
log.setLevel(logging.INFO)

# Output logs to stdout as well, as they get lost in the ffmpeg read errors
stdout_ch = logging.StreamHandler(sys.stdout)
stdout_ch.setLevel(logging.INFO)
log.addHandler(stdout_ch)


def pre_process(frames, width, height):
    stacked_frames = np.empty(
        (1, 3, len(frames), height, width)).astype('float32')
    
    for i in range(len(frames)):
        stacked_frames[:, 0, i, :, :] = (frames[i][:, :, 0] - 110.201) / 58.1489
        stacked_frames[:, 1, i, :, :] = (frames[i][:, :, 1] - 100.64) / 56.4701
        stacked_frames[:, 2, i, :, :] = (frames[i][:, :, 2] - 95.9966) / 55.3324
    return stacked_frames

def put_in_shape(image, resize_to, crop_to):
    resized = cv2.resize(image, (resize_to[1], resize_to[0]))
    height, width, _ = resized.shape
    h_off = int((height - crop_to[0]) / 2)
    w_off = int((width - crop_to[1]) / 2)
    cropped = resized[h_off:h_off+crop_to[0], w_off:w_off+crop_to[1]]
    return cropped

def fetch_activations(model, outputs):

    all_activations = {}
    workspace.RunNet(model.net.Proto().name)

    for output_name in outputs:
        blob_name = 'gpu_{}/'.format(0) + output_name
        activations = workspace.FetchBlob(blob_name)
        if output_name not in all_activations:
            all_activations[output_name] = []
        all_activations[output_name].append(activations)

    # each key holds a list of activations obtained from each minibatch.
    # we now concatenate these lists to get the final arrays.
    # concatenating during the loop requires a realloc and can get slow.
    for key in all_activations:
        all_activations[key] = np.concatenate(all_activations[key])

    return all_activations

def create_model_ops(model, loss_scale, args):
    return model_builder.build_model(
        model=model,
        model_name=args.model_name,
        model_depth=args.model_depth,
        num_labels=args.num_labels,
        num_channels=args.num_channels,
        crop_size=args.crop_size,
        clip_length=(
            args.clip_length_of if args.input_type == 1
            else args.clip_length_rgb
        ),
        loss_scale=loss_scale,
        is_test=1,
    )

def put_text_on_image(image, values):
    y0, dy = 25, 25
    text_size = cv2.getTextSize(values[0], cv2.FONT_HERSHEY_SIMPLEX, 1.1e-3 * image.shape[0], 1)

    cv2.rectangle(img=image, pt1=(20, 20), pt2=(text_size[0][0]+35, text_size[0][1]+25), color=(0,0,0),
                  thickness=-1)
                  
    cv2.putText(image, str("{0}.{1}".format(1,values[0])), (25, 30 ), cv2.FONT_HERSHEY_SIMPLEX, 1e-3 * image.shape[0], (255,255,255),1,cv2.LINE_AA)

    for i,value in enumerate(values[1:]):
        y = y0 + (i+1)*dy
        cv2.putText(image, str("{0}.{1}".format(i+2,value)), (25, y ), cv2.FONT_HERSHEY_SIMPLEX, 1e-3 * image.shape[0], (255,255,255),1,cv2.LINE_AA)

def run_inference(args):

    if args.gpus is not None:
        gpus = [int(x) for x in args.gpus.split(', ')]
        num_gpus = len(gpus)
    else:
        gpus = range(args.num_gpus)
        num_gpus = args.num_gpus

    my_arg_scope = {
        'order': 'NCHW',
        'use_cudnn': True,
        'cudnn_exhaustive_search': True
    }

    model = cnn.CNNModelHelper(
        name="Extract Features",
        **my_arg_scope
    )
    
    # gpu?
    if num_gpus > 0:
        log.info("Running on GPUs: {}".format(gpus))
        model._device_type = caffe2_pb2.CUDA
        model._cuda_gpu_id = 0
        model._devices = [0]

    # cpu
    else:
        log.info("Running on CPU")
        model._device_type = caffe2_pb2.CPU
        model._devices = [0]

    # create the scope 
    device_opt = core.DeviceOption(model._device_type, 0)
    with core.DeviceScope(device_opt):
        with core.NameScope("{}_{}".format("gpu", 0)):
            create_model_ops(model, 1.0, args)

    # gather parameters
    batch = 1
    channels_rgb = args.num_channels
    frames_per_clip = args.clip_length_rgb
    crop_size = args.crop_size
    width = args.scale_w
    height = args.scale_h
    input_video = args.input

    # configuration for the input
    #data = np.empty((1, channels_rgb, frames_per_clip, crop_size, crop_size))
    #label = np.empty((1, 1))

    # initialize the network
    workspace.CreateBlob("gpu_0/data")
    workspace.CreateBlob("gpu_0/label")
    workspace.RunNetOnce(model.param_init_net)
    workspace.CreateNet(model.net)

    if args.db_type == 'minidb':
        with core.DeviceScope(core.DeviceOption(caffe2_pb2.CPU, 0)):
            model_helper.LoadModel(args.load_model_path, args.db_type)
    elif args.db_type == 'pickle':
        model_loader.LoadModelFromPickleFile(
            model,
            args.load_model_path,
            use_gpu=False,
        )
    else:
        log.warning("Unsupported db_type: {}".format(args.db_type))

    outputs = [name.strip() for name in args.features.split(', ')]
    assert len(outputs) > 0

    input_video = cv2.VideoCapture(input_video)

    with open(args.labels) as f:
        matching_labels = np.array(json.load(f))

    clip_list = []
    label = np.empty((1)).astype('int32')

    # create windows for opencv
    cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('frame', 800,600)
    cv2.namedWindow('processed',cv2.WINDOW_NORMAL)
    cv2.moveWindow('processed',800,0)

    while True:
        # get a frame from the video
        video_available, frame = input_video.read()

        if not video_available:
            break

        pre_processed_frame = put_in_shape(frame, resize_to=(
            width, height), crop_to=(crop_size, crop_size))
        clip_list.append(pre_processed_frame)

        if len(clip_list) != frames_per_clip:
            continue

        print('sending one set of images to the network!')

        # put the list of frames in the shape for the network
        input_clip = pre_process(clip_list, crop_size, crop_size)

        # remove the first frame
        del clip_list[0]

        # send the data to the network
        workspace.FeedBlob("gpu_0/data", input_clip)
        workspace.FeedBlob("gpu_0/label", label)

        # fetch the outputs
        activations = fetch_activations(model, outputs)

        # get the score for each class
        softmax = activations['softmax']

        cv2.imshow('frame', frame)
        cv2.imshow('processed', pre_processed_frame)

        for i in range(len(softmax)):
            sorted_preds = \
                np.argsort(softmax[i])
            sorted_preds[:] = sorted_preds[::-1]
            
            put_text_on_image(frame, matching_labels[sorted_preds[0:5]])
            cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(
        description="Simple inference example"
    )
    parser.add_argument("--db_type", type=str, default='pickle',
                        help="Db type of the testing model")
    parser.add_argument("--model_name", type=str, default='r2plus1d',
                        help="Model name")
    parser.add_argument("--model_depth", type=int, default=18,
                        help="Model depth")
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma separated list of GPU devices to use")
    parser.add_argument("--num_gpus", type=int, default=1,
                        help="Number of GPU devices (instead of --gpus)")
    parser.add_argument("--scale_h", type=int, default=128,
                        help="Scale image height to")
    parser.add_argument("--scale_w", type=int, default=171,
                        help="Scale image width to")
    parser.add_argument("--crop_size", type=int, default=112,
                        help="Input image size (to crop to)")
    parser.add_argument("--clip_length_rgb", type=int, default=4,
                        help="Length of input clips")
    parser.add_argument("--sampling_rate_rgb", type=int, default=1,
                        help="Frame sampling rate")
    parser.add_argument("--num_labels", type=int, default=400,
                        help="Number of labels")
    parser.add_argument("--labels", type=str, default='/home/code/src/labels/kinetics.json',
                        help="Path to the labels to be displayed")
    parser.add_argument("--num_channels", type=int, default=3,
                        help="Number of channels")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size, per-GPU")
    parser.add_argument("--load_model_path", type=str, default='',
                        required=True,
                        help="Load saved model for testing")
    parser.add_argument("--use_cudnn", type=int, default=1,
                        help="Use CuDNN")
    parser.add_argument("--features", type=str, default="final_avg",
                        help="Comma-separated list of blob names to fetch")
    parser.add_argument("--clip_length_of", type=int, default=8,
                        help="Frames of optical flow data")
    parser.add_argument("--sampling_rate_of", type=int, default=2,
                        help="Sampling rate for optial flows")
    parser.add_argument("--frame_gap_of", type=int, default=2,
                        help="Frame gap of optical flows")
    parser.add_argument("--input_type", type=int, default=0,
                        help="0=rgb, 1=optical flow")
    parser.add_argument("--flow_data_type", type=int, default=0,
                        help="0=Flow2C, 1=Flow3C, 2=FlowWithGray, " +
                        "3=FlowWithRGB")
    parser.add_argument("--do_flow_aggregation", type=int, default=0,
                        help="whether to aggregate optical flow across " +
                        "multiple frames")
    parser.add_argument("--clip_per_video", type=int, default=1,
                        help="When clips_per_video > 1, sample this many " +
                        "clips uniformly in time")
    parser.add_argument("--input", type=str, default='/dev/video0',
                        help="Input video to be processed")


    args = parser.parse_args()
    log.info(args)

    run_inference(args)

if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', ' -- caffe2_log_level = 2'])
    main()

