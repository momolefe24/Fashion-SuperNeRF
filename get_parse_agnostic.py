import json
import sys
from os import path as osp
import os

import numpy as np
from PIL import Image, ImageDraw

import argparse

from tqdm import tqdm


def get_im_parse_agnostic(im_parse, pose_data, w=768, h=1024):
    parse_array = np.array(im_parse)
    parse_upper = ((parse_array == 5).astype(np.float32) +
                    (parse_array == 6).astype(np.float32) +
                    (parse_array == 7).astype(np.float32))
    parse_neck = (parse_array == 10).astype(np.float32)

    r = 10
    agnostic = im_parse.copy()

    # mask arms
    for parse_id, pose_ids in [(14, [2, 5, 6, 7]), (15, [5, 2, 3, 4])]:
        mask_arm = Image.new('L', (w, h), 'black')
        mask_arm_draw = ImageDraw.Draw(mask_arm)
        i_prev = pose_ids[0]
        for i in pose_ids[1:]:
            if (pose_data[i_prev, 0] == 0.0 and pose_data[i_prev, 1] == 0.0) or (pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
                continue
            mask_arm_draw.line([tuple(pose_data[j]) for j in [i_prev, i]], 'white', width=r*10)
            pointx, pointy = pose_data[i]
            radius = r*4 if i == pose_ids[-1] else r*15
            mask_arm_draw.ellipse((pointx-radius, pointy-radius, pointx+radius, pointy+radius), 'white', 'white')
            i_prev = i
        parse_arm = (np.array(mask_arm) / 255) * (parse_array == parse_id).astype(np.float32)
        agnostic.paste(0, None, Image.fromarray(np.uint8(parse_arm * 255), 'L'))

    # mask torso & neck
    agnostic.paste(0, None, Image.fromarray(np.uint8(parse_upper * 255), 'L'))
    agnostic.paste(0, None, Image.fromarray(np.uint8(parse_neck * 255), 'L'))

    return agnostic

# --data_path data/rail/temp --person julian --clothing gray_long_sleeve --type cihp --output_path image-parse-agnostic
def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="CIHP_PGN")
    parser.add_argument("--root_dir", default="data/rail/temp")
    parser.add_argument("--person", default="julian")
    parser.add_argument("--clothing", default="gray_long_sleeve")
    parser.add_argument("--filename", default="julian_gray_long_sleeve_27.jpg")
    opt = parser.parse_args()
    return opt

# --root_dir data/rail/temp --person julian --clothing gray_long_sleeve --filename julian_gray_long_sleeve_27.jpg

opt = get_opt()
person = f"{opt.person}_{opt.clothing}"
path = f"{opt.root_dir}/{opt.person}_{opt.clothing}/image" # directory
output_path = f"{opt.root_dir}/{opt.person}_{opt.clothing}/image-parse-agnostic" # directory
image_file = f"{path}/{opt.filename}" # Existing filename
output_filename = f"{output_path}/{opt.filename}".replace(".jpg",".png")
print(opt)
print('path: ', path)
print('Person: ', person)
print('image_file: ', image_file)
print('output image_file: ', output_filename)
if os.path.isfile(image_file):
    print('Exists!')



    # load pose image
pose_name = image_file.replace('image', 'openpose_json').replace('.jpg', '_keypoints.json')
try:
    with open(pose_name, 'r') as f:
        pose_label = json.load(f)
        pose_data = pose_label['people'][0]['pose_keypoints_2d']
        pose_data = np.array(pose_data)
        pose_data = pose_data.reshape((-1, 3))[:, :2]
except IndexError:
    print(pose_name)
    sys.exit()

parse_name = image_file.replace('image','cihp').replace('.jpg', '.png')
im_parse = Image.open(parse_name)
agnostic = get_im_parse_agnostic(im_parse, pose_data)
agnostic.save(output_filename)
# try:
#     with open(osp.join(data_path, 'openpose_json', pose_name), 'r') as f:
#         pose_label = json.load(f)
#         pose_data = pose_label['people'][0]['pose_keypoints_2d']
#         pose_data = np.array(pose_data)
#         pose_data = pose_data.reshape((-1, 3))[:, :2]
# except IndexError:
#     print(pose_name)
#     continue
#
# # load parsing image
# parse_name = im_name.replace('.jpg', '.png')
# im_parse = Image.open(osp.join(data_path, 'image-parse-v3', parse_name))
#
# agnostic = get_im_parse_agnostic(im_parse, pose_data)
#
# agnostic.save(osp.join(output_path, parse_name))
