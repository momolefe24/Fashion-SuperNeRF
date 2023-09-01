#coding=utf-8
import sys
import torch
import torch.utils.data as data
import os
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
from glob import glob
import os.path as osp
import numpy as np
import json
from utils import get_transforms_data, get_transform_matrix
class FashionNeRFDatasetTest(data.Dataset):
    """
        Test Dataset for CP-VTON.
    """

    def __init__(self, opt):
        super(FashionNeRFDatasetTest, self).__init__()
        # base setting
        self.opt = opt
        self.root = opt.dataroot
        self.cihp = "./cihp_pgn.sh"
        self.detectron = "./densepose.sh"
        self.openpose = "./openpose.sh"
        self.parse_agnostic = "./parse_agnostic.sh"
        self.person = opt.person
        self.clothing = opt.clothing
        self.in_shop_clothing = opt.in_shop_clothing
        self.person_clothing = f"{self.person}_{self.clothing}"
        self.datamode = opt.datamode  # train or test or self-defined
        self.fine_height = opt.fine_height
        self.fine_width = opt.fine_width
        self.semantic_nc = opt.semantic_nc
        self.data_path = osp.join(opt.dataroot, self.person_clothing)
        self.transforms_data = get_transforms_data(self.data_path, self.person_clothing)
        self.camera_angle_x = self.transforms_data['camera_angle_x']
        self.camera_angle_y = self.transforms_data['camera_angle_y']
        self.fl_x = self.transforms_data['fl_x']
        self.fl_y = self.transforms_data['fl_y']
        self.k1 = self.transforms_data['k1']
        self.k2 = self.transforms_data['k2']
        self.k3 = self.transforms_data['k3']
        self.k4 = self.transforms_data['k4']
        self.p1 = self.transforms_data['p1']
        self.p2 = self.transforms_data['p2']
        self.is_fisheye = self.transforms_data['is_fisheye']
        self.cx = self.transforms_data['cx']
        self.cy = self.transforms_data['cy']
        self.W = self.transforms_data['w']
        self.H = self.transforms_data['h']
        self.aabb_scale = self.transforms_data['aabb_scale']
        self.transform = transforms.Compose([ \
            transforms.ToTensor(), \
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        # load data list
        im_names = []
        c_names = []
        transform_matrices =[]
        data_items = glob(f"{self.data_path}/image/*")
        # self.data_items_ = glob(osp.join(self.data_path, person))
        for data_item in data_items:
            data_string = data_item.split("/")[-1]
            im_names.append(data_string)
            frames = self.transforms_data['frames']
            frame = next(filter(lambda file_path: data_string in (file_path.get('file_path')), frames), None)
            transform_matrices.append(frame['transform_matrix'])
        self.im_names = im_names
        self.c_names = dict()
        self.c_names['paired'] = im_names
        self.c_names['unpaired'] = [self.in_shop_clothing] * len(self.im_names)
        self.transform_matrices = transform_matrices

    def name(self):
        return "FashionNeRFData"

    def get_agnostic(self, im, im_parse, pose_data):
        parse_array = np.array(im_parse)
        parse_head = ((parse_array == 4).astype(np.float32) +
                      (parse_array == 13).astype(np.float32))
        parse_lower = ((parse_array == 9).astype(np.float32) +
                       (parse_array == 12).astype(np.float32) +
                       (parse_array == 16).astype(np.float32) +
                       (parse_array == 17).astype(np.float32) +
                       (parse_array == 18).astype(np.float32) +
                       (parse_array == 19).astype(np.float32))

        agnostic = im.copy()
        agnostic_draw = ImageDraw.Draw(agnostic)

        length_a = np.linalg.norm(pose_data[5] - pose_data[2])
        length_b = np.linalg.norm(pose_data[12] - pose_data[9])
        point = (pose_data[9] + pose_data[12]) / 2
        pose_data[9] = point + (pose_data[9] - point) / length_b * length_a
        pose_data[12] = point + (pose_data[12] - point) / length_b * length_a

        r = int(length_a / 16) + 1

        # mask torso
        for i in [9, 12]:
            pointx, pointy = pose_data[i]
            agnostic_draw.ellipse((pointx - r * 3, pointy - r * 6, pointx + r * 3, pointy + r * 6), 'gray', 'gray')
        agnostic_draw.line([tuple(pose_data[i]) for i in [2, 9]], 'gray', width=r * 6)
        agnostic_draw.line([tuple(pose_data[i]) for i in [5, 12]], 'gray', width=r * 6)
        agnostic_draw.line([tuple(pose_data[i]) for i in [9, 12]], 'gray', width=r * 12)
        agnostic_draw.polygon([tuple(pose_data[i]) for i in [2, 5, 12, 9]], 'gray', 'gray')

        # mask neck
        pointx, pointy = pose_data[1]
        agnostic_draw.rectangle((pointx - r * 5, pointy - r * 9, pointx + r * 5, pointy), 'gray', 'gray')

        # mask arms
        agnostic_draw.line([tuple(pose_data[i]) for i in [2, 5]], 'gray', width=r * 12)
        for i in [2, 5]:
            pointx, pointy = pose_data[i]
            agnostic_draw.ellipse((pointx - r * 5, pointy - r * 6, pointx + r * 5, pointy + r * 6), 'gray', 'gray')
        for i in [3, 4, 6, 7]:
            if (pose_data[i - 1, 0] == 0.0 and pose_data[i - 1, 1] == 0.0) or (
                    pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
                continue
            agnostic_draw.line([tuple(pose_data[j]) for j in [i - 1, i]], 'gray', width=r * 10)
            pointx, pointy = pose_data[i]
            agnostic_draw.ellipse((pointx - r * 5, pointy - r * 5, pointx + r * 5, pointy + r * 5), 'gray', 'gray')

        for parse_id, pose_ids in [(14, [5, 6, 7]), (15, [2, 3, 4])]:
            # mask_arm = Image.new('L', (self.fine_width, self.fine_height), 'white')
            mask_arm = Image.new('L', (768, 1024), 'white')
            mask_arm_draw = ImageDraw.Draw(mask_arm)
            pointx, pointy = pose_data[pose_ids[0]]
            mask_arm_draw.ellipse((pointx - r * 5, pointy - r * 6, pointx + r * 5, pointy + r * 6), 'black', 'black')
            for i in pose_ids[1:]:
                if (pose_data[i - 1, 0] == 0.0 and pose_data[i - 1, 1] == 0.0) or (
                        pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
                    continue
                mask_arm_draw.line([tuple(pose_data[j]) for j in [i - 1, i]], 'black', width=r * 10)
                pointx, pointy = pose_data[i]
                if i != pose_ids[-1]:
                    mask_arm_draw.ellipse((pointx - r * 5, pointy - r * 5, pointx + r * 5, pointy + r * 5), 'black',
                                          'black')
            mask_arm_draw.ellipse((pointx - r * 4, pointy - r * 4, pointx + r * 4, pointy + r * 4), 'black', 'black')

            parse_arm = (np.array(mask_arm) / 255) * (parse_array == parse_id).astype(np.float32)
            agnostic.paste(im, None, Image.fromarray(np.uint8(parse_arm * 255), 'L'))

        agnostic.paste(im, None, Image.fromarray(np.uint8(parse_head * 255), 'L'))
        agnostic.paste(im, None, Image.fromarray(np.uint8(parse_lower * 255), 'L'))
        return agnostic
    def __getitem__(self, index):
        im_name = self.im_names[index]
        im_name = "image/" + im_name
        transform_matrix = np.array(self.transform_matrices[index])
        c_name = {}
        c = {}
        cm = {}
        for key in self.c_names:
            c_name[key] = self.c_names[key][index]
            c[key] = Image.open(osp.join(self.data_path, 'cloth', c_name[key])).convert('RGB')
            c[key] = transforms.Resize(self.fine_width, interpolation=2)(c[key])
            cm[key] = Image.open(osp.join(self.data_path, 'cloth-mask', c_name[key]))
            cm[key] = transforms.Resize(self.fine_width, interpolation=0)(cm[key])

            c[key] = self.transform(c[key])  # [-1,1]
            cm_array = np.array(cm[key])
            cm_array = (cm_array >= 128).astype(np.float32)
            cm[key] = torch.from_numpy(cm_array)  # [0,1]
            cm[key].unsqueeze_(0)

        # person image
        filename = im_name.split("/")[-1]
        im_pil_big = Image.open(osp.join(self.data_path, im_name))
        im_pil = transforms.Resize(self.fine_width, interpolation=2)(im_pil_big)
        im = self.transform(im_pil.convert('RGB'))

        # load parsing image
        self.cihp += f" {self.root} {self.person} {self.clothing} {filename}"
        cihp_err = os.system(self.cihp)

        if 1 == 0:
            print("FATAL: CIHP command failed")
            sys.exit(1)
            # sys.exit(cihp_err)
        else:
            cihp_name = im_name.replace('image', 'cihp').replace('.jpg', '.png')
            if os.path.isfile(osp.join(self.data_path, cihp_name)):
                # parse_name = im_name.replace('image', 'cihp').replace('.jpg', '.png')  # VITON
                im_parse_pil_big = Image.open(osp.join(self.data_path, cihp_name))
                im_parse = transforms.Resize(self.fine_width, interpolation=0)(im_parse_pil_big)
                parse = torch.from_numpy(np.array(im_parse)[None]).long()
            else:
                print("FATAL: Could not save CIHP generation")
                sys.exit(1)

        labels = {
            0: ['background', [0, 10]],
            1: ['hair', [1, 2]],
            2: ['face', [4, 13]],
            3: ['upper', [5, 6, 7]],
            4: ['bottom', [9, 12]],
            5: ['left_arm', [14]],
            6: ['right_arm', [15]],
            7: ['left_leg', [16]],
            8: ['right_leg', [17]],
            9: ['left_shoe', [18]],
            10: ['right_shoe', [19]],
            11: ['socks', [8]],
            12: ['noise', [3, 11]]
        }

        parse_map = torch.FloatTensor(20, self.fine_height, self.fine_width).zero_()
        parse_map = parse_map.scatter_(0, parse, 1.0)
        new_parse_map = torch.FloatTensor(self.semantic_nc, self.fine_height, self.fine_width).zero_()

        for i in range(len(labels)):
            for label in labels[i][1]:
                new_parse_map[i] += parse_map[label]

        parse_onehot = torch.FloatTensor(1, self.fine_height, self.fine_width).zero_()
        for i in range(len(labels)):
            for label in labels[i][1]:
                parse_onehot[0] += parse_map[label] * i

        # load pose points
        self.openpose += f" {self.root} {self.person} {self.clothing} {filename}"
        openpose_err = os.system(self.openpose)
        if openpose_err:
            print("FATAL: Openpose command failed")
            sys.exit(1)
            # sys.exit(openpose_err)
        # else:
        #     openpose_name = im_name.replace('image', 'openpose_img').replace('.jpg', '_rendered.png')
        #     openpose_json = im_name.replace('image', 'openpose_img').replace('.jpg', '_rendered.png')
        #     if os.path.isfile(osp.join(self.data_path, openpose_name)):
        #         # parse_name = im_name.replace('image', 'cihp').replace('.jpg', '.png')  # VITON
        #         pose_map = Image.open(osp.join(self.data_path, openpose_name))
        #         pose_map = transforms.Resize(self.fine_width, interpolation=2)(pose_map)
        #         pose_map = self.transform(pose_map)  # [-1,1]

        #         pose_name = im_name.replace('image', 'openpose_json').replace('.jpg',
        #                                                                       '_keypoints.json')  # VITON
        #         with open(osp.join(self.data_path, pose_name), 'r') as f:
        #             pose_label = json.load(f)
        #             pose_data = pose_label['people'][0]['pose_keypoints_2d']
        #             pose_data = np.array(pose_data)
        #             pose_data = pose_data.reshape((-1, 3))[:, :2]
        #     else:
        #         print("FATAL: Could not save Openpose generation")
        #         sys.exit(1)



        # load image-parse-agnostic
        # self.parse_agnostic += f" {self.root} {self.person} {self.clothing} {filename}"
        # parse_agnostic_err = os.system(self.parse_agnostic)
        if 1 == 0:
            print("FATAL: Parse agnostic command failed")
            sys.exit(1)
            # sys.exit(parse_agnostic_err)
        else:
            parse_agnostic_name = im_name.replace('image', 'image-parse-agnostic').replace('.jpg', '.png')
            if os.path.isfile(osp.join(self.data_path, parse_agnostic_name)):
                # parse_name = im_name.replace('image', 'cihp').replace('.jpg', '.png')  # VITON
                image_parse_agnostic = Image.open(osp.join(self.data_path,parse_agnostic_name))
                image_parse_agnostic = transforms.Resize(self.fine_width, interpolation=0)(image_parse_agnostic)
                parse_agnostic = torch.from_numpy(np.array(image_parse_agnostic)[None]).long()
                image_parse_agnostic = self.transform(image_parse_agnostic.convert('RGB'))
            else:
                print("FATAL: Could not save Parse agnostic generation")
                sys.exit(1)


        parse_agnostic_map = torch.FloatTensor(20, self.fine_height, self.fine_width).zero_()
        parse_agnostic_map = parse_agnostic_map.scatter_(0, parse_agnostic, 1.0)
        new_parse_agnostic_map = torch.FloatTensor(self.semantic_nc, self.fine_height, self.fine_width).zero_()
        for i in range(len(labels)):
            for label in labels[i][1]:
                new_parse_agnostic_map[i] += parse_agnostic_map[label]

        # parse cloth & parse cloth mask
        pcm = new_parse_map[3:4]
        im_c = im * pcm + (1 - pcm)


        # load densepose
        # self.densepose += f" {self.root} {self.person} {self.clothing} {filename}"
        # densepose_err = os.system(self.densepose)
        if 1 == 0:
            print("FATAL: Densepose command failed")
            sys.exit(1)
            # sys.exit(densepose_err)
        else:
            densepose_name = im_name.replace('image', 'densepose')
            if os.path.isfile(osp.join(self.data_path, densepose_name)):
                # parse_name = im_name.replace('image', 'cihp').replace('.jpg', '.png')  # VITON
                densepose_map = Image.open(osp.join(self.data_path, densepose_name))
                densepose_map = transforms.Resize(self.fine_width, interpolation=2)(densepose_map)
                densepose_map = self.transform(densepose_map)  # [-1,1]
            else:
                print("FATAL: Could not save Densepose generation")
                sys.exit(1)


        agnostic = self.get_agnostic(im_pil_big, im_parse_pil_big, pose_data)
        agnostic = transforms.Resize(self.fine_width, interpolation=2)(agnostic)
        agnostic = self.transform(agnostic)


        focal = .5 * self.W / np.tan(.5 * self.camera_angle_x)
        K = np.array([[focal, 0, 0.5 * self.W], [0, focal, 0.5 * self.H], [0, 0, 1]])
        result = {
            'c_name': c_name,  # for visualization
            'im_name': im_name,  # for visualization or ground truth,
            'transform_matrix': torch.tensor(transform_matrix),
            'H': self.H,
            'W': self.W,
            'K': K,
            'agnostic': agnostic,
            # intput 1 (clothfloww)
            'cloth': c,  # for input
            'cloth_mask': cm,  # for input
            # intput 2 (segnet)
            'parse_agnostic': new_parse_agnostic_map,
            'densepose': densepose_map,
            'pose': pose_map,  # for conditioning
            # GT
            'parse_onehot': parse_onehot,  # Cross Entropy
            'parse': new_parse_map,  # GAN Loss real
            'pcm': pcm,  # L1 Loss & vis
            'parse_cloth': im_c,  # VGG Loss & vis
            # visualization
            'image': im,  # for visualization
        }

        return result

    def __len__(self):
        return len(self.im_names)


class CPDataLoader(object):
    def __init__(self, opt, dataset):
        super(CPDataLoader, self).__init__()

        if opt.shuffle:
            train_sampler = torch.utils.data.sampler.RandomSampler(dataset)
        else:
            train_sampler = None

        self.data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
            num_workers=opt.workers, pin_memory=True, drop_last=True, sampler=train_sampler)
        self.dataset = dataset
        self.data_iter = self.data_loader.__iter__()

    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()

        return batch
