#coding=utf-8
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from glob import glob
import os.path as osp
import numpy as np

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
        self.person = opt.person
        self.clothing = opt.clothing
        self.in_shop_clothing = opt.in_shop_clothing
        self.datamode = opt.datamode  # train or test or self-defined
        self.data_list = opt.data_list
        self.fine_height = opt.fine_height
        self.fine_width = opt.fine_width
        self.semantic_nc = opt.semantic_nc
        self.data_path = osp.join(opt.dataroot, opt.datamode)
        self.transforms_data = get_transforms_data(opt)
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
        person = f"image/{opt.person}_{opt.clothing}*"
        data_items = glob(osp.join(self.data_path, person))
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
        return "CPDataset"

    def __getitem__(self, index):
        im_name = self.im_names[index]
        transform_matrix = self.transform_matrices[index]
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
        im = Image.open(osp.join(self.data_path, 'image', im_name))
        im = transforms.Resize(self.fine_width, interpolation=2)(im)
        im = self.transform(im.convert('RGB'))

        # load parsing image
        parse_name = im_name.replace('.jpg', '.png')
        im_parse = Image.open(osp.join(self.data_path, 'image-parse-v3', parse_name))
        im_parse = transforms.Resize(self.fine_width, interpolation=0)(im_parse)
        parse = torch.from_numpy(np.array(im_parse)[None]).long()
        im_parse = self.transform(im_parse.convert('RGB'))

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

        # load image-parse-agnostic
        image_parse_agnostic = Image.open(osp.join(self.data_path, 'image-parse-agnostic-v3.2', parse_name))
        image_parse_agnostic = transforms.Resize(self.fine_width, interpolation=0)(image_parse_agnostic)
        parse_agnostic = torch.from_numpy(np.array(image_parse_agnostic)[None]).long()
        image_parse_agnostic = self.transform(image_parse_agnostic.convert('RGB'))

        parse_agnostic_map = torch.FloatTensor(20, self.fine_height, self.fine_width).zero_()
        parse_agnostic_map = parse_agnostic_map.scatter_(0, parse_agnostic, 1.0)
        new_parse_agnostic_map = torch.FloatTensor(self.semantic_nc, self.fine_height, self.fine_width).zero_()
        for i in range(len(labels)):
            for label in labels[i][1]:
                new_parse_agnostic_map[i] += parse_agnostic_map[label]

        # parse cloth & parse cloth mask
        pcm = new_parse_map[3:4]
        im_c = im * pcm + (1 - pcm)

        # load pose points
        pose_name = im_name.replace('.jpg', '_rendered.png')
        pose_map = Image.open(osp.join(self.data_path, 'openpose_img', pose_name))
        pose_map = transforms.Resize(self.fine_width, interpolation=2)(pose_map)
        pose_map = self.transform(pose_map)  # [-1,1]

        # load densepose
        densepose_name = im_name.replace('image', 'image-densepose')
        densepose_map = Image.open(osp.join(self.data_path, 'image-densepose', densepose_name))
        densepose_map = transforms.Resize(self.fine_width, interpolation=2)(densepose_map)
        densepose_map = self.transform(densepose_map)  # [-1,1]
        focal = .5 * self.W / np.tan(.5 * self.camera_angle_x)
        K = np.array([[focal, 0, 0.5 * self.W], [0, focal, 0.5 * self.H], [0, 0, 1]])
        result = {
            'c_name': c_name,  # for visualization
            'im_name': im_name,  # for visualization or ground truth,
            'transform_matrix': transform_matrix,
            'H': self.H,
            'W': self.W,
            'K': K,
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
