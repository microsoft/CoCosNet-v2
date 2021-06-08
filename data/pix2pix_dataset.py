# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import os
import random
from PIL import Image

from data.base_dataset import BaseDataset, get_params, get_transform


class Pix2pixDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--no_pairing_check', action='store_true', help='If specified, skip sanity check of correct label-image file pairing')
        return parser

    def initialize(self, opt):
        self.opt = opt
        label_paths, image_paths = self.get_paths(opt)
        label_paths = label_paths[:opt.max_dataset_size]
        image_paths = image_paths[:opt.max_dataset_size]
        if not opt.no_pairing_check:
            for path1, path2 in zip(label_paths, image_paths):
                assert self.paths_match(path1, path2), \
                    "The label-image pair (%s, %s) do not look like the right pair because the filenames are quite different. Are you sure about the pairing? Please see data/pix2pix_dataset.py to see what is going on, and use --no_pairing_check to bypass this." % (path1, path2)
        self.label_paths = label_paths
        self.image_paths = image_paths
        size = len(self.label_paths)
        self.dataset_size = size
        self.real_reference_probability = 1 if opt.phase == 'test' else opt.real_reference_probability
        self.hard_reference_probability = 0 if opt.phase == 'test' else opt.hard_reference_probability
        self.ref_dict, self.train_test_folder = self.get_ref(opt)
        
    def get_paths(self, opt):
        label_paths = []
        image_paths = []
        assert False, "A subclass of Pix2pixDataset must override self.get_paths(self, opt)"
        return label_paths, image_paths

    def paths_match(self, path1, path2):
        filename1_without_ext = os.path.splitext(os.path.basename(path1))[0]
        filename2_without_ext = os.path.splitext(os.path.basename(path2))[0]
        return filename1_without_ext == filename2_without_ext

    def get_label_tensor(self, path):
        label = Image.open(path)
        params1 = get_params(self.opt, label.size)
        transform_label = get_transform(self.opt, params1, method=Image.NEAREST, normalize=False)
        label_tensor = transform_label(label) * 255.0
        label_tensor[label_tensor == 255] = self.opt.label_nc
        # 'unknown' is opt.label_nc
        return label_tensor, params1

    def __getitem__(self, index):
        # label Image
        label_path = self.label_paths[index]
        label_path = os.path.join(self.opt.dataroot, label_path)
        label_tensor, params1 = self.get_label_tensor(label_path)
        # input image (real images)
        image_path = self.image_paths[index]
        image_path = os.path.join(self.opt.dataroot, image_path)
        image = Image.open(image_path).convert('RGB')
        transform_image = get_transform(self.opt, params1)
        image_tensor = transform_image(image)
        ref_tensor = 0
        label_ref_tensor = 0
        random_p = random.random()
        if random_p < self.real_reference_probability or self.opt.phase == 'test':
            key = image_path.split('deepfashionHD/')[-1]
            val = self.ref_dict[key]
            if random_p < self.hard_reference_probability:
                #hard reference
                path_ref = val[1]
            else:
                #easy reference
                path_ref = val[0]
            if self.opt.dataset_mode == 'deepfashionHD':
                path_ref = os.path.join(self.opt.dataroot, path_ref)
            else:
                path_ref = os.path.dirname(image_path).replace(self.train_test_folder[1], self.train_test_folder[0]) + '/' + path_ref
            image_ref = Image.open(path_ref).convert('RGB')
            if self.opt.dataset_mode != 'deepfashionHD':
                path_ref_label = path_ref.replace('.jpg', '.png')
                path_ref_label = self.imgpath_to_labelpath(path_ref_label)
            else: 
                path_ref_label = self.imgpath_to_labelpath(path_ref)
            label_ref_tensor, params = self.get_label_tensor(path_ref_label)
            transform_image = get_transform(self.opt, params)
            ref_tensor = transform_image(image_ref)
            self_ref_flag = 0.0
        else:
            pair = False
            if self.opt.dataset_mode == 'deepfashionHD' and self.opt.video_like:
                key = image_path.replace('\\', '/').split('deepfashionHD/')[-1]
                val = self.ref_dict[key]
                ref_name = val[0]
                key_name = key
                path_ref = os.path.join(self.opt.dataroot, ref_name)
                image_ref = Image.open(path_ref).convert('RGB')
                label_ref_path = self.imgpath_to_labelpath(path_ref)
                label_ref_tensor, params = self.get_label_tensor(label_ref_path)
                transform_image = get_transform(self.opt, params)
                ref_tensor = transform_image(image_ref) 
                pair = True
            if not pair:
                label_ref_tensor, params = self.get_label_tensor(label_path)
                transform_image = get_transform(self.opt, params)
                ref_tensor = transform_image(image)
            self_ref_flag = 1.0
        input_dict = {'label': label_tensor,
                      'image': image_tensor,
                      'path': image_path,
                      'self_ref': self_ref_flag,
                      'ref': ref_tensor,
                      'label_ref': label_ref_tensor
                      }
        return input_dict

    def __len__(self):
        return self.dataset_size

    def get_ref(self, opt):
        pass

    def imgpath_to_labelpath(self, path):
        return path
