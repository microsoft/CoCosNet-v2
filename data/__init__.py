# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import importlib
import torch.utils.data
from data.base_dataset import BaseDataset


def find_dataset_using_name(dataset_name):
    dataset_filename = "data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)
    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset):
            dataset = cls
    if dataset is None:
        raise ValueError("In %s.py, there should be a subclass of BaseDataset "
                         "with class name that matches %s in lowercase." %
                         (dataset_filename, target_dataset_name))
    return dataset


def get_option_setter(dataset_name):    
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


def create_dataloader(opt):
    dataset = find_dataset_using_name(opt.dataset_mode)
    instance = dataset()
    instance.initialize(opt)
    print("Dataset [%s] of size %d was created" % (type(instance).__name__, len(instance)))
    dataloader = torch.utils.data.DataLoader(
        instance,
        batch_size=opt.batchSize,
        shuffle=(opt.phase=='train'),
        num_workers=int(opt.nThreads),
        drop_last=(opt.phase=='train')
    )
    return dataloader
