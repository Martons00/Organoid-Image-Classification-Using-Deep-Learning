'''
Training code for MRBrainS18 datasets segmentation
Written by Whalechen
'''

from setting import parse_opts 
from model import generate_model
import torch
import numpy as np
from torch import nn
from torch import optim
import time
from utils.logger import log
from scipy import ndimage
import os



if __name__ == '__main__':
    # settting
    sets = parse_opts()   
    if sets.ci_test:
        sets.img_list = './toy_data/test_ci.txt' 
        sets.n_epochs = 1
        sets.no_cuda = True
        sets.data_root = './toy_data'
        sets.pretrain_path = "/home/mraffael/martone_project/Organoid-Image-Classification-Using-Deep-Learning/models/MedicalNet/pretrain/resnet_50.pth"
        sets.num_workers = 0
        sets.model_depth = 18
        sets.resnet_shortcut = 'A'
        sets.gpu_id = [0]
        sets.input_D = 128
        sets.input_H = 128
        sets.input_W = 128
       
     
    
    # getting model
    torch.manual_seed(sets.manual_seed)
    model, parameters = generate_model(sets) 
    # optimizer
    if sets.ci_test:
        params = [{'params': parameters, 'lr': sets.learning_rate}]
    else:
        params = [
                { 'params': parameters['base_parameters'], 'lr': sets.learning_rate }, 
                { 'params': parameters['new_parameters'], 'lr': sets.learning_rate*100 }
                ]
    optimizer = torch.optim.SGD(params, momentum=0.9, weight_decay=1e-3)   
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    
    # train from resume
    if sets.resume_path:
        if os.path.isfile(sets.resume_path):
            print("=> loading checkpoint '{}'".format(sets.resume_path))
            checkpoint = torch.load(sets.resume_path)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
              .format(sets.resume_path, checkpoint['epoch']))
    
    print(model)
    samples = torch.randn(1, 1, sets.input_D, sets.input_H, sets.input_W)
    out = model(samples)
    print(out.shape)


