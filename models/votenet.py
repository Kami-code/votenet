#coding: UTF-8
# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Deep hough voting network for 3D object detection in point clouds.

Author: Charles R. Qi and Or Litany
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
from backbone_module import Pointnet2Backbone
from voting_module import VotingModule
from proposal_module import ProposalModule
from dump_helper import dump_results
from loss_helper import getKPLoss


class VoteNet(nn.Module):
    r"""
        A deep neural network for 3D object detection with end-to-end optimizable hough voting.

        Parameters
        ----------
        num_class: int
            Number of semantics classes to predict over -- size of softmax classifier
        num_heading_bin: int
        num_size_cluster: int
        input_feature_dim: (default: 0)
            Input dim in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        num_proposal: int (default: 128)
            Number of proposals/detections generated from the network. Each proposal is a 3D OBB with a semantic class.
        vote_factor: (default: 1)
            Number of votes generated from each seed point.
    """

    def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr,
        input_feature_dim=0, num_proposal=128, vote_factor=1, sampling='vote_fps'):
        super().__init__()

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        #assert(mean_size_arr.shape[0] == self.num_size_cluster)
        self.input_feature_dim = input_feature_dim
        self.num_proposal = num_proposal
        self.vote_factor = vote_factor
        self.sampling = sampling

        # Backbone point feature learning
        self.backbone_net = Pointnet2Backbone(input_feature_dim=self.input_feature_dim)

        # Hough voting
        self.vgen = VotingModule(self.vote_factor, 256)


    def forward(self, inputs):
        """ Forward pass of the network

        Args:
            inputs: dict
                {point_clouds}

                point_clouds: Variable(torch.cuda.FloatTensor)
                    (B, N, 3 + input_channels) tensor
                    Point cloud to run predicts on
                    Each point in the point-cloud MUST
                    be formated as (x, y, z, features...)
        Returns:
            end_points: dict
        """
        # 输入B * N * 3
        end_points = {}
        #cdinputs['point_clouds'] = inputs['point_clouds'].view(-1, inputs['point_clouds'].shape[2], inputs['point_clouds'].shape[3])
        batch_size = inputs['point_clouds'].shape[0]

        end_points = self.backbone_net(inputs['point_clouds'], end_points)  # 跑一个pointnet++ backbone得到B * M * (3+C)
        # --------- HOUGH VOTING ---------
        xyz = end_points['fp2_xyz']
        features = end_points['fp2_features']
        end_points['seed_inds'] = end_points['fp2_inds']  # 取fp2层的特征作为feature
        end_points['seed_xyz'] = xyz
        end_points['seed_features'] = features
        # end_points['seed_xyz'] = B * M * 3
        # end_points['seed_features'] = B * M * C
        
        xyz, features = self.vgen(xyz, features)
        features_norm = torch.norm(features, p=2, dim=1)
        features = features.div(features_norm.unsqueeze(1))  # 特征归一化
        end_points['vote_xyz'] = xyz
        end_points['vote_features'] = features
        # end_points['vote_xyz'] = B * M * 3
        # end_points['vote_features'] = B * M * C

        #end_points = self.pnet(xyz, features, end_points) #这行是跑proposal module，我们的任务中不需要

        return end_points

import sys
sys.path.append("..")
from hammerLoader import hammerLoader
from plot import *

if __name__=='__main__':
    model = VoteNet(10, 12, 10, np.random.random((10, 3))).cuda()
    DATA_PATH = '../data/modelnet40_normal_resampled/'
    npoint = 2048
    batch_size = 2


    # Define dataset
    TEST_DATASET = hammerLoader(root=DATA_PATH, npoints=2048, split='test')
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=batch_size, shuffle=False, num_workers=4) # dataloader一次性创建num_worker个工作进程

    # Model forward pass
    sample = dict()
    GT = dict()
    for point_cloud, x_g, x_f, x_e in testDataLoader:
        sample['point_clouds'] = point_cloud
        GT['x_g'] = x_g
        GT['x_f'] = x_f
        GT['x_e'] = x_e
        break
    print(sample['point_clouds'])
    draw_point_cloud(point_cloud, with_no_gui=True, title="origin")

    inputs = {'point_clouds': (sample['point_clouds']).unsqueeze(0).cuda()}

    end_points = model(inputs)
    draw_point_cloud(end_points['vote_xyz'], with_no_gui=True, title="endpoint")
    #for key in end_points:
    #    print(key, end_points[key])

    # Compute loss
    for key in sample:
        end_points[key] = sample[key].unsqueeze(0).cuda()
    loss, end_points = getKPLoss(end_points, GT)
    print('loss', loss)
    #dump_results(end_points, 'tmp', GT)

'''
# The origin version of votenet
if __name__=='__main__':
    sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))
    from sunrgbd_detection_dataset import SunrgbdDetectionVotesDataset, DC
    from loss_helper import get_loss

    # Define model
    model = VoteNet(10,12,10,np.random.random((10,3))).cuda()
    
    try:
        # Define dataset
        TRAIN_DATASET = SunrgbdDetectionVotesDataset('train', num_points=20000, use_v1=True)

        # Model forward pass
        sample = TRAIN_DATASET[5]
        inputs = {'point_clouds': torch.from_numpy(sample['point_clouds']).unsqueeze(0).cuda()}
    except:
        print('Dataset has not been prepared. Use a random sample.')
        inputs = {'point_clouds': torch.rand((20000,3)).unsqueeze(0).cuda()}

    end_points = model(inputs)
    for key in end_points:
        print(key, end_points[key])

    try:
        # Compute loss
        for key in sample:
            end_points[key] = torch.from_numpy(sample[key]).unsqueeze(0).cuda()
        loss, end_points = get_loss(end_points, DC)
        print('loss', loss)
        end_points['point_clouds'] = inputs['point_clouds']
        end_points['pred_mask'] = np.ones((1,128))
        dump_results(end_points, 'tmp', DC)
    except:
        print('Dataset has not been prepared. Skip loss and dump.')
'''