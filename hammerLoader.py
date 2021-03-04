#encoding: utf-8
import numpy as np
import warnings
import os
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

from tqdm import tqdm


class hammerLoader(Dataset):
    def __init__(self, root,  npoints=1024, split='train', uniform=False, cache_size=15000):
        self.root = "/home/baochen/votenet/data/hammer/SyntheticHammer"
        self.data = list()
        self.npoints = npoints
        self.split = split
        self.uniform = uniform
        self.cache_size = 15000

        for i in tqdm(range(0)):
            file_path_1 = os.path.join(self.root, 'hammers_pcd')
            file_path_1 = os.path.join(file_path_1, 'Hammer_'+str(i)+'.pcd')
            file_1 = open(file_path_1, "r")  # 打开文件以便写入
            file_path_2 = os.path.join(self.root, 'hammers_kp')
            file_path_2 = os.path.join(file_path_2, 'Hammer_'+str(i)+'_kp.txt')
            file_2 = open(file_path_2, "r")  # 打开文件以便写入

            instance = dict()
            point_cloud = list()

            for j, line in enumerate(file_1):
                if j <= 11:  # 忽略前十一行
                    continue
                words = line.split(" ")
                words[-1] = words[-1][:-1] #删除掉最后一个元素的换行符
                point = [float(words[i]) for i in range(len(words))]
                point_cloud.append(point)
            instance['point_cloud'] = point_cloud

            for j, line in enumerate(file_2):
                if j == 0:  # 忽略第一行
                    continue
                words = line.split(" ")
                words[-1] = words[-1][:-1] #删除掉最后一个元素的换行符
                if j == 1:
                    x_g = [float(words[i]) for i in range(len(words))]
                    instance['x_g'] = x_g
                if j == 2:
                    x_f = [float(words[i]) for i in range(len(words))]
                    instance['x_f'] = x_f
                if j == 3:
                    x_e = [float(words[i]) for i in range(len(words))]
                    instance['x_e'] = x_e

            file_1.close()
            file_2.close()
            #print(instance)
            self.data.append(instance)

    def __len__(self):
        if self.split == 'train':
            return 1500
        elif self.split == 'test':
            return 500

    def _get_item(self, index):
        #instance = self.data[index]
        #point_cloud = np.array(instance['point_cloud']).astype(np.float32)
        if self.split == 'test':
            index += 1500

        file_path_1 = os.path.join(self.root, 'hammers_pcd')
        file_path_1 = os.path.join(file_path_1, 'Hammer_' + str(index) + '.pcd')
        file_1 = open(file_path_1, "r")  # 打开文件以便写入
        file_path_2 = os.path.join(self.root, 'hammers_kp')
        file_path_2 = os.path.join(file_path_2, 'Hammer_' + str(index) + '_kp.txt')
        file_2 = open(file_path_2, "r")  # 打开文件以便写入

        point_cloud_list = list()

        for j, line in enumerate(file_1):
            if j <= 11:  # 忽略前十一行
                continue
            words = line.split(" ")
            words[-1] = words[-1][:-1]  # 删除掉最后一个元素的换行符
            point = [float(words[i]) for i in range(len(words))]
            point_cloud_list.append(point)
        point_cloud = np.array(point_cloud_list).astype(np.float32)
        point_cloud = farthest_point_sample(point_cloud, self.npoints)

        for j, line in enumerate(file_2):
            if j == 0:  # 忽略第一行
                continue
            words = line.split(" ")
            words[-1] = words[-1][:-1]  # 删除掉最后一个元素的换行符
            if j == 1:
                x_g = [float(words[i]) for i in range(len(words))]
            if j == 2:
                x_f = [float(words[i]) for i in range(len(words))]
            if j == 3:
                x_e = [float(words[i]) for i in range(len(words))]
        x_g = np.array(x_g).astype(np.float32)
        x_f = np.array(x_f).astype(np.float32)
        x_e = np.array(x_e).astype(np.float32)

        return point_cloud, x_g, x_f, x_e

    def __getitem__(self, index):
        return self._get_item(index)




if __name__ == '__main__':
    import torch

    root = os.path.dirname(os.getcwd())
    data = hammerLoader(root, split='train', uniform=False)
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    for point_cloud, x_g, x_f, x_e in DataLoader:
        print(point_cloud.shape)
        print(x_g.shape)