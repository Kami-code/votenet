from readData import *
from tools import *
from plot import *
import torch
import numpy
import os
from tqdm import tqdm


def find_point(filename): #读入点云输出两个特征点
    my_data = read_data(filename, 1) #读入点云数据
    tensor_data = torch.from_numpy(my_data[:,0:3]) #转化为tensor的x，y，z数据
    tensor_data = tensor_data.view(1,tensor_data.shape[0], -1) #添加第一维
    indices = farthest_point_sample(tensor_data, 500) #最远点采样得到稀疏点云的编号
    new_data = torch.index_select(tensor_data, 1, indices.view(-1))
    new_data = new_data.view(-1,3) #删除第一维
    #draw_point_cloud(new_data)
    from sklearn.mixture import GaussianMixture
    gmm = GaussianMixture(n_components=2).fit(new_data)
    dict = gmm._get_parameters()
    #labels = gmm.predict(new_data)
    #draw_point_cloud(new_data, labels,dict[1])
    return_tensor = torch.from_numpy(numpy.array(dict[1]))
    return return_tensor.view(-1)


class ModelNetDataProcessor:
    def __init__(self, root):
        self.root = root
        catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')  # 得到类别名称的路径
        cat = [line.rstrip() for line in open(catfile)]
        self.classes = dict(zip(cat, range(len(cat))))  # 得到dict[class]=class.size
        self.shape_ids = dict()
        self.shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]  # 得到每一类的每一个样本的id
        self.shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]
        self.shape_names = dict()
        self.shape_names['train'] = ['_'.join(x.split('_')[0:-1]) for x in self.shape_ids['train']]
        self.shape_names['test'] = ['_'.join(x.split('_')[0:-1]) for x in self.shape_ids['test']]

    def create(self, split='test'):
        assert (split == 'train' or split == 'test')
        # list of (shape_name, shape_txt_file_path) tuple
        vector = []
        for i in tqdm(range(len(self.shape_ids[split]))):
            f = open(split + ".txt", "a+")  # 打开文件以便写入
            filename_i = os.path.join(self.root, self.shape_names[split][i], self.shape_ids[split][i]) + '.txt'
            vector_i = find_point(filename_i)
            vector.append(vector_i)
            vector_i = vector_i.numpy().tolist()
            print(vector_i)  # 输出到终端
            f.write(self.shape_ids[split][i] + " ")  # 输出id
            for j in range(len(vector_i)):
                f.write(str(vector_i[j]) + " ")  # 输出两对点
            f.write("\n")  # 输出换行符
            f.close()

    def processor(self, split='train'):
        assert (split == 'train' or split == 'test')
        if split == 'train':
            f = open((os.path.join(self.root, 'training_data.txt')), "r")  # 打开文件以便写入
        if split == 'test':
            f = open((os.path.join(self.root, 'test_data.txt')), "r")  # 打开文件以便写入
        data = {}
        for line in f:
            l = line.split(" ")
            data_vector = [float(l[i]) for i in range(1, 7)]
            data[l[0]] = data_vector
            datapath = os.path.join(self.root, l[0][:-5], l[0] + '.txt')
            point_set = np.loadtxt(datapath, delimiter=',').astype(np.float32)
            point_set = torch.from_numpy(point_set[:, 0:3])  # 转化为tensor的x，y，z数据
            point_set = point_set.view(1, point_set.shape[0], -1)  # 添加第一维
            indices = farthest_point_sample(point_set, 500)  # 最远点采样得到稀疏点云的编号
            new_data = torch.index_select(point_set, 1, indices.view(-1))
            new_data = new_data.view(-1, 3)  # 删除第一维
            data_vector = torch.from_numpy(numpy.array(data_vector)).view(2, 3)
            draw_point_cloud(new_data, clusterCenters=data_vector, title=l[0])
        f.close()