import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import time

def draw_point_cloud(data, with_no_gui = False, label=None, clusterCenters=None, title='feature visulization'):
    plt.clf()
    fig = plt.figure()
    data = data.view(-1, 3)
    ax = fig.add_subplot(111, projection='3d')
    plt.rcParams['figure.max_open_warning'] = 400
    print("data size = ", data.size())
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    if label is not None:
        color = 2*label + 6
    else:
        color = 'r'

    ax.scatter(x,  # x
               y,  # y
               z,  # z
               c='b',  # height data for color
               cmap='Blues',
               marker="o", zorder=1, alpha=0.1)
    if clusterCenters is not None:
        ax.scatter(clusterCenters[:, 0], clusterCenters[:, 1], clusterCenters[:, 2], marker='*', c='r', s=200, zorder=2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.axis('on')  # 设置坐标轴不可见
    ax.grid(True)  # 设置背景网格不可见

    plt.savefig(title + '.png')
    if with_no_gui is not True:
        plt.show()

    #time.sleep(0.1)
    plt.close()