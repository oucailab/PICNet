from net import *
from dataset import *
import numpy as np

import time
from sklearn.metrics import accuracy_score

print("debug:")

datasetType = int(input("Choose Dataset: 0 Houston2013; 1 Houston2018; 2 Trento: "))
# datasetType = 4

if(datasetType == 0):
    hsi_path = '../data/Houston2013/houston_hsi.mat'
    lidar_path = '../data/Houston2013/houston_lidar.mat'
    gt_path = '../data/Houston2013/houston_gt.mat'
    index_path = '../data/Houston2013/houston_index.mat'
    train_loader, test_loader, trntst_loader, all_loader = getHouston2013Data(hsi_path, lidar_path, gt_path, index_path, 30, 11, 128, 0)
elif(datasetType == 1):
    hsi_path = '../data/Houston2018/houston_hsi.mat'
    lidar_path = '../data/Houston2018/houston_lidar.mat'
    gt_path = '../data/Houston2018/houston_gt.mat'
    index_path = '../data/Houston2018/houston_index.mat'
    train_loader, test_loader, trntst_loader, all_loader = getHouston2018Data(hsi_path, lidar_path, gt_path, index_path, 30, 11, 128, 0)
elif(datasetType == 2):    
    hsi_path = '../data/Trento/trento_hsi.mat'
    lidar_path = '../data/Trento/trento_lidar.mat'
    gt_path = '../data/Trento/trento_gt.mat'
    index_path = '../data/Trento/trento_index.mat'
    train_loader, test_loader, trntst_loader, all_loader = getTrentoData(hsi_path, lidar_path, gt_path, index_path, 30, 11, 128, 0)
elif(datasetType == 3):    
    hsi_path = "../data/Berlin/berlin_hsi.mat"
    sar_path = "../data/Berlin/berlin_sar.mat"
    gt_path = "../data/Berlin/berlin_gt.mat"
    index_path = "../data/Berlin/berlin_index.mat"
    train_loader, test_loader, all_loader = getBerlinData(hsi_path, sar_path, gt_path, index_path, 30, 11, 128, 0)
elif(datasetType == 4):    
    hsi_path = "../data/Augsburg/augsburg_hsi.mat"
    sar_path = "../data/Augsburg/augsburg_sar.mat"
    gt_path = "../data/Augsburg/augsburg_gt.mat"
    index_path = "../data/Augsburg/augsburg_index.mat"
    train_loader, test_loader, all_loader = getAugsburgData(hsi_path, sar_path, gt_path, index_path, 30, 11, 128, 0)

epoch_nums = 1

# data = train_loader.dataset.__getitem__(0)

# print(train_loader.dataset.__dict__.keys())
# print(train_loader.dataset.hsi.shape)
# print(train_loader.dataset.sar.shape)
# print(np.array(train_loader.dataset.pos).shape)
# print(np.array(train_loader.dataset.gt).shape)

# # print(train_loader.dataset.__dict__)
# print(len(data))
# print("data[0]")
# print(data[0])
# print(data[0].shape) # torch.Size([1, 30, 11, 11])
# print("data[1]")
# print(data[1])
# print(data[1].shape) # torch.Size([4, 11, 11])
# print("data[2]")
# print(data[2])
# print(data[2].shape) # torch.Size([])

# train(epoch_nums, train_loader, test_loader, out_features[datasetType], model_savepath[datasetType])