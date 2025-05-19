import os
import math
import torch
import random
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from skimage import io
import torch.optim as optim
from operator import truediv
from scipy.io import loadmat
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
from sklearn.decomposition import PCA
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from S2Enet import S2ENet

from Endnet import EndNet,EndNet_Loss

from Cross_fusion_CNN import Cross_fusion_CNN
from Cross_fusion_CNN import Cross_fusion_CNN_Loss

def set_random_seed(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def applyPCA(data, n_components):

    h, w, b = data.shape
    pca = PCA(n_components=n_components)
    data = np.reshape(pca.fit_transform(np.reshape(data, (-1, b))), (h, w, -1))
    return data

# Dataset 类


class HXDataset(Dataset):

    def __init__(self, hsi, X, pos, windowSize, gt=None, flip=False):

        self.modes = ['symmetric', 'reflect']
        self.flip = flip
        self.pad = windowSize // 2
        self.windowSize = windowSize
        self.hsi = np.pad(hsi, ((self.pad, self.pad),
                                (self.pad, self.pad), (0, 0)), mode=self.modes[windowSize % 2])

        self.X = np.pad(X, ((self.pad, self.pad),
                            (self.pad, self.pad), (0, 0)), mode=self.modes[windowSize % 2])

        self.pos = pos
        self.gt = None
        if gt is not None:
            self.gt = gt

    def __getitem__(self, index):

        h, w = self.pos[index, :]
        hsi = self.hsi[h: h + self.windowSize, w: w + self.windowSize]
        X = self.X[h: h + self.windowSize, w: w + self.windowSize]
        hsi = ToTensor()(hsi).float()
        X = ToTensor()(X).float()
        if self.flip:
            trans = [transforms.RandomHorizontalFlip(1.),
                     transforms.RandomVerticalFlip(1.)]
            if random.random() < 0.5:
                i = random.randint(0, 1)
                hsi = trans[i](hsi)
                X = trans[i](X)
        if self.gt is not None:
            gt = torch.tensor(self.gt[h, w] - 1).long()
            return hsi.unsqueeze(0), X, gt
        return hsi.unsqueeze(0), X, h, w

    def __len__(self):
        return self.pos.shape[0]


def createDataLoader(hsi_path, X_path, gt_path, index_path, keys, channels, windowSize, batch_size, num_workers, flip):

    # 加载图片数据和坐标位置
    hsi = loadmat(hsi_path)[keys[0]]
    X = loadmat(X_path)[keys[1]]
    gt = loadmat(gt_path)[keys[2]]
    train_index = loadmat(index_path)[keys[3]]
    test_index = loadmat(index_path)[keys[4]]
    trntst_index = np.concatenate((train_index, test_index), axis=0)
    all_index = loadmat(index_path)[keys[5]]

    # 使用 PCA 对 HSI 进行降维
    hsi = applyPCA(hsi, channels)

    # 创建 Dataset, 用于生成对应的 Dataloader
    HXtrainset = HXDataset(hsi, X, train_index, windowSize, gt, flip)
    HXtestset = HXDataset(hsi, X, test_index, windowSize, gt)
    HXtrntstset = HXDataset(hsi, X, trntst_index, windowSize)
    HXallset = HXDataset(hsi, X, all_index, windowSize)

    # 创建 Dataloader
    train_loader = DataLoader(
        HXtrainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(
        HXtestset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    trntst_loader = DataLoader(
        HXtrntstset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    all_loader = DataLoader(
        HXallset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print("Success!")
    return train_loader, test_loader, trntst_loader, all_loader

# 获取 Houston2018 数据集

def createHouston2018DataLoader(hsi_path, lidar_path, gt_path, index_path, channels, windowSize, batch_size, num_workers, flip=False):

    print("Houston2018!")

    # Houston2018 mat keys
    houston2018_keys = ['houston_hsi', 'houston_lidar', 'houston_gt', 'houston_train', 'houston_test', 'houston_all']

    return createDataLoader(hsi_path, lidar_path, gt_path, index_path, houston2018_keys, channels, windowSize, batch_size, num_workers, flip)


# 获取 Berlin 数据集

def createBerlinDataLoader(hsi_path, sar_path, gt_path, index_path, channels, windowSize, batch_size, num_workers, flip=False):

    print("Berlin!")

    # Berlin mat keys
    berlin_keys = ['berlin_hsi', 'berlin_sar', 'berlin_gt',
                   'berlin_train', 'berlin_test', 'berlin_all']

    return createDataLoader(hsi_path, sar_path, gt_path, index_path, berlin_keys, channels, windowSize, batch_size, num_workers, flip)


# 获取 Augsburg 数据集

def createAugsburgDataLoader(hsi_path, sar_path, gt_path, index_path, channels, windowSize, batch_size, num_workers, flip=False):

    print("Augsburg!")

    # Augsburg mat keys
    augsburg_keys = ['augsburg_hsi', 'augsburg_sar', 'augsburg_gt',
                     'augsburg_train', 'augsburg_test', 'augsburg_all']

    return createDataLoader(hsi_path, sar_path, gt_path, index_path, augsburg_keys, channels, windowSize, batch_size, num_workers, flip)


from re import X

# 计算 average accuracy 和 每个类别的准确率
def AA_andEachClassAccuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc

# 生成报告
def createReport(net, data, report_path, class_names, device):

    net.eval()
    count = 0
    for hsi, x, test_labels in data:
        # hsi = hsi.squeeze(1)
        hsi = hsi.to(device)
        x = x.to(device)
        outputs = net(hsi, x)
        outputs = np.argmax(outputs[0].detach().cpu().numpy(), axis=1)
        if count == 0:
            y_pred = outputs
            y_true = test_labels
            count = 1
        else:
            y_pred = np.concatenate((y_pred, outputs))
            y_true = np.concatenate((y_true, test_labels))

    classification = classification_report(
        y_true, y_pred, target_names=class_names, digits=4)
    confusion = confusion_matrix(y_true, y_pred)
    oa = accuracy_score(y_true, y_pred)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_true, y_pred)

    classification = str(classification)
    confusion = str(confusion)
    oa = oa * 100
    each_acc = each_acc * 100
    aa = aa * 100
    kappa = kappa * 100

    with open(report_path, 'w') as report:
        report.write('{}'.format(classification))
        report.write('\n')
        report.write('{} Overall accuracy (%)'.format(oa))
        report.write('\n')
        report.write('{} Average accuracy (%)'.format(aa))
        report.write('\n')
        report.write('{} Kappa accuracy (%)'.format(kappa))
        report.write('\n')
        report.write('\n')
        report.write('{}'.format(confusion))

# 生成 Houston2018 数据集的报告
def createHouston2018Report(net, data, report_path, device):

    # Houston2018 数据集的类别名
    houston2018_class_names = ['Healthy grass', 'Stressed grass', 'Artificial turf', 'Evergreen trees', 'Deciduous trees', 'Bare earth', 'Water', 'Residential buildings', 'Non-residential buildings',
                    'Roads', 'Sidewalks', 'Crosswalks', 'Major thoroughfares', 'Highways', 'Railways', 'Paved parking lots', 'Unpaved parking lots', 'Cars', 'Trains', 'Stadium seats']

    print("Houston2018 Start!")
    createReport(net, data, report_path, houston2018_class_names, device)
    print("Report Success!")


# 生成 Berlin 数据集的报告
def createBerlinReport(net, data, report_path, device):

    # Berlin 数据集的类别名
    berlin_class_names = ['Forest', 'Residential Area', 'Industrial Area', 'Low Plants', 'Soil', 'Allotment', 'Commercial Area', 'Water']

    print("Berlin Start!")
    createReport(net, data, report_path, berlin_class_names, device)
    print("Report Success!")


# 生成 Augsburg 数据集的报告
def createAugsburgReport(net, data, report_path, device):

    # Augsburg 数据集的类别名
    augsburg_class_names = ['Forest', 'Residential Area', 'Industrial Area', 'Low Plants', 'Allotment', 'Commercial Area', 'Water']

    print("Augsburg Start!")
    createReport(net, data, report_path, augsburg_class_names, device)
    print("Report Success!")

# 可视化 data 中的数据
def draw(net, data, save_path, device, color_map, size):

    net.eval()
    h, w = size[:]
    pred = -np.ones((h, w))
    for hsi, x, i, j in tqdm(data):
        # hsi = hsi.squeeze(1)
        hsi = hsi.to(device)
        x = x.to(device)
        output = net(hsi, x)
        output = np.argmax(output[0].detach().cpu().numpy(), axis=1)
        idx = 0
        for x, y in zip(i, j):
            pred[x, y] = output[idx]
            idx += 1
    res = np.zeros((h, w, 3), dtype=np.uint8)
    pos = pred > -1
    for i in range(h):
        for j in range(w):
            if pos[i, j]:
                res[i, j] = color_map[int(pred[i, j])]
            else:
                res[i, j] = [0, 0, 0]
    io.imsave(save_path, res)

# 可视化 Houston2018 数据集
def drawHouston2018(net, data, save_path, device):

    # Houston2018 color map
    houston2018_color_map = [[50, 205, 51], [173, 255, 48], [0, 128, 129], [34, 139, 34], [46, 79, 78], [139, 69, 18], [0, 255, 255], [255, 255, 255], [211, 211, 211], [
        254, 0, 0], [169, 169, 169], [105, 105, 105], [139, 0, 1], [200, 100, 0], [254, 165, 0], [255, 255, 0], [218, 165, 33], [255, 0, 254], [0, 0, 254], [63, 224, 208]]

    # Houston2018 尺寸
    houston2018_size = [1202, 4768]

    print("Houston2018 Start!")
    draw(net, data, save_path, device, houston2018_color_map, houston2018_size)
    print("Draw Success!")


# 可视化 Berlin 数据集
def drawBerlin(net, data, save_path, device):

    # Berlin color map
    berlin_color_map = [[26, 163, 25], [216, 216, 216], [216, 89, 89], [0, 204, 51], [204, 153, 52], [244, 231, 1], [204, 102, 204], [0, 53, 255]]

    # Berlin 尺寸
    berlin_size = [1723, 476]

    print("Berlin Start!")
    draw(net, data, save_path, device, berlin_color_map, berlin_size)
    print("Draw Success!")


# 可视化 Augsburg 数据集
def drawAugsburg(net, data, save_path, device):

    # Augsburg color map
    augsburg_color_map = [[26, 163, 25], [216, 216, 216], [216, 89, 89], [0, 204, 51], [244, 231, 1], [204, 102, 204], [0, 53, 255]]

    # Augsburg 尺寸
    augsburg_size = [332, 485]

    print("Augsburg Start!")
    draw(net, data, save_path, device, augsburg_color_map, augsburg_size)
    print("Draw Success!")


def conv3x3(in_planes, out_planes, stride=1, bias=False):
    "3x3 convolution without padding"
    return ModuleParallel(nn.Conv2d(in_planes, out_planes, kernel_size=3,
                                    stride=stride, padding=0, bias=bias))

def conv3x3_p(in_planes, out_planes, stride=1, bias=False):
    "3x3 convolution with padding"
    return ModuleParallel(nn.Conv2d(in_planes, out_planes, kernel_size=3,stride=stride, padding=1, bias=bias))

def conv1x1(in_planes, out_planes, stride=1, bias=False):
    "1x1 convolution"
    return ModuleParallel(nn.Conv2d(in_planes, out_planes, kernel_size=1,
                                    stride=stride, padding=0, bias=bias))

class Exchange(nn.Module):
    def __init__(self):
        super(Exchange, self).__init__()

    def forward(self, x, bn, bn_threshold):
        bn1, bn2 = bn[0].weight.abs(), bn[1].weight.abs()
        x1, x2 = torch.zeros_like(x[0]), torch.zeros_like(x[1])
        x1[:, bn1 >= bn_threshold] = x[0][:, bn1 >= bn_threshold]
        x1[:, bn1 < bn_threshold] = x[1][:, bn1 < bn_threshold]
        x2[:, bn2 >= bn_threshold] = x[1][:, bn2 >= bn_threshold]
        x2[:, bn2 < bn_threshold] = x[0][:, bn2 < bn_threshold]
        return [x1, x2]

class ModuleParallel(nn.Module):
    def __init__(self, module):
        super(ModuleParallel, self).__init__()
        self.module = module

    def forward(self, x_parallel):
        return [self.module(x) for x in x_parallel]

class BatchNorm2dParallel(nn.Module):
    def __init__(self, num_features, num_parallel):
        super(BatchNorm2dParallel, self).__init__()
        for i in range(num_parallel):
            setattr(self, 'bn_' + str(i), nn.BatchNorm2d(num_features))

    def forward(self, x_parallel):
        return [getattr(self, 'bn_' + str(i))(x) for i, x in enumerate(x_parallel)]

class Bottleneck(nn.Module):

    def __init__(self, planes,expansion, num_parallel, bn_threshold, stride=1):
        super(Bottleneck, self).__init__()
        self.midplane = planes//expansion
        self.conv1 = conv1x1(planes, self.midplane)
        self.bn1 = BatchNorm2dParallel(self.midplane, num_parallel)
        self.conv2 = conv3x3_p(self.midplane, self.midplane, stride=stride)
        self.bn2 = BatchNorm2dParallel(self.midplane, num_parallel)
        self.conv3 = conv1x1(self.midplane, planes)
        self.bn3 = BatchNorm2dParallel(planes, num_parallel)

        self.relu = ModuleParallel(nn.ReLU(inplace=True))
        self.num_parallel = num_parallel
        self.exchange = Exchange()
        self.bn_threshold = bn_threshold
        self.bn2_list = []
        for module in self.bn2.modules():
            if isinstance(module, nn.BatchNorm2d):
                self.bn2_list.append(module)

    def forward(self, x):
        residual = x
        out = x
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # if len(x) > 0:
        out = self.exchange(out, self.bn2_list, self.bn_threshold)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = [out[l] + residual[l] for l in range(self.num_parallel)]
        out = self.relu(out)

        return out

class Dropout(nn.Module):
    def __init__(self):
        super(Dropout, self).__init__()
    def forward(self, x):
        out = F.dropout(x, p=0.2, training=self.training)
        return out


class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SCConv(nn.Module):
    def __init__(self, inplanes, planes, pooling_r =2):
        super(SCConv, self).__init__()

        self.k1 = nn.Sequential(nn.Conv2d(planes, planes, kernel_size=3,  padding =1, bias=False),
                    nn.BatchNorm2d(planes),)
        self.k2 = nn.Sequential(nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r),
                    nn.Conv2d(planes, planes, kernel_size=3, padding = 1, bias=False),
                    nn.BatchNorm2d(planes), )
        self.k3 = nn.Sequential(nn.Conv2d(planes, planes, kernel_size=3, padding = 1, bias=False),
                    nn.BatchNorm2d(planes),)
        self.k4 = nn.Sequential(nn.Conv2d(planes, planes, kernel_size=3,  padding = 1, bias=False),
                    nn.BatchNorm2d(planes),)
        self.conv1_a = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1_a = nn.BatchNorm2d(planes)
        self.conv1_b = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1_b = nn.BatchNorm2d(planes)

    def forward(self, x):

        out_a = self.conv1_a(x)
        identity = out_a
        out_a = self.bn1_a(out_a)
        out_b = self.conv1_b(x)
        out_b = self.bn1_b(out_b)
        out_a = F.relu(out_a)
        out_b = F.relu(out_b)

        out = torch.sigmoid(torch.add(identity, F.interpolate(self.k2(out_a), identity.size()[2:])))
        out = torch.mul(self.k3(out_a), out)
        out1 = self.k4(out)
        out2 = self.k1(out_b)
        out = torch.cat((out1,out2),1)
        return out

class External_attention(nn.Module):

    def __init__(self, c):
        super(External_attention, self).__init__()
        
        self.conv1 = nn.Conv2d(c, c, 1)
        self.k = c//4
        self.linear_0 = nn.Conv1d(c, self.k, 1, bias=False)
        self.linear_1 = nn.Conv1d(self.k, c, 1, bias=False)
        # self.linear_1.weight = self.linear_0.weight.permute(1, 0, 2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c))        

        self.relu = nn.ReLU()

    def forward(self, x):
        idn = x
        x = self.conv1(x)
        b, c, h, w = x.size()
        x = x.view(b, c, h*w)
        attn = self.linear_0(x)
        attn = F.softmax(attn, dim=-1)
        attn = attn / (1e-9 + attn.sum(dim=1, keepdims=True))
        x = self.linear_1(attn)
        x = x.view(b, c, h, w)
        x = self.conv2(x)
        x = x + idn
        x = self.relu(x)
        return x

class Classifier(nn.Module):
    def __init__(self,  hidden_size, num_classes):
        super(Classifier, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(hidden_size*2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        out = self.fc2(F.relu(self.fc1(x)))
        return out


class Net(nn.Module):
    def __init__(self, hsi_channels, sar_channels, hidden_size, block, num_parallel, num_reslayer=2, num_classes=7, bn_threshold=2e-2):
        self.planes = hidden_size
        self.num_parallel = num_parallel
        self.expansion = 2

        super(Net, self).__init__()

        self.conv_00 = nn.Sequential(nn.Conv2d(hsi_channels, hidden_size, 1, bias=False),
            nn.BatchNorm2d(hidden_size))
        self.conv_11 = nn.Sequential(nn.Conv2d(sar_channels, hidden_size, 1, bias=False),
            nn.BatchNorm2d(hidden_size))

        self.conv1 = ModuleParallel(nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=0, bias=False))
        self.bn1 = BatchNorm2dParallel(hidden_size, num_parallel)
        self.relu = ModuleParallel(nn.ReLU(inplace=True))
        self.layer = self._make_layer(block, hidden_size, num_reslayer, bn_threshold)

        self.classifier = Classifier(hidden_size , num_classes)
        self.Attention = External_attention(hidden_size*2)
        self.SCConv = SCConv(hidden_size*2, hidden_size)
        self.alpha = nn.Parameter(torch.ones(num_parallel, requires_grad=True))
        self.register_parameter('alpha', self.alpha)
        
    def _make_layer(self, block, planes, num_blocks, bn_threshold, stride=1):
        layers = []
        layers.append(block(planes, self.expansion, self.num_parallel, bn_threshold, stride))
        for i in range(1, num_blocks):
            layers.append(block(planes, planes, self.num_parallel, bn_threshold))
        return nn.Sequential(*layers)

    def forward(self, x, y):

        x = F.relu(self.conv_00(x)).unsqueeze(0)
        y = F.relu(self.conv_11(y)).unsqueeze(0)
        x = torch.cat((x, y), 0)
        x = self.relu(self.bn1(self.conv1(x)))
        out = self.layer(x)

        ens = 0
        alpha_soft = F.softmax(self.alpha, dim=0)
        for l in range(self.num_parallel):
            ens += alpha_soft[l] * out[l].detach()
        out.append(ens)

        x = torch.cat((out[0], out[1]), dim=1)
        x = self.SCConv(self.Attention(x))

        out = self.classifier(x)

        return out, alpha_soft

def calc_loss(outputs, labels):

    # criterion = nn.CrossEntropyLoss()
    criterion = Cross_fusion_CNN_Loss()
    loss = criterion(outputs, labels)
    return loss

def L1_penalty(var):
    return torch.abs(var).sum()


if __name__ == '__main__':
    # sar_bands = 4
    # num_class = 7

    # hsi_path = '../data_sar/Augsburg/augsburg_hsi.mat'
    # sar_path = '../data_sar/Augsburg/augsburg_sar.mat'
    # gt_path = '../data_sar/Augsburg/augsburg_gt.mat'
    # index_path = '../data_sar/Augsburg/augsburg_index.mat'

    # train_loader, test_loader, trntst_loader, all_loader = createAugsburgDataLoader(
    #     hsi_path, sar_path, gt_path, index_path, 30, 1, 128, 0)
    

    sar_bands = 4
    num_class = 8
    
    hsi_path = '../data_sar/Berlin/berlin_hsi.mat'
    sar_path = '../data_sar/Berlin/berlin_sar.mat'
    gt_path = '../data_sar/Berlin/berlin_gt.mat'
    index_path = '../data_sar/Berlin/berlin_index.mat'

    train_loader, test_loader, trntst_loader, all_loader = createBerlinDataLoader(
        hsi_path, sar_path, gt_path, index_path, 30, 1, 128, 0)
    
    # sar_bands = 1
    # num_class = 20

    # hsi_path = '../dataset/Houston2018/houston_hsi.mat'
    # lidar_path = '../dataset/Houston2018/houston_lidar.mat'
    # gt_path = '../dataset/Houston2018/houston_gt.mat'
    # index_path = '../dataset/Houston2018/houston_index.mat'

    # train_loader, test_loader, trntst_loader, all_loader = createHouston2018DataLoader(
    #     hsi_path, lidar_path, gt_path, index_path, 30, 11, 128, 0)

    hsi_bands = 30
    hidden_size = 128
    num_reslayer = 2
    num_parallel = 2
    bn_threshold = 0.002

    device = torch.device("cuda:0")
    # net = Net(hsi_bands, sar_bands, hidden_size, Bottleneck, num_parallel, num_reslayer, num_class, bn_threshold).to(device)
    # net = S2ENet(hsi_bands, sar_bands, num_class, 11).to(device)
    net = Cross_fusion_CNN(hsi_bands, sar_bands, num_class).to(device)
    # net = torch.load('../model/Augsburg_TBCNN_0.8453948641223737.pth')
    # re_path = '../report/augsburg_TBCNN.txt'
    # pic_path = '../pic/augsburg_TBCNN.png'
    # createAugsburgReport(net, test_loader, re_path, device)
    # drawAugsburg(net, trntst_loader, pic_path, device)

    net = torch.load('../model/Berlin_TBCNN_0.6760492020153686.pth')
    re_path = '../report/berlin_TBCNN.txt'
    pic_path = '../pic/berlin_TBCNN.png'
    pic_path2 = '../pic/berlin_TBCNN_.png'
    createBerlinReport(net, test_loader, re_path, device)
    drawBerlin(net, all_loader, pic_path, device)
    drawBerlin(net, trntst_loader, pic_path2, device)

    # net = torch.load('./model/houston2018_Endnet_0.9128016404485477.pth')
    # re_path = './report/houston2018_Endnet.txt'
    # pic_path = './pic/houston2018_Endnet_.png'
    # createHouston2018Report(net, test_loader, re_path, device)
    # drawHouston2018(net, trntst_loader, pic_path, device)
    
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.00005, momentum=0.9, weight_decay=0.0005)
    # optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    # max_accuracy = 0
    # model_path = '../model/'


    # for epoch in range(100):
    #     net.train()
    #     for hsi, sar, labels in train_loader:
    #         # hsi = hsi.squeeze(1)
    #         hsi = hsi.to(device)
    #         sar = sar.to(device)
    #         labels = labels.to(device)
    #         optimizer.zero_grad()
    #         # outputs, _ = net(hsi, sar)
    #         outputs = net(hsi, sar)
    #         loss = calc_loss(outputs, labels)
    #         loss.backward()
    #         optimizer.step()
        
    #     net.eval()
    #     count = 0
    #     for hsi, sar, labels in test_loader:
    #         # hsi = hsi.squeeze(1)
    #         hsi = hsi.to(device)
    #         sar = sar.to(device)
    #         outputs = net(hsi, sar)
    #         outputs = np.argmax(outputs[0].detach().cpu().numpy(), axis=1)
    #         if count == 0:
    #             pred = outputs
    #             gt = labels
    #             count = 1
    #         else:
    #             pred = np.concatenate((pred, outputs))
    #             gt = np.concatenate((gt, labels))
    #     accuracy = accuracy_score(gt, pred)
    #     # if accuracy > 0.90 and accuracy < 0.915:
    #     if accuracy>max_accuracy:
    #         torch.save(net, model_path + 'Berlin_TBCNN_' + str(accuracy) + '.pth')
    #         max_accuracy = accuracy
    #     print('[Epoch: %d] [current loss: %.4f]' %
    #           (epoch + 1, loss.item()), ' acc: ', accuracy)
    # # print('max_accuracy:', max_accuracy)

    # print('Finished Training')
