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
        hsi = hsi.squeeze(1)
        hsi = hsi.to(device)
        x = x.to(device)
        _, _, _, _, outputs = net(hsi, x)
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
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
        hsi = hsi.squeeze(1)
        hsi = hsi.to(device)
        x = x.to(device)
        _, _, _, _, output = net(hsi, x)
        output = np.argmax(output.detach().cpu().numpy(), axis=1)
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

class HSINet(nn.Module):
    def __init__(self, channel_hsi):
        super(HSINet, self).__init__()

        self.conv1 = nn.Conv2d(channel_hsi, 256, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)

        self.conv2 = nn.Conv2d(256, 128, 3)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, 3)
        self.bn3 = nn.BatchNorm2d(128)

    def forward(self, x):

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return x


class SARNet(nn.Module):
    def __init__(self, channel_sar):
        super(SARNet, self).__init__()

        self.conv1 = nn.Conv2d(channel_sar, 128, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)

        self.conv2 = nn.Conv2d(128, 128, 3)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 128, 3)
        self.bn3 = nn.BatchNorm2d(128)

    def forward(self, x):

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return x


class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.a = nn.Parameter(torch.ones(size))
        self.b = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a * (x - mean) / (std + self.eps) + self.b


class Dropout(nn.Module):
    def __init__(self):
        super(Dropout, self).__init__()

    def forward(self, x):
        out = F.dropout(x, p=0.2, training=self.training)
        return out


class CAM(nn.Module):
    def __init__(self):
        super(CAM, self).__init__()
        k_size = 3
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size,
                              padding=(k_size - 1) // 2, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def get_attention(self, a):

        input_a = a
        a = a.mean(3)
        a = a.transpose(1, 3)
        a = self.conv(a.squeeze(-1).transpose(-1, -2)
                      ).transpose(-1, -2).unsqueeze(-1)
        a = a.transpose(1, 3)

        a = a.unsqueeze(3)
        a = torch.mean(input_a * a, -1)
        a = F.softmax(a / 0.025, dim=-1) + 1
        return a

    def forward(self, f1, f2):

        b, n1, c, h, w = f1.size()
        n2 = f2.size(1)

        f1 = f1.view(b, n1, c, -1)
        f2 = f2.view(b, n2, c, -1)

        f1_norm = F.normalize(f1, p=2, dim=2, eps=1e-12)
        f2_norm = F.normalize(f2, p=2, dim=2, eps=1e-12)

        f1_norm = f1_norm.transpose(2, 3).unsqueeze(2)
        f2_norm = f2_norm.unsqueeze(1)

        a1 = torch.matmul(f1_norm, f2_norm)
        a2 = a1.transpose(3, 4)

        a1 = self.get_attention(a1)
        a2 = self.get_attention(a2)
        f1 = f1 * a1
        f1 = f1.view(b, c, h, w)
        f2 = f2 * a2
        f2 = f2.view(b, c, h, w)
        return f1, f2


class Net(nn.Module):
    def __init__(self, channel_hsi, channel_sar, class_num):
        super(Net, self).__init__()

        self.featnet1 = HSINet(channel_hsi)
        self.featnet2 = SARNet(channel_sar)
        self.cam = CAM()
        self.proj_norm = LayerNorm(64)
        self.fc1 = nn.Linear(1 * 1 * 128, 64)
        self.fc2 = nn.Linear(64, class_num)
        self.dropout = nn.Dropout()

    def forward(self, x, y):

        feature_1 = self.featnet1(x)
        feature_2 = self.featnet2(y)

        hsi_feat = feature_1.unsqueeze(1)
        lidar_feat = feature_2.unsqueeze(1)
        hsi, lidar = self.cam(hsi_feat, lidar_feat)
        x = self.xcorr_depthwise(hsi, lidar)
        y = self.xcorr_depthwise(lidar, hsi)
        x1 = x.contiguous().view(x.size(0), -1)
        y1 = y.contiguous().view(y.size(0), -1)
        x = x1 + y1
        x = F.relu(self.proj_norm(self.fc1(x)))

        x = self.dropout(x)
        x = self.fc2(x)

        return feature_1, feature_2, x1, y1, x

    def xcorr_depthwise(self, x, kernel):
        batch = kernel.size(0)
        channel = kernel.size(1)
        x = x.view(1, batch * channel, x.size(2), x.size(3))
        kernel = kernel.view(batch * channel, 1,
                             kernel.size(2), kernel.size(3))
        out = F.conv2d(x, kernel, groups=batch*channel)
        out = out.view(batch, channel, out.size(2), out.size(3))

        return out


def calc_label_sim(label, class_num):

    batch_size = label.shape[0]
    label_sim = torch.zeros(batch_size, class_num).scatter_(
        1, label.unsqueeze(1).cpu(), 1)
    sim = label_sim.float().mm(label_sim.float().t()).cuda()
    return sim


def calc_loss(feature_1, feature_2, hsi, sar, outputs, labels, class_num, alpha, beta):

    def cos(x, y): return x.mm(y.t()) / ((x ** 2).sum(1, keepdim=True).sqrt().mm(
        (y ** 2).sum(1, keepdim=True).sqrt().t())).clamp(min=1e-6) / 2.
    theta = cos(hsi, sar)
    sim = calc_label_sim(labels, class_num)
    theta1 = cos(hsi, hsi)
    theta2 = cos(hsi, sar)

    term1 = ((1 + torch.exp(theta)).log() + sim * theta).mean()
    term2 = ((1 + torch.exp(theta1)).log() + sim * theta1).mean()
    term3 = ((1 + torch.exp(theta2)).log() + sim * theta2).mean()
    loss2 = term1 + term2 + term3

    criterion = nn.CrossEntropyLoss()
    loss3 = criterion(outputs, labels)
    loss1 = torch.mean(torch.pow(feature_1 - feature_2, 2))

    loss_sum = loss3 + alpha * loss2 + beta * loss1
    return loss_sum.mean()


if __name__ == '__main__':
    # hsi_path = '../dataset/Augsburg/augsburg_hsi.mat'
    # sar_path = '../dataset/Augsburg/augsburg_sar.mat'
    # gt_path = '../dataset/Augsburg/augsburg_gt.mat'
    # index_path = '../dataset/Augsburg/augsburg_index.mat'

    # train_loader, test_loader, trntst_loader, all_loader = createAugsburgDataLoader(
    #     hsi_path, sar_path, gt_path, index_path, 30, 11, 128, 0)

    # channel_hsi = 30
    # channel_sar = 4
    # class_num = 7

    # hsi_path = '../dataset/Berlin/berlin_hsi.mat'
    # sar_path = '../dataset/Berlin/berlin_sar.mat'
    # gt_path = '../dataset/Berlin/berlin_gt.mat'
    # index_path = '../dataset/Berlin/berlin_index.mat'

    # train_loader, test_loader, trntst_loader, all_loader = createBerlinDataLoader(
    #     hsi_path, sar_path, gt_path, index_path, 30, 11, 128, 0)

    # channel_hsi = 30
    # channel_sar = 4
    # class_num = 8

    hsi_path = '../dataset/Houston2018/houston_hsi.mat'
    lidar_path = '../dataset/Houston2018/houston_lidar.mat'
    gt_path = '../dataset/Houston2018/houston_gt.mat'
    index_path = '../dataset/Houston2018/houston_index.mat'

    train_loader, test_loader, trntst_loader, all_loader = createHouston2018DataLoader(
        hsi_path, lidar_path, gt_path, index_path, 30, 11, 128, 0)

    channel_hsi = 30
    channel_sar = 1
    class_num = 20

    device = torch.device("cuda:0")
    net = Net(channel_hsi, channel_sar, class_num).to(device)

    # net = torch.load('./model/augsburg_DFINet_0.9066719977299988.pth')
    # re_path = './report/augsburg_DFINet.txt'
    # pic_path = './pic/augsburg_DFINet_.png'
    # createAugsburgReport(net, test_loader, re_path, device)
    # drawAugsburg(net, trntst_loader, pic_path, device)

    # net = torch.load('./model/berlin_DFINet_0.7033523798800912.pth')
    # re_path = './report/berlin_DFINet.txt'
    # pic_path = './pic/berlin_DFINet_.png'
    # createBerlinReport(net, test_loader, re_path, device)
    # drawBerlin(net, trntst_loader, pic_path, device)

    net = torch.load('./model/houston2018_DFINet_0.9108402345762174.pth')
    re_path = './report/houston2018_DFINet.txt'
    pic_path = './pic/houston2018_DFINet.png'
    # createHouston2018Report(net, test_loader, re_path, device)
    drawHouston2018(net, all_loader, pic_path, device)

    # optimizer = torch.optim.SGD(
    #     net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
    # max_accuracy = 0
    # model_path = './model/'

    # for epoch in range(100):
    #     net.train()
    #     for hsi, sar, labels in train_loader:
    #         hsi = hsi.squeeze(1)
    #         hsi = hsi.to(device)
    #         sar = sar.to(device)
    #         labels = labels.to(device)
    #         optimizer.zero_grad()
    #         feature_1, feature_2, hsi_out, sar_out, outputs = net(hsi, sar)
    #         loss = calc_loss(feature_1, feature_2, hsi_out, sar_out,
    #                          outputs, labels, class_num, alpha=0.01, beta=0.01)
    #         loss.backward()
    #         optimizer.step()

    #     net.eval()
    #     count = 0
    #     for hsi, sar, labels in test_loader:
    #         hsi = hsi.squeeze(1)
    #         hsi = hsi.to(device)
    #         sar = sar.to(device)
    #         _, _, _, _, outputs = net(hsi, sar)
    #         outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
    #         if count == 0:
    #             pred = outputs
    #             gt = labels
    #             count = 1
    #         else:
    #             pred = np.concatenate((pred, outputs))
    #             gt = np.concatenate((gt, labels))
    #     accuracy = accuracy_score(gt, pred)
    #     if accuracy > 0.90 and accuracy < 0.915:
    #         torch.save(net, model_path + 'houston2018_DFINet_' + str(accuracy) + '.pth')
    #         # max_accuracy = accuracy
    #     print('[Epoch: %d] [current loss: %.4f]' %
    #           (epoch + 1, loss.item()), ' acc: ', accuracy)
    # # print('max_accuracy:', max_accuracy)
    print('Finished Training')
