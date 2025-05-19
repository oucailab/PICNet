from net import *
from dataset import *
from report import *
from visualization import *
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import parameter

parameter._init()


def t_sne(model, test_loader, dataset):
    """Validation and get the metric
    """
    epoch_losses, epoch_accuracy = 0.0, 0.0
    criterion = torch.nn.CrossEntropyLoss()
    houston2018_color_map = [
        [50, 205, 51],
        [173, 255, 48],
        [0, 128, 129],
        [34, 139, 34],
        [46, 79, 78],
        [139, 69, 18],
        [0, 255, 255],
        [100, 100, 100],  # 255,255,255改成100
        [211, 211, 211],
        [254, 0, 0],
        [169, 169, 169],
        [105, 105, 105],
        [139, 0, 1],
        [200, 100, 0],  #####
        [254, 165, 0],
        [255, 255, 0],
        [218, 165, 33],
        [255, 0, 254],
        [0, 0, 254],
        [63, 224, 208]
    ]

    berlin_color_map = [[26, 163, 25], [216, 216, 216], [216, 89, 89], [
        0, 204, 51], [204, 153, 52], [244, 231, 1], [204, 102, 204], [0, 53, 255]]

    augsburg_color_map = [[26, 163, 25], [216, 216, 216], [216, 89, 89], [
        0, 204, 51], [244, 231, 1], [204, 102, 204], [0, 53, 255]]

    trento_color_map = [[0, 47, 255], [0, 223, 255], [
        143, 255, 111], [255, 207, 0], [255, 31, 0], [127, 0, 0]]

    # Houston2013 color map
    houston2013_color_map = [[0, 0, 131], [0, 0, 203], [0, 19, 255], [0, 91, 255], [0, 167, 255], [0, 239, 255], [55, 255, 199], [
        131, 255, 123], [203, 255, 51], [255, 235, 0], [255, 163, 0], [255, 87, 0], [255, 15, 0], [199, 0, 0], [127, 0, 0]]

    feature_list = []
    gt_list = []
    count = 0
    device = torch.device('cuda:0')
    with torch.no_grad():
        for batch_idx, (hsi_pca, lidar, tr_labels) in enumerate(test_loader):
            # hsi = hsi.to(device)
            # hsi = hsi[:, 0, :, :, :]
            lidar = lidar.to(device)
            hsi_pca = hsi_pca.to(device)
            tr_labels = tr_labels.to(device)

            feature, _, _, _, _ = model(hsi_pca, lidar)
            tr_labels = tr_labels.detach().cpu().numpy().astype(int)
            # print(tr_labels)
            # print(type(tr_labels))

            feature = feature.detach().cpu().numpy()
            # print(feature.shape)
            # print(type(tr_labels))
            houston2018_color_map = np.array(houston2018_color_map)
            berlin_color_map = np.array(berlin_color_map)
            augsburg_color_map = np.array(augsburg_color_map)
            trento_color_map = np.array(trento_color_map)
            houston2013_color_map = np.array(houston2013_color_map)
            feature_list.append(feature[0])
            if dataset == "Houston2018":
                gt_list.append(houston2018_color_map[tr_labels[0]] * 1.0 / 255.0)
            elif dataset == "Berlin":
                gt_list.append(berlin_color_map[tr_labels[0]] * 1.0 / 255.0)
            elif dataset == "Augsburg":
                gt_list.append(augsburg_color_map[tr_labels[0]] * 1.0 / 255.0)
            elif dataset == "Trento":
                gt_list.append(trento_color_map[tr_labels[0]] * 1.0 / 255.0)
            elif dataset == "Houston2013":
                gt_list.append(houston2013_color_map[tr_labels[0]] * 1.0 / 255.0)
            else:
                print(dataset + " TSNE Dateset not find!")

    tsne = TSNE(n_components=2, perplexity=10, learning_rate=100)
    features = np.array(feature_list)
    features = features.reshape(features.shape[0], -1)
    features_tsne = tsne.fit_transform(features)
    plt.xlim(-100, 100)
    plt.ylim(-100, 100)
    # 清空当前绘图
    plt.clf()    
    plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=gt_list, s=10)
    plt.tick_params(labelsize=9)
    # plt.show()
    oa_str = str(parameter.get_value("oa"))
    save_name = '../pic/' + dataset + '_tsne_' + oa_str + '.png'
    plt.savefig(save_name)
    return


def getMyTSNE(datasetType, model, test_loder):
    dataset = ["Houston2013", "Houston2018", "Trento", "Berlin", "Augsburg"][datasetType]
    t_sne(model, test_loder, dataset)
    print(f'{dataset} TSNE pic Success!')


def myTest(datasetType):
    cuda = parameter.get_value('cuda')
    if cuda == 'cuda0':
        device = torch.device("cuda:0")
    elif cuda == 'cuda1':
        device = torch.device("cuda:1")
    elif cuda == 'cuda2':
        device = torch.device("cuda:2")

    channels = parameter.get_value('channels')
    windowSize = parameter.get_value('windowSize')

    batch_size = parameter.get_value('batch_size')
    num_workers = parameter.get_value('num_workers')
    random_seed = parameter.get_value('random_seed')
    visualization = parameter.get_value('visualization')
    report = parameter.get_value('report')
    tsne = parameter.get_value('tsne')

    model_savepath = parameter.get_value('model_savepath')[datasetType]
    report_path = parameter.get_value('report_path')
    image_path = parameter.get_value('image_path')

    # model_savepath = '../model/Trento_model_97.4673202614379.pth'

    net = torch.load(model_savepath)
    train_loader, test_loader, trntst_loader, all_loader = getMyData(datasetType, channels, windowSize, batch_size,
                                                                     num_workers)

    set_random_seed(random_seed)
    if report:
        getMyReport(datasetType, net, test_loader, report_path[datasetType], device)
    if tsne:
        getMyTSNE(datasetType, net, test_loader)
    if visualization:
        getMyVisualization(datasetType, net, trntst_loader, image_path[datasetType], device)


if __name__ == "__main__":
    parameter.set_value('cuda', 'cuda0')
    # Houston2013
    # datasetType = 0

    # Houston2018
    # datasetType = 1

    # Trento
    datasetType = 2

    # Berlin
    # datasetType = 3

    # Augsburg
    # datasetType = 4
    parameter.set_value('report', True)
    parameter.set_value('visualization', False)
    parameter.set_value('tsne', True)
    myTest(datasetType)
