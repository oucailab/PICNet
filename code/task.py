from train import *
from test import *
import os

import parameter

parameter._init()

# 修改保存的模型名称，参数名+oa
def update_model_name(datasetType):
    model_savepath = parameter.get_value('model_savepath')[datasetType]
    oa_str = str(parameter.get_value("oa"))
    # 获取文件名和扩展名
    base, ext = os.path.splitext(model_savepath)
    # 构建新的文件名
    new_model_savepath = f"{base}_{oa_str}{ext}"
    # 重命名文件
    os.rename(model_savepath, new_model_savepath)


def myTask(lr, epoch_nums, datasetType):
    parameter.set_value('test_base', [1, 3, 1, 2, 1][datasetType])
    parameter.set_value('epoch_nums', epoch_nums)
    parameter.set_value('lr', lr)
    parameter.set_value('cuda', cuda)
    parameter.set_value('data_type', datasetType)
    myTrain(datasetType, net)
    myTest(datasetType)
    update_model_name(datasetType)


def task():
    # Houston2013
    myTask(0.0001, 200, 0)

    # Houston2018
    # myTask(0.0001, 130, 1)

    # Trento
    # myTask(0.0001, 200, 2)

    # Berlin
    # myTask(0.0001, 200, 3)

    # Augsburg
    # myTask(0.0001, 200, 4)

    # myTask(0.0001, 200, cuda, net, 3, False)
    # myTask(0.00001, 200, cuda, net, 3, False)


def channelsTask(window_size):
    print('Start channelsTask')
    parameter.set_value('windowSize', window_size)
    channels_list = [20, 25, 30, 35, 40]
    # channels_list = [25, 30, 35, 40]
    # channels_list = [30, 35, 40]
    # channels_list = [35, 40]
    # channels_list = [40]
    for channels in channels_list:
        parameter.set_value('channels', channels)
        task()
    print('End channelsTask')


def windowSizeTask(channels):
    print('Start windowSizeTask')
    parameter.set_value('channels', channels)
    window_size_list = [
        8,
        10,
        12,
        14,
        16,
    ]
    for window_size in window_size_list:
        parameter.set_value('windowSize', window_size)
        task()
    print('End windowSizeTask')


cuda = 'cuda0'
net = 'PICNet'

# channelsTask(8)
# channelsTask(16)

# windowSizeTask(20)
# windowSizeTask(25)
# windowSizeTask(30)
# windowSizeTask(35)
# windowSizeTask(40)
myTask(0.0001, 200, 4)