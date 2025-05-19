def _init():  # 初始化
    global parameter
    parameter = {
        # net
        'channels': 35,
        'windowSize': 11,
        'out_features': [15, 20, 6, 8, 7],

        # train
        'device': 'cuda0',
        'lr': 0.00001,
        'epoch_nums': 10,
        'batch_size': 128,
        'num_workers': 0,
        'random_seed': 6,
        'visualization': False,
        'report': True,
        'tsne': False,
        'test_base': 1,
        'model_savepath': ['../model/Houston2013_model.pth',
                           '../model/Houston2018_model.pth',
                           '../model/Trento_model.pth',
                           '../model/Berlin_model.pth',
                           '../model/Augsburg_model.pth'],
        'log_path': ['../log/Houston2013_log.txt',
                     '../log/Houston2018_log.txt',
                     '../log/Trento_log.txt',
                     '../log/Berlin_log.txt',
                     '../log/Augsburg_log.txt'],
        'report_path': ['../report/Houston2013_report.txt',
                        '../report/Houston2018_report.txt',
                        '../report/Trento_report.txt',
                        '../report/Berlin_report.txt',
                        '../report/Augsburg_report.txt'],
        'image_path': ['../pic/Houston2013.png',
                       '../pic/Houston2018.png',
                       '../pic/Trento.png',
                       '../pic/Berlin.png',
                       '../pic/Augsburg.png'],

        'save_max_acc': [1.0, 1.0, 1.0, 1.0, 1.0]
    }


def set_value(key, value):
    # 定义一个全局变量
    parameter[key] = value


def get_value(key):
    # 获得一个全局变量，不存在则提示读取对应变量失败
    try:
        return parameter[key]
    except:
        print('读取' + key + '失败\r\n')


def get_taskInfo():
    return '-----------------------taskInfo----------------------- \n lr:\t{} \n epoch_nums:\t{} \n batch_size:\t{} \n window_size:\t{} \n channels:\t{} \n------------------------------------------------------'.format(
        parameter['lr'], parameter['epoch_nums'], parameter['batch_size'], parameter['windowSize'], parameter['channels'])
