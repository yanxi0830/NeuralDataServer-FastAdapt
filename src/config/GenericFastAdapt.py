batch_size = 128

config = {}
data_test_opt = {}
data_test_opt['batch_size'] = batch_size
data_test_opt['epoch_size'] = None
data_test_opt['dataset_name'] = 'generic'
data_test_opt['split'] = 'train'

config['data_test_opt'] = data_test_opt

net_opt = {}
net_opt['num_classes'] = 4

networks = {}
networks['model'] = {'def_file': 'architectures/ResNet.py', 'pretrained': None, 'opt': net_opt}
config['networks'] = networks

config['networks'] = networks

criterions = {}
criterions['loss'] = {'ctype': 'CrossEntropyLoss', 'opt': None}
config['criterions'] = criterions
config['algorithm_type'] = 'ClassificationModel'
config['best_metric'] = 'prec1'
