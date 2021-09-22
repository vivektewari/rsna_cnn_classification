

import numpy as np
from funcs import get_dict_from_class, updateMetricsSheet
from models import FeatureExtractor
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
from config import *

#model = model(**get_dict_from_class(model_param))
checkpoint = torch.load(pre_trained_model)['model_state_dict']


chan = [10, 10]
side = [(8,8),( 10,9)]

for j in range(2):
    weightMatrix = checkpoint['conv_blocks.' + str(j) + '.conv1.weight']
    _min,_max=torch.min(weightMatrix),torch.max(weightMatrix)
    fig = plt.figure()
    data=np.array([_min+i*(_max-_min)/8 for i in range(9)])
    plt.imshow(data.reshape((3,3)), aspect='auto', vmin=_min, vmax=_max)
    plt.autoscale('False')
    plt.title(str(_min)+"_"+str(_max))
    fig.savefig('/home/pooja/PycharmProjects/digitRecognizer/weightDist/0_reference' + str(j) + '.png')
    plt.close(fig)
    for i in range(chan[j]):

        data = weightMatrix[i].flatten(start_dim=1, end_dim=-1)
        data = data.reshape(side[j])
        fig = plt.figure()
        plt.imshow(data,aspect='auto', vmin = _min, vmax = _max)
        plt.autoscale('False')
        fig.savefig('/home/pooja/PycharmProjects/digitRecognizer/weightDist/conv'+str(j)+'_'+str(i)+'.png')
        plt.close(fig)
# for i in range(30):
#     fig = plt.figure()
#     plt.imshow(torch.reshape(data[i],shape=(28,28)),aspect='auto')
#     fig.savefig('/home/pooja/PycharmProjects/digitRecognizer/weightDist/fc1_'+str(i)+'.png')
#     plt.close(fig)
#
# data=torch.transpose(weightMatrix2,0,1)
# fig = plt.figure()
# plt.imshow(data)
# fig.savefig('/home/pooja/PycharmProjects/digitRecognizer/weightDist/fc2_.png')
# plt.close(fig)
#
# #model.load_state_dict(checkpoint, strict=True)
# d=1