
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from funcs import get_dict_from_class
from models import FeatureExtractor, FTWithLocalization
from losses import BCELoss

from torch.utils.data import DataLoader
import pandas as pd
from config import model,model_param,pre_trained_model,data_loader_param,data_loader,root

def plot_channels(data, loc="", vmin=0, vmax=1, name_prefix="", aspect=1):
    assert len(data.shape) == 3

    fig = plt.figure()
    image_in_row = int(np.sqrt(data.shape[0])) + 1
    channel = data.shape[0]
    if channel==1:plt.imshow(data[0], vmin=vmin, vmax=vmax,aspect=aspect)
    else:
        for i1 in range(channel):
            fig.add_subplot(image_in_row, image_in_row, i1 + 1)
            plt.imshow(data[i1], vmin=vmin, vmax=vmax)
    fig.suptitle(str(vmin) + "_" + str(vmax), fontsize=16)
    fig.savefig(loc + name_prefix + str(channel) + '.png', bbox_inches='tight')
    plt.close(fig)
def get_layer_output(model=model):
    pre_trained_model = root+"/codes/fold0/checkpoints/last.pth"
    model = model(**get_dict_from_class(model_param))
    num_layer = len(model.conv_blocks)
    if True:
        checkpoint = torch.load(pre_trained_model)['model_state_dict']
        model.load_state_dict(checkpoint)
        model.eval()




    d = DataLoader(data_loader( **get_dict_from_class(data_loader_param)),
                                batch_size=200,
                                shuffle=False,
                                num_workers=4,
                                pin_memory=True,
                                drop_last=False)
    # min max finder
    for dict_ in tqdm(d):
        with torch.no_grad():
            min_max=[]
            fig = plt.figure()

            for i in range(1,num_layer+1):
                model.num_blocks = i
                if i <num_layer:
                    data= model.cnn_feature_extractor(dict_['image_pixels']/255)
                else:
                    data = model(dict_['image_pixels']/255 )[:]



                min_max.append([torch.min(data),torch.max(data)])
                data = np.array([min_max[-1][0] + j * (min_max[-1][1] - min_max[-1][0]) / 8 for j in range(9)])
                plt.imshow(data.reshape((3, 3)), aspect='auto')
                plt.title(str(min_max[-1][0]) + "_" + str(min_max[-1][1]))
                fig.savefig( str(root)+'/diagnostics/'+'z_'+str(i)+'_reference.png')
            plt.close()
            break


    d = DataLoader(data_loader( **get_dict_from_class(data_loader_param)),
                   batch_size=10,
                   shuffle=False,
                   num_workers=1,
                   pin_memory=True,
                   drop_last=False)
    k=0
    for dict_ in tqdm(d):
        with torch.no_grad():
                for i in range(1, num_layer+1):
                    model.num_blocks = i
                    if i <num_layer:
                        data = model.cnn_feature_extractor(dict_['image_pixels'] /255)
                        aspect=1
                    else:
                        data =model(dict_['image_pixels']/255 )[:]
                        data=data[0].reshape((1,1,1,1))
                        aspect = None
                    # print(i)
                    # plot_channels(data[0],loc="/home/pooja/PycharmProjects/digitRecognizer/weightDist/layer_1mages/",vmin=min_max[0][0], vmax=min_max[0][1],name_prefix=str(i)+"_")
                    # data = model(dict_['image_pixels'] / 255)

                    plot_channels(data[0], loc= str(root)+'/diagnostics/',
                              vmin=min_max[i-1][0], vmax=min_max[i-1][1], name_prefix=str(k)+"_"+str(i) + "_",aspect=aspect)

                k=k+1
                if k>0:break

if __name__ == "__main__":
    get_layer_output()