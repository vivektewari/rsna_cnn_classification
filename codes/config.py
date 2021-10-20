import time
from pathlib import Path
import random
import os
from dataLoaders import *
from losses import *
from models import *
from param_options import *

#from funcs import *
root ='/home/pooja/PycharmProjects/rsna_cnn_classification/'
dataCreated = root+'/data/dataCreated/'
raw_data=root+ '/data/'
image_loc =dataCreated +'/mixed/'
blank_loc =dataCreated + '/auxilary/'


#data_loader
data_loader_param =rsna_param
data_loader_param.base_loc=image_loc
data_loader_param.blank_loc=blank_loc
data_loader_param.data_frame_path=dataCreated+'image_info/'+rsna_param.data_frame_path
data_loader_param.label=raw_data+rsna_param.label
data_loader = rsna_loader

#Model
model_param = Model1_nii
model =modelling_3d

#loss function
loss_func =BCELoss(loss_func=nn.BCELoss())


# metricSheetPath = root / 'metricSheet2.csv'
saveDirectory = root + '/outputs/weights/'
device = 'cpu'
config_id = str(os.getcwd()).split()[-1]
startTime = time.time()

lr = 0.05

epoch = 200

random.seed(23)


  #BCELoss pixel_shape=(28,28)


image_scale=None
pre_trained_model ="//home/pooja/Downloads/last.pth"
#pre_trained_model =root+"/outputs/weights/rsna_9.pth"

pre_trained_model ='/home/pooja/Downloads/rsna_20.pth'
#pre_trained_model =None
#'/home/pooja/PycharmProjects/digitRecognizer/rough/localization/fold0/checkpoints/train.17.pth'




