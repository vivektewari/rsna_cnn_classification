import os, importlib, sys

import pandas as pd
from model_configs  import Model1,Model2,Model3,Model4,Model5

from config import *
import config, train, inference , start
temp = pd.read_csv(metricSheetPath,index_col='index')
temp = temp.drop(temp.index)
temp.to_csv(metricSheetPath)

def iterator(directory,extra):
    os.chdir(directory)
    importlib.reload(sys.modules['config'])
    counter = 0
    for model_ in [Model1,Model2,Model3,Model4,Model5]:
        model=model_
        dataload = config.DataLoad1
        model.input_image_dim, dataload.reshape_pixel = extra[0],extra[0]
        dataload.pixel_col = ['pixel' + str(i) for i in
                              range(dataload.reshape_pixel[0] * dataload.reshape_pixel[1])]
        train.train(model, dataload)
        inference.get_inference(model, dataload,config.holdData,str(counter)+directory)



base = '/home/pooja/PycharmProjects/digitRecognizer/'
dirs=['','rough/shift/','rough/scale/','rough/shiftScale/']
#dirs=['rough/shift/']
input_image_size=[(28,28),(28*4,28*4),(28*4,28*4),(28*4*4,28*4*4)]
for i in range(len(dirs)):
    print(dirs[i])
    iterator(base+dirs[i],[input_image_size[i]])

