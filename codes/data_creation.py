
from config import *
from funcs import get_dict_from_class, updateMetricsSheet, DataCreation
from models import FeatureExtractor
from itertools import permutations,combinations
from tqdm import tqdm
from torch.utils.data import DataLoader
data_count=5000
looper=0
pixel = ['pixel' + str(i) for i in range(28*28*16)]


#data_load = data_loader(**get_dict_from_class(data_loader_param))
#data = data_load.data
#for i in range(784):
#data[pixel] = 0
#dataCreation = DataCreation(data_path=dataPath, image_path_=image_path)
# dataCreation.circles_and_rectngles(data)
data = pd.read_csv(str(dataPath) + '/train.csv')
# # dataCreation.shifter( data, data_count=data_count, size=(112, 112),size2=(28,28))
dataCreation = DataCreation(data_path='/home/pooja/PycharmProjects/digitRecognizer/rough/scale/data', image_path_='/home/pooja/PycharmProjects/digitRecognizer/rough/scale/images')
dataCreation.image_path = None
# # dataCreation.to_csv,dataCreation.image_path = False, None
data=dataCreation.scaler( data, data_count=data_count, size=(28*2, 28*2),size2=(28,28), scales=2)
dataCreation = DataCreation(data_path='/home/pooja/PycharmProjects/digitRecognizer/rough/shiftScale/data', image_path_='/home/pooja/PycharmProjects/digitRecognizer/rough/shiftScale/images')
dataCreation.image_path = None
data=dataCreation.shifter( data, data_count=data_count, size=(28*4, 28*4),size2=(28*2,28*2))
#darker(data)  not required
#dataPath='/home/pooja/PycharmProjects/digitRecognizer/rough/shiftScale/data'
#data = pd.read_csv(str(dataPath) + '/newData.csv')# to ut the above steps
dataCreation = DataCreation(data_path='/home/pooja/PycharmProjects/digitRecognizer/rough/localization/data', image_path_='/home/pooja/PycharmProjects/digitRecognizer/rough/localization/images')
dataCreation.create_localization( data, data_count=data_count, size=(28*4, 28*4))


