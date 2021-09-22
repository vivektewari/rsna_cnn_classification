import pandas as pd

from config import *
from funcs import toImage,updateMetricsSheet
import random
stage='holdOutSample' # 'randomModel'
if stage=='imageExtraction':
    toImage(dataPath / 'toImage')#dataPath / 'toImage'
elif stage=='holdOutSample':
    train = pd.read_csv(dataPath / 'newData.csv')
    train['index']=range(train.shape[0])
    orderedCols=list(train.columns[-1:])+list(train.columns[0:-1])
    train=train[orderedCols]
    breakParts=20
    rows=random.sample(range(train.shape[0]), int(train.shape[0]/breakParts))
    rows=list(set(rows)-set([i for i in range(100)]))# keepint this elemnet in training to visulize these lement
    remaining=list(set(list(range(train.shape[0])))-set(rows))
    train.iloc[rows].to_csv(dataCreated / 'holdout.csv',index=False)
    train.iloc[remaining].to_csv(dataCreated / 'dev.csv',index=False)
elif stage=='randomModel':
    dev=pd.read_csv(dataCreated / 'dev.csv')
    holdOut=pd.read_csv(dataCreated / 'holdout.csv')
    predDev=[random.randint(0,9) for i in range(dev.shape[0])]
    predHold = [random.randint(0, 9) for i in range(holdOut.shape[0])]
    updateMetricsSheet(dev['label'],predDev,holdOut['label'],predHold,modelName='random',force=True)





