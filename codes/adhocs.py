import pandas as pd
import numpy as np
import pydicom as di
import torch,time
from commonFuncs import packing
from funcs import DataCreation
from config import root,dataCreated
import itertools
import os,cv2
from os import listdir
from pathlib import Path
pth=str(root)+'/data/rsna-miccai-brain-tumor-radiogenomic-classification/train/'

if False:
    #making a dataframe for each image to keep infor for test_type and patirent_id

    test_type,patient_id,image_name,loc=[],[],[],[]
    for ro,dirs, files in os.walk(pth):
        for file in files:
            if file.endswith(".dcm"):
                temp=ro.split("/")
                test_type.append(temp[-1])
                patient_id.append(str(temp[-2]))
                image_name.append(file.split(".")[0])

    df=pd.DataFrame(data={'patient_id':patient_id,'test_type':test_type,'image_name':image_name})
    df.to_csv(dataCreated / 'image_info' / 'images0.csv')

if False:
    start=time.time()
    #adding .dcm non pixel information

    df0=pd.read_csv(dataCreated / 'image_info' / 'images0.csv',dtype='str')
    #df1=df1[0:8765]
    temp=df0.apply(lambda row: pth+str(row['patient_id'])+"/"+row['test_type']+"/"+row['image_name']+".dcm",axis=1)
    Images1 = []
    image_vars=['coords','ImageOrientationPatient','SliceLocation','PhotometricInterpretation','PixelSpacing','SamplesPerPixel']
    list_=[[] for i in range(len(image_vars))]
    d=DataCreation()
    loop=0
    df=df0[loop:loop+1000]
    for k in list(temp):

        loop+=1
        Images = di.read_file(k, force=True)
        data = Images.pixel_array
        data=np.where(data>0,1,0)
        list_[0].append(packing.pack(d.coords(data)))



        for i in range(1,len(image_vars)):
            try:
                list_[i].append(packing.pack(Images[image_vars[i]].value))
            except :
                list_[i].append("error")


        if loop%1000==0 or loop==df1.shape[0]:
            print(loop,time.time()-start)
            for i in range(len(image_vars)):
                df[image_vars[i]] = list_[i]
            if loop==1000:df.to_csv(dataCreated / 'image_info' / 'images1.csv')
            else :df.to_csv(dataCreated / 'image_info' / 'images1.csv', mode='a', header=False)
            list_ = [[] for i in range(len(image_vars))]
            df = df0[loop:min(df0.shape[0]+1,loop + 1000)]

if False:
    #deriving variable on image based on variables

    df0 = pd.read_csv(dataCreated / 'image_info' / 'images1.csv', dtype='str')
    temp=df0['coords'].apply(lambda x:packing.unpack(x))
    coords=torch.tensor(list(temp))
    df0['area']=(coords[:,2]-coords[:,0])*(coords[:,3]-coords[:,1])
    temp = df0['ImageOrientationPatient'].apply(lambda x: packing.unpack(x))
    orient= np.array(list(temp))
    # ['1', '0', '0', '0', '0', '-1']-Coronal plane view
    #
    # ['0', '1', '0', '0', '0', '-1']=Sagittal plane view
    #
    # ['1', '0', '0', '0', '1', '0']-Axial plane view
    orient=np.where(np.abs(orient-0)>0.1,orient,0)
    orient = np.where(np.abs(orient - 1) > 0.1, orient, 1)
    orient = np.where(np.abs(orient -(-1)) > 0.1, orient, -1)
    coronal=np.all(np.abs(orient-np.array([1,0,0,0,0,-1]))<0.01,axis=1)
    sagittal= np.all(np.abs(orient - np.array([0, 1, 0, 0, 0, -1]))<0.01,axis=1)
    axial= np.all(np.abs(orient - np.array([1, 0, 0, 0, 1, 0]))<0.01,axis=1)
    df0['plane']='undef'
    loop=0
    pl=['coronal','sagittal','axial']
    for p in [coronal,sagittal,axial]:
        df0['plane'][p]=pl[loop]
        loop+=1
    drops= [col for col in df0.columns if col.find('Unnamed') >=0]
    df=df0.drop(drops,axis=1)
    df.to_csv(dataCreated / 'image_info' / 'images2.csv')


if True:
    root = '/home/pooja/PycharmProjects/rsna_cnn_classification/'
    dataCreated = root + '/data/dataCreated/'
    blank_loc = dataCreated + '/auxilary/'
    output=blank_loc+'/blank.png'
    im=np.zeros((256,256))
    cv2.imwrite(output,im )

if False:
    #adding array shape, max ,min 10,90 percentile of pixel value
    start = time.time()
    df0=pd.read_csv(dataCreated / 'image_info' / 'images0.csv',dtype='str')
    #df1=df1[0:8765]
    temp=df0.apply(lambda row: pth+str(row['patient_id'])+"/"+row['test_type']+"/"+row['image_name']+".dcm",axis=1)
    Images1 = []
    image_vars=['image_shape_x','image_shape_y','pixel_mean','pixel_std','pixel_max','pixel_min','pixel_0.75','Pixel_0.9']

    list_=[[] for i in range(len(image_vars))]
    d=DataCreation()
    loop=0
    df=df0[loop:loop+1000]
    for k in list(temp):

        loop+=1
        Images = di.read_file(k, force=True)
        data = Images.pixel_array

        for i in range(0,len(image_vars)):
            try:
                if i ==0:temp=data.shape[0]
                elif i == 1:temp = data.shape[1]
                elif i==2:temp=np.mean(data)
                elif i == 3:temp = np.std(data)
                elif i == 4:temp = np.max(data)
                elif i == 5: temp = np.min(data)
                elif i == 6:temp = np.percentile(data,75)
                elif i == 7: temp = np.percentile(data,90)
                list_[i].append(temp)
            except :
                list_[i].append("error")
        if loop%1000==0 or loop==df0.shape[0]:
            print(loop,time.time()-start)
            for i in range(len(image_vars)):
                df[image_vars[i]] = list_[i]
            if loop==1000:df.to_csv(dataCreated / 'image_info' / 'images4.csv')
            else :df.to_csv(dataCreated / 'image_info' / 'images4.csv', mode='a', header=False)
            list_ = [[] for i in range(len(image_vars))]
            df = df0[loop:min(df0.shape[0]+1,loop + 1000)]
if False:
    # joining images2 and image 4 to find area
    df0 = pd.read_csv(dataCreated / 'image_info' / 'images2.csv', dtype='str')
    df1 = pd.read_csv(dataCreated / 'image_info' / 'images4.csv', dtype='str')
    temp=df0['coords'].apply(lambda x:packing.unpack(x))
    coords=torch.tensor(list(temp))
    df0['raw_area']=(coords[:,2]-coords[:,0])*(coords[:,3]-coords[:,1])
    df1=df1.set_index(['patient_id','test_type','image_name'])
    df1['max_area']=df1['image_shape_x'].astype('int')*df1['image_shape_y'].astype('int')
    df2=df1.join(df0[['patient_id','test_type','image_name','raw_area','plane']].set_index(['patient_id','test_type','image_name']))
    df2['occupied_perc']=df2['raw_area']/df2['max_area']
    df2.reset_index().to_csv(dataCreated / 'image_info' / 'images5.csv',index=False)

if True:
    # fitering images with coverage_ratio<threshold and creating probabilty for based on coverage+ratio
    threshold=0.2
    key_grouper=['patient_id','test_type','plane']


    df0 = pd.read_csv(Path(dataCreated) / 'image_info' / 'images5.csv')
    df1=df0[df0['occupied_perc']>threshold]
    df2=pd.DataFrame(df1.groupby(key_grouper)['occupied_perc'].sum()).rename(columns={'occupied_perc':'occupied_perc_sum'})
    df3=df1.set_index(key_grouper).join(df2)
    df3['occupied_perc_prob']=df3['occupied_perc']/df3['occupied_perc_sum']
    df3.reset_index(inplace=True)
    #considering only t1w axial images
    df3=df3[df3['test_type']=='T1w']
    df3 = df3[df3['plane'] == 'axial']
    df3.to_csv(Path(dataCreated)/ 'image_info' / 'images6.csv',index=False)
if True:
    # preparing data set for dataloader
    def req(plain, test_type):
        if plain == 'axial':
            if test_type in ('T1wCE', 'T1w'):
                return 3
            else:
                return 2
        elif plain == 'coronal' and test_type in ('FLAIR'):
            return 1
        elif plain == 'sagittal' and test_type in ('T2w'):
            return 1
        else:
            return 0


    df0 = pd.read_csv(Path(dataCreated) / 'image_info' / 'images6.csv')
    df2 = df0.groupby(['patient_id', 'plane', 'test_type'])['image_name','occupied_perc_prob'].agg(list).reset_index().rename(columns={'image_name':'images'})
    df2['image_count'] = df2['images'].apply(lambda x: len(x))
    df2['req'] = df2.apply(lambda row: req(row['plane'], row['test_type']), axis=1)
    df2 = df2[df2['req'] != 0]
    patient_ids = df2['patient_id'].unique()
    patient_ids = [[p] * 6 for p in patient_ids]
    plains = [['axial'] * 4 + ['coronal'] + ['sagittal'] for p in range(len(patient_ids))]
    test_types = [['T1wCE', 'T1w', 'FLAIR', 'T2w', 'FLAIR', 'T2w'] for p in range(len(patient_ids))]
    requireds = [[3] * 2 + [2] * 2 + [1, 1] for p in range(len(patient_ids))]
    template = pd.DataFrame(data={'patient_id': list(itertools.chain(*patient_ids)),
                                  'plane': list(itertools.chain(*plains)),
                                  'test_type': list(itertools.chain(*test_types)),
                                  'required': list(itertools.chain(*requireds))
                                  })

    template.set_index(['patient_id', 'plane', 'test_type'], inplace=True)
    df3 = template.join(df2.set_index(['patient_id', 'plane', 'test_type']))
    df3.reset_index(inplace=True)

    df3['images'] = df3['images'].fillna("").apply(list)
    df3['images'] = df3.apply(lambda row: row['images'] + ['blank'] * (max(0,row['required']-len(row['images']))),axis=1)
    #df3['select_images'] = df3.apply(lambda row: row['images'][0:row['required']], axis=1)
    df3['loc'] = '/' + df3['patient_id'].astype('str') + "/" + df3['test_type'] + '/'

    df3['occupied_perc_prob']=df3['occupied_perc_prob'].fillna('blank')
    for col in df3.columns:
        df3[col] = df3[col].apply(lambda x: packing.pack(x))

    df3.to_csv(Path(dataCreated) / 'image_info' / 'images7.csv',index=False)





