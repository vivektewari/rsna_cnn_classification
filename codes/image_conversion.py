import pandas as pd
import numpy as np
import pydicom as di
import torch,time,cv2
from commonFuncs import packing
from funcs import DataCreation
from config import root,dataCreated
import os
from multiprocessing import Process, Manager,Pool,cpu_count

def conversion(x):

        # As 16 bit images to converted to 8 bit for faster implementation, there was need to find a conversion with aim to minimize the loss.
        # Instead of scaling each pixel by 256(256 bin becomes 1 after conversion), as the majority of pixel falls in 0-2**11 space , it seems better
        # to convert first 2048 pixel as scale of 10 (taking 204/256), next 2**11-2**13  with scale of 200 (30/256), next 2**13-2**15 pixel as scale of 1500
        # (16/256) , 2**15-2**16 as scale of 6000 (5/256)
        #
        # Once done , we will resize all image to 256*256 and save in a location.
        #
        # :param x:
        # :return: converted int value

       x= np.where(np.logical_and(x >= 0, x < 2048), x/10, x) #204.8
       x = np.where(np.logical_and(x >= 2048, x < 8192), 204+((x-2048) / 200), x) #+30.72
       x = np.where(np.logical_and(x >= 8192, x < 32758), 234+((x-8192) / 1500), x) #+16.384
       x = np.where(x >= 32758, 250+((x-32768)/6000), x) #5.46

       return x.astype(int)

def conv_save(input_path,output_path):
    Images = di.read_file(input_path, force=True)
    data = conversion(Images.pixel_array)
    im = cv2.resize(data.astype('float32'), (256, 256), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(output_path, im)
def create_directories(pth,folders,sub_folder):
    root=pth+"/preprocessed2/"
    if not os.path.exists(root):
        os.mkdir(root)
        for f in folders:
            os.mkdir(root+str(f)+"/")
            for sf in sub_folder:
                os.mkdir(root + str(f) + "/"+sf)
def conv_save2(input_path,output_path):
    """
    avoid resizing where length and breadth are both less than 256 and using padding instead to
    make the image same size

    :param input_path: str|image loc
    :param output_path: where images needs to be saved
    :return: 
    """
    Images = di.read_file(input_path, force=True)
    data =Images.pixel_array
    if data.shape[0]<=256 and data.shape[1]<=256:
        im = cv2.copyMakeBorder(data, 0, 256-data.shape[0],0, 256-data.shape[1], cv2.BORDER_CONSTANT)
    else:
        im = cv2.resize(data.astype(np.uint16), (256, 256), interpolation=cv2.INTER_CUBIC)

    cv2.imwrite(output_path,im)

def conv_save3(input_path,output_path):
    """
    avoid resizing where length and breadth are both less than 256 and using padding instead to
    make the image same size

    :param input_path: str|image loc
    :param output_path: where images needs to be saved
    :return:
    """
    Images = di.read_file(input_path, force=True)
    data =Images.pixel_array
    if data.shape[0]<=256 and data.shape[1]<=256:
        im = cv2.copyMakeBorder(data, 0, 256-data.shape[0],0, 256-data.shape[1], cv2.BORDER_CONSTANT)
    else:
        im = cv2.resize(data.astype(np.uint16), (256, 256), interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=5)

    cv2.imwrite(output_path,im)



if  True:
    start = time.time()
    cores = cpu_count()
    pool = Pool(processes=cores)
    pth = str(root) + '/data/rsna-miccai-brain-tumor-radiogenomic-classification/train/'
    output_path=str(dataCreated)+"/preprocessed3/"
    df0=pd.read_csv(dataCreated +str('/image_info/images0.csv'),dtype='str')#,nrows=20000
    #df1=df1[0:8765]
    temp=df0.apply(lambda row: str(row['patient_id'])+"/"+row['test_type']+"/"+row['image_name'],axis=1)
    create_directories(pth=str(dataCreated),folders=list(df0['patient_id'].unique()),sub_folder=['FLAIR','T1wCE','T1w','T2w'])
    loop=0
    for k in list(temp):

        pool.apply_async(conv_save2, args=(pth + k + ".dcm",output_path + k + '.png'))
        loop+=1
        if loop%50==0 :

            pool.close()
            pool.join()
            if loop % 10000 == 0: print(loop, time.time() - start)
            pool = Pool(processes=cores)
    pool.close()
    pool.join()
    print(loop, time.time() - start)

if __name__ == "__main__":
    pth = str(root) + '/data/rsna-miccai-brain-tumor-radiogenomic-classification/train/'
    output_path=str(dataCreated)+"/preprocessed2/"
    Images_orig = di.read_file(pth + '/00417/T1w/Image-93'+ ".dcm", force=True)
    Images_orig = Images_orig.pixel_array
    Images_saved= cv2.imread(output_path + '/00417/T1w/Image-93' + ".png", cv2.IMREAD_ANYDEPTH )
    #data= conv_save2(pth + '/00417/T1w/Image-93'+ ".dcm",output_path + '/00417/T1w/Image-93' + ".png")

    Images_saved= cv2.imread(output_path + '/00133/FLAIR/Image-35' + ".png", cv2.IMREAD_ANYDEPTH )



    c=0

        
