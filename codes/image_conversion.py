import pandas as pd
import numpy as np
import pydicom as di
import torch,time,cv2
from commonFuncs import packing
from funcs import DataCreation
from config import root,dataCreated
import os
from multiprocessing import Process, Manager,Pool,cpu_count
"""
As 16 bit images to converted to 8 bit for faster implementation, there was need to find a conversion with aim to minimize the loss.
Instead of scaling each pixel by 256(256 bin becomes 1 after conversion), as the majority of pixel falls in 0-2**11 space , it seems better
to convert first 2048 pixel as scale of 10 (taking 204/256), next 2**11-2**13  with scale of 200 (30/256), next 2**13-2**15 pixel as scale of 1500
(16/256) , 2**15-2**16 as scale of 6000 (5/256)

Once done , we will resize all image to 256*256 and save in a location.

"""
def conversion(x):
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
    root=pth+"/preprocessed/"
    if not os.path.exists(root):
        os.mkdir(root)
        for f in folders:
            os.mkdir(root+str(f)+"/")
            for sf in sub_folder:
                os.mkdir(root + str(f) + "/"+sf)





if True:
    start = time.time()
    cores = cpu_count()
    pool = Pool(processes=4)
    pth = str(root) + '/data/rsna-miccai-brain-tumor-radiogenomic-classification/train/'
    output_path=str(dataCreated)+"/preprocessed/"
    df0=pd.read_csv(dataCreated / 'image_info' / 'images0.csv',dtype='str')#,nrows=20000
    #df1=df1[0:8765]
    temp=df0.apply(lambda row: str(row['patient_id'])+"/"+row['test_type']+"/"+row['image_name'],axis=1)
    create_directories(pth=str(dataCreated),folders=list(df0['patient_id'].unique()),sub_folder=['FLAIR','T1wCE','T1w','T2w'])
    loop=0
    for k in list(temp):
        pool.apply_async(conv_save, args=(pth + k + ".dcm",output_path + k + '.jpg'))
        loop+=1
        if loop%50==0 :

            pool.close()
            pool.join()
            if loop % 10000 == 0: print(loop, time.time() - start)
            pool = Pool(processes=4)
    pool.close()
    pool.join()
    print(loop, time.time() - start)

if __name__ == "__main__1":
    pth = str(root) + '/data/rsna-miccai-brain-tumor-radiogenomic-classification/train/'
    output_path=str(dataCreated)+"/preprocessed/"
    Images = di.read_file(pth + k + ".dcm", force=True)
    data=conversion(Images.pixel_array)
    im_orig = cv2.resize(data.astype('float32'), (256, 256), interpolation=cv2.INTER_CUBIC).astype(np.uint8)
    im=cv2.imread(output_path+k+'.jpg',cv2.IMREAD_UNCHANGED)

    c=0

        
