import os
from pathlib import Path
import time
import torch,cv2
from funcs import create_directories,DataCreation
import numpy as np
import random,pandas as pd
import torchio as tio
from multiprocessing import Process, Manager,Pool,cpu_count

def brain_maker(input_loc, tumor_location, output_loc):
    """

    :param input_loc: str|input image location
    :param tumor_location str|tumor location which needs to be added
    :param output_loc: str|save location of image
    :return:
    """

    tumor_removed_img = tio.ScalarImage(input_loc)
    tumor_removed = tumor_removed_img.data
    tumor = tio.ScalarImage(tumor_location).data
    tumor_removed[tumor > 0] = 0
    tumor_removed_img.data=tumor_removed+tumor
    tumor_removed_img.save(output_loc)


def augmentor(input_loc,augmentor_list,output_loc):
        """
        perform augmentaton and saves the image
        :param input_loc: str|input image location
        :return: None
        """
        if (not os.path.exists(output_loc)):
            mas = tio.ScalarImage(input_loc)
            for aug in augmentor_list:
                mas=aug(mas)
            mas.save(output_loc)
if 0:#creating relevant directories to keep augmented files
    images_dir=Path('/home/pooja/PycharmProjects/rsna_cnn_classification/data/dataCreated/mixed/')
    images_dir = Path('/home/pooja/PycharmProjects/rsna_cnn_classification/data/dataCreated/kaggle_data_aug/')
    output_dir = Path('/home/pooja/PycharmProjects/rsna_cnn_classification/data/dataCreated/kaggle_data_aug/')
    #images_dir = Path('/home/pooja/PycharmProjects/rsna_cnn_classification/data/dataCreated/nii/')
    #output_dir=Path('/home/pooja/PycharmProjects/rsna_cnn_classification/data/dataCreated/mixed/')
    image_paths = sorted(images_dir.glob('*/FLAIR/resampled.nii'))
    num_aug=4
    num_aug = 2
    offset=1
    offset = 0
    patient_list=[]
    for im in image_paths:
        splitted = str(im).split("/")
        patient_id=splitted[-3]
        for i in range(offset+1,num_aug+offset+1):
            temp=list(patient_id)
            temp[0]=str(i)
            temp="".join(temp)
            patient_list.append(temp)
    patient_list=list(set(patient_list))
    #create_directories(str(output_dir)+"/",list(set(patient_list)),sub_folder=['FLAIR','T1wCE','T1w','T2w'],functionality='add1')
    flip = tio.RandomFlip(axes=('LR',))
    rb=tio.RandomBiasField()
    re=tio.RandomElasticDeformation()
    augs=[flip,rb,re]
    test_type=['FLAIR']
    cores = cpu_count()
    pool = Pool(processes=cores)
    loop=0
    start = time.time()
    #patient_list=[p for p in patient_list if list(p)[0] in ['2','3','4','5'] ]
    for p in patient_list:
        for t in test_type:
            temp=list(p)
            if temp[0] in ['1']:#['2','3']
                aug=random.choices(augs,k=1)
            if temp[0] in ['2']:#['4','5']
                aug=random.choices(augs,k=2)
            patient_id=list(p)
            patient_id[0]='0'#str(int(p[0])%2)
            patient_id="".join(patient_id)
            #augmentor(str(images_dir)+"/"+patient_id+"/"+t+"/"+"task.nii",aug,str(images_dir)+"/"+p+"/"+t+"/"+"task.nii")
            pool.apply_async(augmentor,args=(str(images_dir)+"/"+patient_id+"/"+t+"/"+"resampled.nii",aug,str(images_dir)+"/"+p+"/"+t+"/"+"resampled.nii"))
            loop += 1
            if loop % 50 == 0:

                pool.close()
                pool.join()
                if loop % 50 == 0: print(loop, time.time() - start)
                pool = Pool(processes=cores)
    pool.close()
    pool.join()
    print(loop, time.time() - start)
if 0:# inserting mgmt_1 to mgmt_0 brains and vice versa
    images_dir=Path('/home/pooja/PycharmProjects/rsna_cnn_classification/data/dataCreated/nii/')
    output_dir='/home/pooja/PycharmProjects/rsna_cnn_classification/data/dataCreated/mixed/'
    train_file = pd.read_csv('/home/pooja/PycharmProjects/rsna_cnn_classification/data/train_labels.csv')

    train_file['BraTS21ID'] = train_file['BraTS21ID'].apply(lambda x: str(x).zfill(5))

    patient_dict=train_file.set_index('BraTS21ID').to_dict()['MGMT_value']
    test_type='FLAIR'
    image_paths = sorted(images_dir.glob('*/' + test_type + '/' + 'tumor.nii'))
    patients = [str(l).split("/")[-3] for l in image_paths]
    train_file=train_file[train_file['BraTS21ID'].isin(patients)]
    mgmt_1=list(train_file[train_file['MGMT_value']==1]['BraTS21ID'])
    mgmt_0 = list(train_file[train_file['MGMT_value'] == 0]['BraTS21ID'])
    mgmt=[mgmt_1,mgmt_0]
    cores = cpu_count()
    pool = Pool(processes=cores)
    loop=0
    start = time.time()
    for l in image_paths:
        patient_id=str(l).split("/")[-3]
        new_id = list(patient_id)
        new_id[0] = '1'
        new_id = "".join(new_id)
        mgmt_value=patient_dict[patient_id]
        tumor_choice=random.choice(mgmt[patient_dict[patient_id]])
        # brain_maker(str(images_dir)+"/"+patient_id+"/"+test_type+"/"+"tumor_removed.nii",
        #             str(images_dir)+"/"+tumor_choice+"/"+test_type+"/"+"tumor.nii",
        #             str(output_dir)+"/"+new_id+"/"+test_type+"/"+"task.nii")
        pool.apply_async(brain_maker,
                         args=(str(images_dir)+"/"+patient_id+"/"+test_type+"/"+"tumor_removed.nii",
                    str(images_dir)+"/"+tumor_choice+"/"+test_type+"/"+"tumor.nii",
                    str(output_dir)+"/"+new_id+"/"+test_type+"/"+"task.nii"))
        loop += 1
        if loop % 50 == 0:

            pool.close()
            pool.join()
            if loop % 50 == 0: print(loop, time.time() - start)
            pool = Pool(processes=cores)
    pool.close()
    pool.join()
    print(loop, time.time() - start)


if 0:#creating augmentation of kaggle images
    images_dir=Path('/home/pooja/PycharmProjects/rsna_cnn_classification/data/dataCreated/nii/')
    #images_dir = Path('/home/pooja/PycharmProjects/rsna_cnn_classification/data/dataCreated/nii/')
    output_dir=Path('/home/pooja/PycharmProjects/rsna_cnn_classification/data/dataCreated/kaggle_data_aug/')
    image_paths = sorted(images_dir.glob('*/FLAIR/resampled.nii'))
    num_aug=2
    offset=0
    patient_list=[]
    for im in image_paths:
        splitted = str(im).split("/")
        patient_id=splitted[-3]
        for i in range(offset,num_aug+offset+1):
            temp=list(patient_id)
            temp[0]=str(i)
            temp="".join(temp)
            patient_list.append(temp)
    patient_list=list(set(patient_list))
    create_directories(str(output_dir)+"/",list(set(patient_list)),sub_folder=['FLAIR','T1wCE','T1w','T2w'],functionality='add1')
if 0: #shifting all images to have starting non zero pixel (7,7,7)
    def get_coords_transformed(input_path,output_path,start_coords):
        image=tio.ScalarImage(input_path)
        coords=DataCreation.get_1_coords(image.data[0])
        data = DataCreation.allign_first_coords(start_coords,coords,image.data[0])
        image.data[0]=data
        image.save(output_path)
    images_dir = Path('/home/pooja/PycharmProjects/rsna_cnn_classification/data/dataCreated/mixed/')
    # images_dir = Path('/home/pooja/PycharmProjects/rsna_cnn_classification/data/dataCreated/nii/')
    output_dir = Path('/home/pooja/PycharmProjects/rsna_cnn_classification/data/dataCreated/mixed/')
    image_paths = sorted(images_dir.glob('*/FLAIR/task.nii'))
    loop=0
    start = time.time()
    cores = cpu_count()
    pool = Pool(processes=cores)
    for im in image_paths:
        pool.apply_async(get_coords_transformed,args=(im,im,[7,7,7]))

        loop += 1
        if loop % 50 == 0:

            pool.close()
            pool.join()
            if loop % 50 == 0: print(loop, time.time() - start)
            pool = Pool(processes=cores)
    pool.close()
    pool.join()



