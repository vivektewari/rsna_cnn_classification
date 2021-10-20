import pandas as pd
import torchio as tio
import dicom2nifti,random
from models import modelling_3d
from funcs import get_dict_from_class,count_parameters
from sklearn.metrics import roc_curve, auc
from config import model_param
import numpy as np
import pydicom as di
import torch,time,cv2
from tqdm import tqdm
from pathlib import Path
import SimpleITK as sitk
from commonFuncs import packing
from funcs import DataCreation,create_directories,lorenzCurve
from config import root,dataCreated
import os,multiprocessing
import matplotlib.pyplot as plt
from multiprocessing import Process, Manager,Pool,cpu_count
import nibabel as nib

# inference pipe line
# 1.iterate over patients
# 2. for folder not with flir assign 0.5 prob and for foldrs with flair do following:
#     a. convert dcm to dicom2nifti
#     b. apply model
#     c . save predcition in dictionary format
#     d. post all iteration convert dictionary datafram-> to csv for submission
# e(optional) calculate auc and losses(for trianign data only

def model_executor(img):

    resample = tio.Resample((1, 1, 1))
    crp = tio.CropOrPad((240, 240, 155))
    print(torch.max(img.data))
    res = resample(img)
    temp = np.flip(res.data.numpy(), axis=2)
    res.data = torch.tensor(temp.copy())
    res = crp(res)
    pixel = res.data.long()
    pixel = pixel.reshape((pixel.shape[0], 1,) + tuple(pixel.shape[1:])) * 1.0
    print(torch.tensor(model(pixel).detach().numpy()[0]))
    #return torch.tensor(model(pixel).detach().numpy()[0])

def dcm_to_nii(input_path,output_path):
    """
    Converts dicom to nift and rescale to 240*240*155
    :param input_path:
    :param output_path:
    :return:
    """
    patient=input_path.split("/")[-3]
    dicom2nifti.dicom_series_to_nifti(Path(input_path), os.path.join(output_path+"/"+patient+"original.nii"))
    img=tio.ScalarImage(output_path+"/"+patient+"original.nii")
    resample =tio.Resample((1,1,1))
    crp = tio.CropOrPad((240, 240, 155))
    res= resample(img)
    temp = np.flip(res.data.numpy(), axis=2)
    res.data=torch.tensor(temp.copy())
    res=crp(res)
    return res
    # res.save(output_path+"/resampled.nii")
def get_score_rough(dcm_path,niifile=None):
    nii=dcm_to_nii(dcm_path,dataCreated+'/rough/')
    nii2=tio.ScalarImage(niifile)

    source_coords = DataCreation.get_1_coords(nii.data[0])
    target_coords=DataCreation.get_1_coords(nii2.data[0])
    try:
        final=DataCreation.allign_first_coords(target_coords,source_coords,nii.data[0])
    except :
        return -1
    nii.data[0] = final
    pixel = nii.data.long()
    pixel = pixel.reshape((pixel.shape[0], 1,) + tuple(pixel.shape[1:])) * 1.0
    return torch.tensor(model(pixel).detach().numpy()[0])
def get_score(dcm_path,niifile=None):
    if niifile is None:
        nii=dcm_to_nii(dcm_path,dataCreated+'/rough/')
        source_coords = DataCreation.get_1_coords(nii.data[0])
        nii.data[0]=DataCreation.allign_first_coords([7,7,7], source_coords, nii.data[0])
    else :nii=tio.ScalarImage(niifile)
    pixel = nii.data.long()
    pixel = pixel.reshape((pixel.shape[0], 1,) + tuple(pixel.shape[1:])) * 1.0
    return torch.tensor(model(pixel).detach().numpy()[0])
def get_patients(pth,file_name):
    pth=Path(pth)
    test_type, patient_id, image_name, names = [], [], [], []
    for ro, dirs, files in os.walk(pth):
        for file in files:
            # if file.endswith(".dcm"):#"tumor.nii"".dcm""task.nii"
            if file.endswith(file_name):  #
                temp = ro.split("/")
                test_type.append(temp[-1])
                patient_id.append(temp[-2])
                names.append(file)
                # loc.append(str(ro)+"/"+file)

    df = pd.DataFrame(data={'patient_id': patient_id, 'test_type': test_type, 'file': names})  # , 'loc': loc
    patients_all = df['patient_id'].unique()
    df = df[df.test_type == 'FLAIR']
    patients_flair = df['patient_id'].unique()
    patients_noflair =list(set(patients_all).difference(set(patients_flair)))
    return patients_flair,patients_noflair
if 1:
    pretrained = '/home/pooja/Downloads/rsna_20.pth'
    #pretrained = '/home/pooja/Downloads/rsna_38.pth'
    model =modelling_3d
    model = model(**get_dict_from_class(model_param))
    model.mode_train=0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if pretrained is not None:

        checkpoint = torch.load(pretrained, map_location=device)
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
        except:
            model.load_state_dict(checkpoint)
        model.eval()




    pth ='/home/pooja/PycharmProjects/rsna_cnn_classification/data/dataCreated/difference/'
    pth1 = '/home/pooja/PycharmProjects/rsna_cnn_classification/data/dataCreated/common/'
    pth1 = '/home/pooja/PycharmProjects/rsna_cnn_classification/data/dataCreated/difference/'
    pth = '/home/pooja/PycharmProjects/rsna_cnn_classification/data/dataCreated/mixed/'
    #pth = '/home/pooja/PycharmProjects/rsna_cnn_classification/data/dataCreated/nii/'
    from_dcm    =   get_patients(pth1,".dcm")[0]
    from_task1  =   get_patients(pth,"task.nii")[0]
    noflair =  get_patients(pth1,".dcm")[1]
    patients_flair=list(set(from_dcm).intersection(set(from_task1)))
    #patients_flair=get_patients(pth,".dcm")[0]
    #assigning 0.5 probability to patient who doesnt have flaor folder

    score_dict={}
    # for p in noflair:
    #     score_dict[p]=0.5
    patients_choosen=['0000'+str(i) for i in range(0,10)]+['000'+str(i) for i in range(10,99)]+['00'+str(i) for i in range(100,500)]#random.choices(patients_flair, k=40)
    patients_flair=list(set(patients_choosen).intersection(set(patients_flair)).difference(set(['00109','00123','00709'])))[:140]
    start = time.time()
    cores = cpu_count()
    pool = Pool(processes=cores)
    loop=0
    for p in from_dcm :#patients_flair:
        dicom2nifti.dicom_series_to_nifti(pth1 + "/" + p + "/FLAIR/", os.path.join(dataCreated+'/rough'+ "/" + p + "original.nii"))
        img = tio.ScalarImage(dataCreated+'/rough'+ "/" + p + "original.nii")
        print(torch.max(img.data))
        #score_dict[p] =model_executor(img)
        #score_dict[p] = get_score(pth1 + "/" + p + "/FLAIR/", niifile=None)#pth+"/"+p+"/FLAIR/task.nii"
        #score_dict[p]=torch.tensor(get_score(pth+"/"+p+"/FLAIR/",niifile=None).detach().numpy()[0])#pth+"/"+p+"/FLAIR/task.nii"
        #temp = get_score_rough(pth1 + "/" + p + "/FLAIR/", niifile=pth+"/"+p+"/FLAIR/task.nii")
        pool.apply_async(model_executor,args=(img,))
        loop += 1
        if loop % 50 == 0:
            pool.close()
            pool.join()
            if loop % 50 == 0: print(loop, time.time() - start)
            pool = Pool(processes=cores)
    pool.close()
    pool.join()
    print(loop, time.time() - start)
    # df=pd.DataFrame({'BraTS21ID':score_dict.keys(),'score':score_dict.values()})#.to_csv(dataCreated+'/inference/big_model.csv')
    # train_file = pd.read_csv('/home/pooja/PycharmProjects/rsna_cnn_classification/data/train_labels.csv')
    # train_file['BraTS21ID'] = train_file['BraTS21ID'].apply(lambda x: str(x).zfill(5))
    # target_dict=train_file.set_index('BraTS21ID').to_dict()['MGMT_value']
    # df['actual']=df['BraTS21ID'].apply(lambda x :target_dict[x])
    # df.to_csv(dataCreated+'/inference/result5.csv',index=False)
    # lorenzCurve(df['actual'],df['score'])



