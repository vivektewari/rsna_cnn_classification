import numpy as np
import nibabel as nib
import pandas as pd
import os
from config import root
import torchio as tio
import torch

from config import dataCreated
from pathlib import Path
if 0:
    img = nib.load("/home/pooja/PycharmProjects/rsna_cnn_classification/rough/4_flair.nii.gz")

    a = np.array(img.dataobj)
c=0
#
# pth = str(root) + '/data/rsna-miccai-brain-tumor-radiogenomic-classification/train/'
#pth = '/home/pooja/PycharmProjects/rsna_cnn_classification/data/dataCreated/preprocessed3/'
if 0:
    # making a dataframe for each image to keep infor for test_type and patirent_id
    pth="/home/pooja/PycharmProjects/rsna_cnn_classification/data/task_2/BraTS2021_Training_Data/"
    loop=0
    test_type, patient_id, image_name, loc = [], [], [], []
    patients,x,y,z=[],[],[],[]
    for ro, dirs, files in os.walk(pth):
        for file in files:
            if file.endswith("_seg.nii.gz"):
                temp = nib.load(ro+"/"+file)
                a = np.array(temp.dataobj)
                area=a.shape[0]*a.shape[1]*a.shape[2]/100.0
                test_type.append(a[a==1].shape[0]/area*1.0)
                patient_id.append(a[a==2].shape[0]/area*1.0)
                image_name.append(a[a==3].shape[0]/area*1.0)
                loc.append(a[a == 4].shape[0] / area * 1.0)
                patients.append(file.replace("_seg.nii.gz","").replace("BraTS2021_",""))
                x.append(a.shape[0])
                y.append(a.shape[1])
                z.append(a.shape[2])
                loop+=1
            if loop>100:break
        if loop > 100: break

    df = pd.DataFrame(data={'patient': patients,'x':x,'y': y,'z':z,'1':patient_id, '2': test_type, '3': image_name,'4':loc})
    df.to_csv(Path(dataCreated) / 'image_info' / 'tumor_.csv')

if 0:
    import dicom2nifti

    from nilearn.image import resample_img
    import skimage.transform as skTrans

    image_path='/home/pooja/PycharmProjects/rsna_cnn_classification/rough/00000.nii'


    img2='/home/pooja/PycharmProjects/rsna_cnn_classification/data/task_2/BraTS2021_Training_Data/BraTS2021_00000/BraTS2021_00000_flair.nii.gz'

    dicom_img='/home/pooja/PycharmProjects/rsna_cnn_classification/data/rsna-miccai-brain-tumor-radiogenomic-classification/train/00000/FLAIR/'
    #convert dcom2 nifti
    #dicom2nifti.dicom_series_to_nifti(dicom_img,image_path)
    #dicom2nifti.dicom_series_to_nifti(Path(dicom_img),Path(image_path))
    #resampling for size
    img=tio.ScalarImage(image_path)
    reference = tio.ScalarImage(img2)

    resample =tio.Resample((1,1,1))
    res= resample(img)
    crp=tio.CropOrPad((240,240,155))
    res=crp(res)

    res.affine[1,3]=239#res.affine=np.matmul(np.array([[1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]]),res.affine)
    res.save('/home/pooja/PycharmProjects/rsna_cnn_classification/rough/0_flair2.nii.gz')

    # result1 = skTrans.resize(img, (240, 240, 155), order=1, preserve_range=True)
    # nib.loadsave.save(result1,'/home/pooja/PycharmProjects/rsna_cnn_classification/rough/4_flair2.nii.gz')
if 1:
    #checking masking
    mask=root+'/data/task_2/BraTS2021_Training_Data/BraTS2021_00000/BraTS2021_00000_seg.nii.gz'
    orig=root+'/data/task_2/BraTS2021_Training_Data/BraTS2021_00000/BraTS2021_00000_flair.nii.gz'
    converted=root+'/rough/0_flair2.nii.gz'

    #pplying mask to original image
    img = tio.ScalarImage(orig)
    mas=tio.ScalarImage(mask)
    mas.data=np.array(mas.data)
    mas.data[mas.data>0]=3
    mas.data[mas.data==0]=1
    img.data=np.multiply(img.data,mas.data)
    img.save(root+'/rough/orig_masked.nii.gz')
    img2=tio.ScalarImage(converted)
    mas.data=torch.flip(mas.data,([2]))
    img2.data = np.multiply(img2.data, mas.data)
    img2.save(root + '/rough/conv_masked.nii.gz')

