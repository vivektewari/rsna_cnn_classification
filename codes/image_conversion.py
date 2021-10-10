import pandas as pd
import torchio as tio
import numpy as np
import pydicom as di
import torch,time,cv2
from tqdm import tqdm
from pathlib import Path
import SimpleITK as sitk
from commonFuncs import packing
from funcs import DataCreation
from config import root,dataCreated
import os,multiprocessing
from multiprocessing import Process, Manager,Pool,cpu_count
import nibabel as nib
import dicom2nifti
def dcm_to_nii(input_path,output_path):
    """
    Converts dicom to nift and rescale to 240*240*155
    :param input_path:
    :param output_path:
    :return:
    """

    dicom2nifti.dicom_series_to_nifti(Path(input_path), os.path.join(output_path+"/original.nii"))
    img=tio.ScalarImage(output_path+"/original.nii")
    resample =tio.Resample((1,1,1))
    res= resample(img)
    crp=tio.CropOrPad((240,240,155))
    res=crp(res)
    res.save(output_path+"/resampled.nii")


def create_landmark(image_path,filter,output_path=None):
        images_dir = Path(image_path)
        image_paths = sorted(images_dir.glob('*/'+filter+'/*.png'))
        histogram_landmarks_path = Path(output_path+filter+'.npy')
        _= tio.HistogramStandardization.train(
            image_paths,
            output_path=histogram_landmarks_path ,
        )
def apply_landmark(image_path,output_path,landmarks):
        sample=cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        subject = tio.Subject(
            mri=tio.ScalarImage(tensor=sample.reshape((1,256,256,1))),)
        standard = landmarks(subject )
        cv2.imwrite(output_path,standard.mri.data.reshape((256,256)).numpy().astype(np.uint16)*500)


def apply_landmark_nii(image_path, output_path, landmarks):
    #img = nib.load(image_path)
    img = tio.ScalarImage(image_path)

    #sample = np.array(img.get_data().copy())

    subject = tio.Subject(
        mri=tio.ScalarImage(tensor=img.data.reshape((1, 240, 240,155))), )
    #standard = np.array(landmarks(subject)['mri'].data)
    img.data=np.array(landmarks(subject)['mri'].data)
    img.save(output_path )
    #new_img = nib.Nifti1Image(standard, img.affine, img.header)
    #new_img.set_data_dtype(np.uint8)
    #nib.save(new_img, output_path)
#apply_landmark(image_path=dataCreated + '/preprocessed2/', filter='T1w', landmark_path=dataCreated + '/landmarks/T1w.npy')

def create_landmark_nii(image_path,filter,output_path=None):
    images_dir = Path(image_path)
    image_paths = sorted(images_dir.glob('*/' + filter + '/resampled.nii'))
    histogram_landmarks_path = Path(output_path + filter + '_nii_.npy')
    _ = tio.HistogramStandardization.train(
        image_paths,
        output_path=histogram_landmarks_path,
    )


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
def create_directories(directory_name,folders,sub_folder):
    root=directory_name
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

    cv2.imwrite(output_path,im.astype(np.uint16))

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
    im = clahe.apply(im) + 30
    cv2.imwrite(output_path,im.astype(np.uint16))
def conv_save4(input_path,output_path):
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


    cv2.imwrite(output_path,im.numpy().astype(np.uint16))


if  1:
    start = time.time()
    cores = cpu_count()
    pool = Pool(processes=cores)
    pth = str(root) + '/data/rsna-miccai-brain-tumor-radiogenomic-classification/train/'
    output_path=str(dataCreated)+"/nii/"
    df0=pd.read_csv(dataCreated +str('/image_info/images0.csv'),dtype='str')#,nrows=20000
    df0=df0[df0.test_type=='T1w']
    #df1=df1[0:8765]
    temp=df0.apply(lambda row: str(row['patient_id'])+"/"+row['test_type']+"/"+row['image_name'],axis=1)
    create_directories(directory_name=output_path,folders=list(df0['patient_id'].unique()),sub_folder=['FLAIR','T1wCE','T1w','T2w'])
    loop=0
    landmarks = np.load(dataCreated + '/landmarks/T1w_nii_.npy')
    landmarks_dict = {'mri': landmarks}
    histogram_transform = tio.HistogramStandardization(landmarks_dict)
    #from dicom to nifti converter
    df0['loc'] =  "/" + df0['patient_id'] + "/" + df0['test_type']
    temp=df0['loc'].unique()

    for k in list(temp):
        #pool.apply_async(conv_save4, args=(pth + k + ".dcm",output_path + k + '.png'))
        #pool.apply_async(dcm_to_nii, args=(pth+"/"+k , output_path+"/"+k ))
        #pool.apply_async(apply_landmark, args=(str(dataCreated)+"/preprocessed2/"+k+".png", str(dataCreated)+"/preprocessed4/"+k+".png", histogram_transform ))
        #apply_landmark(str(dataCreated)+"/preprocessed2/"+k+".png", str(dataCreated)+"/preprocessed4/"+k+".png", histogram_transform )
        #pool.apply_async(,args=
        pool.apply_async(apply_landmark_nii,args=(output_path+"/"+k +"/resampled.nii", output_path+"/"+k +"/resampled_eq.nii", histogram_transform ))
        loop+=1
        if loop%50==0 :

            pool.close()
            pool.join()
            if loop % 50== 0: print(loop, time.time() - start)
            pool = Pool(processes=cores)
    pool.close()
    pool.join()
    print(loop, time.time() - start)


if 0:
    #creting landmarks
    create_landmark_nii(image_path=str(dataCreated)+"/nii/", filter='T1w', output_path=dataCreated + '/landmarks/')

        
if __name__ == "__main__":
    landmarks = np.load(dataCreated + '/landmarks/T1w.npy')
    # landmarks_dict = {'mri': landmarks}
    # histogram_transform = tio.HistogramStandardization(landmarks_dict)
    # apply_landmark_nii(root+"/data/task_2/BraTS2021_Training_Data/BraTS2021_00000/BraTS2021_00000_flair.nii.gz","/home/pooja/PycharmProjects/rsna_cnn_classification/rough/4_flair1.nii.gz",histogram_transform )
    # pth = str(root) + '/data/rsna-miccai-brain-tumor-radiogenomic-classificationtrain/'
    # output_path=str(dataCreated)+"/preprocessed2/"
    # Images_orig = di.read_file(pth + '/00494/T1w/Image-93'+ ".dcm", force=True)
    # Images_orig = Images_orig.pixel_array
    # Images_saved= cv2.imread(output_path + '/00494/T1w/Image-93' + ".png", cv2.IMREAD_UNCHANGED )
    # #conv_save2(pth + '/00494/T1w/Image-93'+ ".dcm",output_path + '/00494/T1w/Image-93' + ".png")
    #
    # Images_saved= cv2.imread(output_path + '/00133/FLAIR/Image-35' + ".png", cv2.IMREAD_ANYDEPTH )
    #dcm_to_nii("/home/pooja/PycharmProjects/rsna_cnn_classification/data/rsna-miccai-brain-tumor-radiogenomic-classification/train/00000/FLAIR", "/home/pooja/PycharmProjects/rsna_cnn_classification/data/dataCreated/nii/00000/FLAIR/")