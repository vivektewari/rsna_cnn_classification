import pandas as pd
import numpy as np
import pydicom as di
import torch, time
from commonFuncs import packing
from funcs import DataCreation,create_directories
from config import root, dataCreated
import os,glob
from shutil import copyfile
import itertools
import os, cv2
from os import listdir
from pathlib import Path

pth = str(root) + '/data/rsna-miccai-brain-tumor-radiogenomic-classification/train/'
#pth = '/home/pooja/PycharmProjects/rsna_cnn_classification/data/dataCreated/preprocessed3/'
if 0:
    # making a dataframe for each image to keep infor for test_type and patirent_id

    test_type, patient_id, image_name, loc = [], [], [], []
    for ro, dirs, files in os.walk(pth):
        for file in files:
            if file.endswith(".png"):
                temp = ro.split("/")
                test_type.append(temp[-1])
                patient_id.append(str(temp[-2]))
                image_name.append(file.split(".")[0])

    df = pd.DataFrame(data={'patient_id': patient_id, 'test_type': test_type, 'image_name': image_name})
    df.to_csv(Path(dataCreated) / 'image_info' / 'images0.csv')

if 0:
    start = time.time()
    # adding .dcm non pixel information
    dataCreated = Path(dataCreated)
    df0 = pd.read_csv(dataCreated / 'image_info' / 'images0.csv', dtype='str')
    # df1=df1[0:8765]
    temp = df0.apply(
        lambda row: pth + str(row['patient_id']) + "/" + row['test_type'] + "/" + row['image_name'] + ".dcm", axis=1)
    Images1 = []
    image_vars = ['coords', 'ImageOrientationPatient', 'ImagePositionPatient', 'SliceLocation',
                  'PhotometricInterpretation', 'SliceThickness', 'SpacingBetweenSlices', 'PixelSpacing',
                  'SamplesPerPixel', 'RescaleIntercept', 'RescaleSlope', 'RescaleType']
    list_ = [[] for i in range(len(image_vars))]
    d = DataCreation()
    loop = 0
    df = df0[loop:loop + 1000]
    for k in list(temp):

        loop += 1
        Images = di.read_file(k, force=True)
        data = Images.pixel_array
        data = np.where(data > 0, 1, 0)
        list_[0].append(packing.pack(d.coords(data)))

        for i in range(1, len(image_vars)):
            try:
                list_[i].append(packing.pack(Images[image_vars[i]].value))
            except:
                list_[i].append("error")

        if loop % 1000 == 0 or loop == df0.shape[0]:
            print(loop, time.time() - start)
            for i in range(len(image_vars)):
                df[image_vars[i]] = list_[i]
            if loop == 1000:
                df.to_csv(dataCreated / 'image_info' / 'images1.csv')
            else:
                df.to_csv(dataCreated / 'image_info' / 'images1.csv', mode='a', header=False)
            list_ = [[] for i in range(len(image_vars))]
            df = df0[loop:min(df0.shape[0] + 1, loop + 1000)]

if 0:
    # deriving variable on image based on variables

    df0 = pd.read_csv(Path(dataCreated) / 'image_info' / 'images1.csv', dtype='str')
    temp = df0['coords'].apply(lambda x: packing.unpack(x))
    coords = torch.tensor(list(temp))
    df0['area'] = (coords[:, 2] - coords[:, 0]) * (coords[:, 3] - coords[:, 1])
    temp = df0['ImageOrientationPatient'].apply(lambda x: packing.unpack(x))
    orient = np.array(list(temp))
    # ['1', '0', '0', '0', '0', '-1']-Coronal plane view
    #
    # ['0', '1', '0', '0', '0', '-1']=Sagittal plane view
    #
    # ['1', '0', '0', '0', '1', '0']-Axial plane view
    # allowing till +-5 degree roation for any plan cos5-cos 0=0.00381  and cos 95-cos 90=0.0872
    # allowing till +-5 degree roation for any plan cos5-cos 0=0.0152  and cos 95-cos 90=0.174
    orient = np.where(np.abs(orient - 0) > 0.174, orient, 0)
    orient = np.where(np.abs(orient - 1) > 0.0152, orient, 1)
    orient = np.where(np.abs(orient - (-1)) > 0.0152, orient, -1)
    coronal = np.all(np.abs(orient - np.array([1, 0, 0, 0, 0, -1])) < 0.00000001, axis=1)
    sagittal = np.all(np.abs(orient - np.array([0, 1, 0, 0, 0, -1])) < 0.00000001, axis=1)
    axial = np.all(np.abs(orient - np.array([1, 0, 0, 0, 1, 0])) < 0.00000001, axis=1)
    df0['plane'] = 'undef'
    loop = 0
    pl = ['coronal', 'sagittal', 'axial']
    for p in [coronal, sagittal, axial]:
        df0['plane'][p] = pl[loop]
        loop += 1
    drops = [col for col in df0.columns if col.find('Unnamed') >= 0]
    df = df0.drop(drops, axis=1)
    df.to_csv(Path(dataCreated) / 'image_info' / 'images2.csv')

if False:
    root = '/home/pooja/PycharmProjects/rsna_cnn_classification/'
    dataCreated = root + '/data/dataCreated/'
    blank_loc = dataCreated + '/auxilary/'
    output = blank_loc + '/blank.png'
    im = np.zeros((256, 256))
    cv2.imwrite(output, im)

if 0:
    # adding array shape, max ,min 10,90 percentile of pixel value
    start = time.time()
    df0 = pd.read_csv(Path(dataCreated) / 'image_info' / 'images0.csv', dtype='str',nrows=2000)
    # df1=df1[0:8765]
    temp = df0.apply(
        lambda row: pth + str(row['patient_id']) + "/" + row['test_type'] + "/" + row['image_name'] + ".png", axis=1)#.dcm
    Images1 = []
    image_vars = ['image_shape_x', 'image_shape_y', 'pixel_mean', 'pixel_std', 'pixel_max', 'pixel_min', 'pixel_0.75',
                  'Pixel_0.9']

    list_ = [[] for i in range(len(image_vars))]
    d = DataCreation()
    loop = 0
    df = df0[loop:loop + 1000]
    for k in list(temp):

        loop += 1
        # Images = di.read_file(k, force=True)
        # data = Images.pixel_array
        data=cv2.imread(k, cv2.IMREAD_UNCHANGED)

        for i in range(0, len(image_vars)):
            try:
                if i == 0:
                    temp = data.shape[0]
                elif i == 1:
                    temp = data.shape[1]
                elif i == 2:
                    temp = np.mean(data[data>0])
                elif i == 3:
                    temp = np.std(data[data>0])
                elif i == 4:
                    temp = np.max(data[data>0])
                elif i == 5:
                    temp = np.min(data[data>0])
                elif i == 6:
                    temp = np.percentile(data[data>0], 75)
                elif i == 7:
                    temp = np.percentile(data[data>0], 90)
                list_[i].append(temp)
            except:
                list_[i].append("error")
        if loop % 1000 == 0 or loop == df0.shape[0]:
            print(loop, time.time() - start)
            for i in range(len(image_vars)):
                df[image_vars[i]] = list_[i]
            if loop == 1000:
                df.to_csv(Path(dataCreated) / 'image_info' / 'images4.csv')
            else:
                df.to_csv(Path(dataCreated) / 'image_info' / 'images4.csv', mode='a', header=False)
            list_ = [[] for i in range(len(image_vars))]
            df = df0[loop:min(df0.shape[0] + 1, loop + 1000)]
if False:
    # joining images2 and image 4 to find area
    df0 = pd.read_csv(dataCreated / 'image_info' / 'images2.csv', dtype='str')
    df1 = pd.read_csv(dataCreated / 'image_info' / 'images4.csv', dtype='str')
    temp = df0['coords'].apply(lambda x: packing.unpack(x))
    coords = torch.tensor(list(temp))
    df0['raw_area'] = (coords[:, 2] - coords[:, 0]) * (coords[:, 3] - coords[:, 1])
    df1 = df1.set_index(['patient_id', 'test_type', 'image_name'])
    df1['max_area'] = df1['image_shape_x'].astype('int') * df1['image_shape_y'].astype('int')
    df2 = df1.join(df0[['patient_id', 'test_type', 'image_name', 'raw_area', 'plane']].set_index(
        ['patient_id', 'test_type', 'image_name']))
    df2['occupied_perc'] = df2['raw_area'] / df2['max_area']
    df2.reset_index().to_csv(dataCreated / 'image_info' / 'images5.csv', index=False)

if 0:
    # fitering images with coverage_ratio<threshold and creating probabilty for based on coverage+ratio
    threshold = 0
    key_grouper = ['patient_id', 'test_type', 'plane']

    df0 = pd.read_csv(Path(dataCreated) / 'image_info' / 'images5.csv')
    df1 = df0[df0['occupied_perc'] > threshold]
    df2 = pd.DataFrame(df1.groupby(key_grouper)['occupied_perc'].sum()).rename(
        columns={'occupied_perc': 'occupied_perc_sum'})
    df3 = df1.set_index(key_grouper).join(df2)
    df3['occupied_perc_prob'] = df3['occupied_perc'] / df3['occupied_perc_sum']
    df3.reset_index(inplace=True)
    # considering only t1w axial images
    # df3=df3[df3['test_type']=='T1w']
    # df3 = df3[df3['plane'] == 'axial']
    df3.to_csv(Path(dataCreated) / 'image_info' / 'images6.csv', index=False)
if False:
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
    df2 = df0.groupby(['patient_id', 'plane', 'test_type'])['image_name', 'occupied_perc_prob'].agg(
        list).reset_index().rename(columns={'image_name': 'images'})
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
    df3['images'] = df3.apply(lambda row: row['images'] + ['blank'] * (max(0, row['required'] - len(row['images']))),
                              axis=1)
    # df3['select_images'] = df3.apply(lambda row: row['images'][0:row['required']], axis=1)
    df3['loc'] = '/' + df3['patient_id'].astype('str') + "/" + df3['test_type'] + '/'

    df3['occupied_perc_prob'] = df3['occupied_perc_prob'].fillna('blank')
    for col in df3.columns:
        df3[col] = df3[col].apply(lambda x: packing.pack(x))

    df3.to_csv(Path(dataCreated) / 'image_info' / 'images7.csv', index=False)

if 0:  # adding slice location information and selecting those images only whose slice location is in close proximity to 5 divisibilty
    # it's 7b replacement for 7 for new addition
    included_tests = ['T1w', 'T1wCE']
    included_plain = ['axial']
    patient_repeat = 21 * (len(included_plain) * len(included_tests))
    df0 = pd.read_csv(dataCreated + '/image_info/images2.csv')
    df1 = pd.read_csv(dataCreated + '/image_info/images6.csv')
    df0 = df0[df0['SliceLocation'] != 'error']

    df0['SliceLocation_integer'] = df0['SliceLocation'].astype('float32').apply(lambda x: round(x))
    df0['SliceLocation_integer'] = df0['SliceLocation_integer'].apply(
        lambda x: x if x % 5 == 0 else x - 1 if (x - 1) % 5 == 0 else x + 1 if (x + 1) % 5 == 0 else -1)
    df0 = df0[df0['SliceLocation_integer'] != -1]
    # df0 = df0.drop_duplicates(['patient_id', 'test_type', 'SliceLocation_integer'])

    df2 = pd.merge(df1, df0[['patient_id', 'test_type', 'image_name', 'SliceLocation_integer']],
                   on=['patient_id', 'test_type', 'image_name'], how='left')
    df2 = df2[df2['test_type'].isin(included_tests)]
    df2 = df2[df2['plane'].isin(included_plain)]

    df2 = df2.groupby(['patient_id', 'plane', 'test_type', 'SliceLocation_integer'])[
        'image_name', 'occupied_perc_prob'].agg(list).reset_index().rename(columns={'image_name': 'images'})
    df2['image_count'] = df2['images'].apply(lambda x: len(x))
    df2['req'] = 1  # df2.apply(lambda row: req(row['plane'], row['test_type']), axis=1)
    df2 = df2[df2['req'] != 0]
    patient_ids = df2['patient_id'].unique()
    patient_ids = [[p] * patient_repeat for p in patient_ids]
    plains = [included_plain * 2 * 21 for p in range(len(patient_ids))]
    test_types = [included_tests * 21 for p in range(len(patient_ids))]
    requireds = [[1] * patient_repeat for p in range(len(patient_ids))]
    SliceLocation_integer = [[-50 + 5 * i for i in range(21)] * 2 for p in range(len(patient_ids))]
    template = pd.DataFrame(data={'patient_id': list(itertools.chain(*patient_ids)),
                                  'plane': list(itertools.chain(*plains)),
                                  'test_type': list(itertools.chain(*test_types)),
                                  'required': list(itertools.chain(*requireds)),
                                  'SliceLocation_integer': list(itertools.chain(*SliceLocation_integer))
                                  })

    template.set_index(['patient_id', 'plane', 'test_type', 'SliceLocation_integer'], inplace=True)
    df3 = template.join(df2.set_index(['patient_id', 'plane', 'test_type', 'SliceLocation_integer']))
    df3.reset_index(inplace=True)

    df3['images'] = df3['images'].fillna("").apply(list)
    df3['images'] = df3.apply(lambda row: row['images'] + ['blank'] * (max(0, row['required'] - len(row['images']))),
                              axis=1)
    # df3['select_images'] = df3.apply(lambda row: row['images'][0:row['required']], axis=1)
    df3['loc'] = '/' + df3['patient_id'].astype('str') + "/" + df3['test_type'] + '/'

    df3['occupied_perc_prob'] = df3['occupied_perc_prob'].fillna('blank')
    for col in df3.columns:
        df3[col] = df3[col].apply(lambda x: packing.pack(x))

    df3.to_csv(Path(dataCreated) / 'image_info' / 'images7b.csv', index=False)

if 0:

    # adjsuting slice location,making data for 3d scan
    # testing for slice location offset. Compute difference of first and last visible slides
    def get_start_offset(input_string, slice_location):

        """
        1.Validate string : see if 01 or 10 doesnt break input strin in 3 parts
        2.if validation passes , get first location of 1 , if not availble then get last location of input and subtract norma range
        :param input_string: str|e.g. 0001100,110000,0111,111
        :param slice_location: list |location of each slice

        :return: int |location of first  visible slice
        """

        data_driven_range = 140
        zero_one = len(input_string.split("01"))
        one_zero = len(input_string.split("10"))
        if zero_one == 1 and one_zero == 1:
            return -1
        elif zero_one <= 2 and one_zero <= 2:
            if zero_one == 2:
                output = torch.tensor(slice_location) - slice_location[one_zero + 1]
            elif one_zero == 2:
                output = torch.tensor(slice_location) - slice_location[len(input_string) - zero_one - 2] - data_driven_range
            return output.tolist()
        else:
            return -1

    def assign_location(slice_location):
        # if slice_thickness >= 5: rounding_value = 5
        # elif slice_thickness > 2: rounding_value=3
        # else: rounding_value=2
        rounding_value=5

        quoteint=int(slice_location/rounding_value)
        remainder=slice_location%rounding_value


        if remainder<=rounding_value/2.0: return  quoteint*rounding_value
        else : return (quoteint+1)*rounding_value





    included_plain = ['axial']
    df0 = pd.read_csv(dataCreated + '/image_info/images2.csv')
    df1 = pd.read_csv(dataCreated + '/image_info/images5.csv')

    df0 = df0[df0['SliceLocation'] != 'error']

    # .apply(lambda x: round(x))
    # df0 = df0.drop_duplicates(['patient_id', 'test_type', 'SliceLocation_integer'])

    df2 = pd.merge(df1.drop('plane', axis=1),
                   df0[['patient_id', 'test_type', 'image_name', 'SliceLocation', 'plane', 'SliceThickness']],
                   on=['patient_id', 'test_type', 'image_name'], how='inner')

    df2 = df2[df2['plane'].isin(included_plain)]

    df2['pixel_max'] = df2['pixel_max'].apply(lambda x: int(x > 0))
    df2['SliceLocation'] = df2['SliceLocation'].astype('float')
    df2 = df2.sort_values(['patient_id', 'test_type', 'SliceLocation'])

    df2 = df2.groupby(['patient_id', 'plane', 'test_type'])[
        'image_name', 'pixel_max', 'SliceLocation'].agg(list).reset_index().rename(columns={'image_name': 'images'})
    df2['pixel_max_str'] = df2['pixel_max'].apply(lambda x: "".join(map(str, x)))
    df2['SliceLocation'] = df2.apply(lambda row: get_start_offset(row['pixel_max_str'], row['SliceLocation']), axis=1)
    df2=df2[df2['SliceLocation']!=-1]
    df2=df2.drop(['pixel_max','pixel_max_str'],axis=1)
    df2=df2.set_index(['patient_id', 'test_type','plane']).apply(pd.Series.explode).reset_index()
    df2['SliceLocation'] = df2['SliceLocation'].apply(lambda x: assign_location(x))
    #getting template ready
    included_tests = ['T1w']#, 'T1wCE'
    included_plain = ['axial']
    patient_repeat = 21 * (len(included_plain) * len(included_tests))
    patient_ids = df2['patient_id'].unique()
    patient_ids = [[p] * patient_repeat for p in patient_ids]
    plains = [included_plain * len(included_tests) * 21 for p in range(len(patient_ids))]
    test_types = [included_tests * 21*len(included_plain)  for p in range(len(patient_ids))]
    requireds = [[1] * patient_repeat for p in range(len(patient_ids))]
    SliceLocation_integer = [[20 + 5 * i for i in range(21)] * len(included_tests)*len(included_plain ) for p in range(len(patient_ids))]
    template = pd.DataFrame(data={'patient_id': list(itertools.chain(*patient_ids)),
                                  'plane': list(itertools.chain(*plains)),
                                  'test_type': list(itertools.chain(*test_types)),
                                  # 'required': list(itertools.chain(*requireds)),
                                  'SliceLocation': list(itertools.chain(*SliceLocation_integer))
                                  })
    template.set_index(['patient_id', 'plane', 'test_type', 'SliceLocation'], inplace=True)
    df2 = df2.groupby(['patient_id', 'plane', 'test_type', 'SliceLocation'])['images'].agg(list)#.reset_index()

    df3 = template.join(df2)

    df3['images'] = df3['images'].fillna("").apply(list)
    df3['images'] = df3.apply(lambda row: row['images'] + ['blank'] * (max(0, 1 - len(row['images']))),
                              axis=1)
    df3.reset_index(inplace=True)
    # df3['select_images'] = df3.apply(lambda row: row['images'][0:row['required']], axis=1)
    df3['loc'] = '/' + df3['patient_id'].astype('str') + "/" + df3['test_type'] + '/'


    for col in df3.columns:
        df3[col] = df3[col].apply(lambda x: packing.pack(x))


    df3.to_csv(Path(dataCreated) / 'image_info' / 'images7c.csv', index=False)


if 0:#for nii file
    #testing the images in data loader
    df0=pd.read_csv(Path(dataCreated) / 'image_info' / 'images7c.csv')
    df1=df0.drop_duplicates(['patient_id','test_type'])
    df1=df1.drop(['plane','SliceLocation','images'],axis=1)
    df1.to_csv(Path(dataCreated) / 'image_info' / 'images7d.csv', index=False)

if 0:#getting segmenttion task file
    pth = '/home/pooja/PycharmProjects/rsna_cnn_classification/data/task_2/BraTS2021_Training_Data/'
    test_type, patient_id, image_name, loc = [], [], [], []
    for ro, dirs, files in os.walk(pth):
        for file in files:
            if file.endswith("tumor_eq.nii.gz"):
                temp = ro.split("/")
                test_type.append(file.split(".")[0].split("_")[2])
                patient_id.append(str(temp[-1].removeprefix("BraTS2021_")))
                #loc.append(str(ro)+"/"+file)

    df = pd.DataFrame(data={'patient_id': patient_id, 'test_type': test_type, 'loc': loc})
    df.to_csv(Path(dataCreated) / 'image_info' / 'ni_images0_tumor_eq.csv')
if 0:#getting segmenttion task file
    pth ='/home/pooja/PycharmProjects/rsna_cnn_classification/data/dataCreated/kaggle_data_aug/'
    #pth = '/home/pooja/PycharmProjects/rsna_cnn_classification/data/dataCreated/mixed/'
    test_type, patient_id, image_name, names = [], [], [], []
    for ro, dirs, files in os.walk(pth):
        for file in files:
            if file.endswith("resampled.nii"):#"tumor.nii"
                temp = ro.split("/")
                test_type.append(temp[-1])
                patient_id.append(temp[-2])
                names.append(file)
                #loc.append(str(ro)+"/"+file)

    df = pd.DataFrame(data={'patient_id': patient_id, 'test_type': test_type,'file':names})#, 'loc': loc
    df=df[df.test_type=='FLAIR']
    df.to_csv(Path(dataCreated) / 'image_info' / 'ni_images0_task_resampled.csv')#'ni_images0_tumor_aug.csv'
if 1:
    #creating train file(target ) for augmented list
    train_file=pd.read_csv('/home/pooja/PycharmProjects/rsna_cnn_classification/data/train_labels.csv')
    train_file['BraTS21ID'] = train_file['BraTS21ID'].apply(lambda x: str(x).zfill(5))
    train_file.set_index('BraTS21ID',inplace=True)#.to_dict()['MGMT_value']

    df0=pd.read_csv(Path(dataCreated) / 'image_info' / 'ni_images0_task_resampled.csv')
    df0['patient_id'] = df0['patient_id'].apply(lambda x: str(x).zfill(5))
    df0['temp']=df0['patient_id'].apply(lambda x:'0'+x[1:])
    df0=df0.join(train_file,on='temp')
    df0=df0.rename(columns={'patient_id':'BraTS21ID'})
    df0=df0[['BraTS21ID','MGMT_value']]
    df0.to_csv('/home/pooja/PycharmProjects/rsna_cnn_classification/data/train_labels_aug_resmapled.csv',index=False)
if 0:
    #creating train file(target ) for augmented list





    train_file=pd.read_csv('/home/pooja/PycharmProjects/rsna_cnn_classification/data/train_labels.csv')
    train_file['BraTS21ID'] = train_file['BraTS21ID'].apply(lambda x: str(x).zfill(5))
    dict_=train_file.set_index('BraTS21ID').to_dict()['MGMT_value']
    dict_1=dict_.copy()
    for key in dict_.keys():
        mgmt_val=dict_[key]
        p=list(key)
        for i in range(1,6):
            p[0]=str(i)
            if int(i) % 2==0:
                dict_1["".join(p)] = mgmt_val
            else:
                dict_1["".join(p)] = 1-mgmt_val


    df0=pd.DataFrame({'BraTS21ID':list(dict_1.keys()),'MGMT_value':list(dict_1.values())})


    df0.to_csv('/home/pooja/PycharmProjects/rsna_cnn_classification/data/train_labels_aug2.csv',index=False)


if 0:# getting common and uncommon patients from kaggle and segmentation data
    def copy_files(pth, cp_pth, match_list):
        for cp in match_list:
            paths = \
                (glob.glob(pth + cp, recursive=True))
            for p in paths:
                p2 = str(p).removeprefix(pth)
                copyfile(p, cp_pth + p2)
    train_file = pd.read_csv('/home/pooja/PycharmProjects/rsna_cnn_classification/data/train_labels.csv')
    train_file['BraTS21ID'] = train_file['BraTS21ID'].apply(lambda x: str(x).zfill(5))
    df0=pd.read_csv('/home/pooja/PycharmProjects/rsna_cnn_classification/data/dataCreated/image_info/ni_images0_task_aug.csv')
    df0['patient_id'] = df0['patient_id'].apply(lambda x: str(x).zfill(5))
    common=list(set(train_file['BraTS21ID']).intersection(set(df0['patient_id'])))
    difference=list(set(train_file['BraTS21ID']).difference(set(df0['patient_id'])))
    #create_directories(directory_name='common_files',folders=common,sub_folder=['FLAIR'])
    #create_directories(directory_name='difference_files', folders=difference, sub_folder=['FLAIR'])
    print(common[0:10])
    print(difference)

