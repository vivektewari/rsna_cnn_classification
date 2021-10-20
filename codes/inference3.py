from catalyst.dl import SupervisedRunner, CallbackOrder, Callback, CheckpointCallback
import dicom2nifti

from sklearn.metrics import roc_curve, auc


from funcs import DataCreation,create_directories,lorenzCurve

import matplotlib.pyplot as plt


from config import *
from funcs import get_dict_from_class,count_parameters
from models import modelling_3d

from torch.utils.data import DataLoader
import pandas as pd

import torch




# inference pipe line
# 1.iterate over patients
# 2. for folder not with flir assign 0.5 prob and for foldrs with flair do following:
#     a. convert dcm to dicom2nifti
#     b. apply model
#     c . save predcition in dictionary format
#     d. post all iteration convert dictionary datafram-> to csv for submission
# e(optional) calculate auc and losses(for trianign data only
def inference(model_param,model_,data_loader_param,data_loader,pretrained=None):
    data_load = data_loader(**get_dict_from_class(data_loader_param))
    model = model_(**get_dict_from_class(model_param))
    count_parameters(model)
    if pretrained is not None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(pretrained, map_location=device)
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
        except:
            model.load_state_dict(checkpoint)
        model.eval()

    val_file =data_load.data

    loaders = {

        "valid": DataLoader(data_loader(data_frame=val_file, **get_dict_from_class(data_loader_param)),
                            batch_size=4,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True,
                            drop_last=False)
    }

    runner = SupervisedRunner(
        model = model,
        output_key="logits",
        input_key="image_pixels",
        target_key="targets")
    # scheduler=scheduler,
    predictions=[]
    for prediction in runner.predict_loader(loader=loaders["valid"]):
        predictions.extend(prediction['logits'].detach().cpu().numpy())
    return predictions

def dcm_to_nii(input_path,output_path):
    """
    Converts dicom to nift and rescale to 240*240*155
    :param input_path:
    :param output_path:
    :return:
    """

    dicom2nifti.dicom_series_to_nifti(Path(input_path), os.path.join(output_path+"task.nii"))

    img=tio.ScalarImage(output_path+"task.nii")

    resample =tio.Resample((1,1,1))
    crp = tio.CropOrPad((240, 240, 155))

    res= resample(img)
    temp = np.flip(res.data[0].numpy(), axis=1)
    res.data[0]=torch.tensor(temp.copy())
    res=crp(res)
    source_coords = DataCreation.get_1_coords(res.data[0])
    res.data[0] = DataCreation.allign_first_coords([7, 7, 7], source_coords, res.data[0])
    res.save(output_path+"/"+"task.nii")
    # img = tio.ScalarImage(output_path + "task.nii")
    # print(torch.equal(img.data[0],res.data[0]))
    return res
    # res.save(output_path+"/resampled.nii")

def get_directory_file(directory_path,file_ending,output):
    test_type, patient_id, image_name, names = [], [], [], []
    for ro, dirs, files in os.walk(Path(directory_path)):
        for file in files:
            if file.endswith(file_ending):#"tumor.nii"
                temp = ro.split("/")
                test_type.append(temp[-1])
                patient_id.append(temp[-2])
                names.append(file)
                #loc.append(str(ro)+"/"+file)

    df = pd.DataFrame(data={'patient_id': patient_id, 'test_type': test_type,'file':names})#, 'loc': loc
    df=df[df.test_type=='FLAIR']
    df.to_csv(output)
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
    start=time.time()
    pretrained = '/home/pooja/Downloads/rsna_20.pth'

    pth ='/home/pooja/PycharmProjects/rsna_cnn_classification/data/dataCreated/difference/'
    pth1 = '/home/pooja/PycharmProjects/rsna_cnn_classification/data/dataCreated/common/'
    pth1 = '/home/pooja/PycharmProjects/rsna_cnn_classification/data/dataCreated/difference/'
    #
    pth = '/home/pooja/PycharmProjects/rsna_cnn_classification/data/dataCreated/mixed/'
    #pth = '/home/pooja/PycharmProjects/rsna_cnn_classification/data/dataCreated/nii/'
    temp=get_patients(pth1,".dcm")
    from_dcm    =   temp[0]

    noflair =  temp[1]


    temp_path= dataCreated + '/rough/'
    dir_temp=temp_path+"nii/"
    create_directories(directory_name=dir_temp,folders=from_dcm,sub_folder=['FLAIR'])

    # for p in from_dcm :
    #     dcm_to_nii(pth1 + "/" + p + "/FLAIR/", dir_temp+p+"/FLAIR/")
    print(time.time()-start)
    get_directory_file(dir_temp,'task.nii',temp_path+"patient.csv")
    #pd.read_csv(temp_path+"patient.csv",nrows=50).to_csv(temp_path+"patient.csv")
    df = pd.read_csv(temp_path + "patient.csv")
    df['BraTS21ID']=df['patient_id']
    df['BraTS21ID'] = df['BraTS21ID'].apply(lambda x: str(x).zfill(5))
    df['MGMT_value']=0
    df[['BraTS21ID','MGMT_value']].to_csv(temp_path+"target.csv")
    data_loader_param.data_frame_path=temp_path+"patient.csv"
    data_loader_param.label=temp_path+"target.csv"
    data_loader_param.base_loc = dir_temp
    predictions=inference(model_param, model, data_loader_param, data_loader, pretrained=pre_trained_model)


    df['pred']=predictions
    #df.to_csv(temp_path+"target.csv")

    train_file = pd.read_csv('/home/pooja/PycharmProjects/rsna_cnn_classification/data/train_labels.csv')
    train_file['BraTS21ID'] = train_file['BraTS21ID'].apply(lambda x: str(x).zfill(5))
    target_dict=train_file.set_index('BraTS21ID').to_dict()['MGMT_value']
    df['actual']=df['BraTS21ID'].apply(lambda x :target_dict[x])
    pd.DataFrame({'BraTS21ID':list(df['BraTS21ID'])+noflair,'MGMT_value':list(df['pred'])+[0.5 for i in range(len(noflair))]}).to_csv(dataCreated+'/inference/sub.csv')
    df.to_csv(dataCreated+'/inference/result7.csv',index=False)
    print(time.time()-start) #without multiprocessing time:82.7 for difference folder.mult processtime :
    lorenzCurve(df['actual'],df['pred'],save_loc=dataCreated+'/inference/com.png')



