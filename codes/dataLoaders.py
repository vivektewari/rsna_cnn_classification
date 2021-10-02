from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch,itertools,cv2,time,random
import pydicom as di
from commonFuncs import packing
import numpy as np

maxrows =50000
class rsna_loader(Dataset):
    def __init__(self, data_frame=None, label=None, base_loc=None,blank_loc=None,data_frame_path=None):
        """
        1. get csv from label and image_info file
        2. convert image info file in dictionary

        :param data_frame: dataframewith image list
        :param label: dataframe with labels

        :param base_loc:
        """

        temp=pd.read_csv(label)
        temp['BraTS21ID']=temp['BraTS21ID'].apply(lambda x: str(x).zfill(5))
        temp['MGMT_value']=temp['MGMT_value'].astype(np.float)
        self.labels=temp.set_index('BraTS21ID').to_dict()['MGMT_value']

        if data_frame is None:
            data_frame = pd.read_csv(data_frame_path)
            data_frame['patient_id']=data_frame['patient_id'].apply(lambda x: str(x).zfill(5))
        for col in data_frame.columns:
            data_frame[col] = data_frame[col].apply(lambda x: packing.unpack(x))
        self.data = data_frame
        self.data.reset_index(inplace=True, drop=True)
        self.image_col = 'images'
        self.patient_col ='patient_id'
        self.base_loc=base_loc

        self.blank_loc=blank_loc
        self.dict=self.nested_dict_from_df(self.data)

        self.range_to_patient=self.index_to_patient_id()
    def refresh(self):
        self.dict = self.nested_dict_from_df(self.data)

    def index_to_patient_id(self):
        dict={}
        temp=self.data['patient_id'].unique()
        for i in range(len(temp)):
            dict[i]=temp[i]
        return dict



    def patient_dict_(self,idx):
        channel = []

        test_types=['T2w','FLAIR','T1wCE', 'T1w']
        plane=['axial','coronal','sagittal']
        SliceLocation=[20 + 5*i for i in range(21) ]
        temp=self.dict[idx]

        for t in  test_types:
            pth=self.base_loc+"/"+str(idx)+"/"+t+"/"
            for p in plane:
                for sl in SliceLocation:
                    if self.req(p,t,sl)==0:continue

                    ims=temp[t][p][sl]["images"]

                    for im in ims:
                        if im == 'blank':Images=cv2.imread(self.blank_loc + im + '.png', cv2.IMREAD_UNCHANGED)
                        else:Images = cv2.imread(pth + im + '.png', cv2.IMREAD_UNCHANGED)

                        channel.append(torch.tensor(Images.astype(np.int32)))


        pixel=torch.stack(channel)
        label=self.labels[idx]

        return {'targets': label, 'image_pixels': pixel}




    def __getitem__(self, idx):
        patient_id=self.range_to_patient[idx]
        return self.patient_dict_(patient_id)

    def __len__(self):
        return len(self.dict.keys())

    def req(self,plain, test_type,SliceLocation):
        """
        provides number of channel allocated fr tthis
        :param plain: str
        :param test_type: str
        :return:
        """
        if plain == 'axial' and test_type in ('T1w') and SliceLocation in [20+ 5*i for i in range(21)]:
            return 1
        else :
            return 0
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


    def rec_dict(self, data_frame, key_list,last_key=None):
        """
        recusively constructs a dictionry
        :param data_frame:
        :param key_list:
        :param last_key:
        :return:
        """
        dict_={}
        var=key_list.pop(0)
        if len(key_list)==0:
            data_frame=data_frame.set_index(var)
            dict_=data_frame.to_dict(orient='index')
            for key in dict_.keys():
                num_choice=1#self.req(key,last_key)

                if len(dict_[key]['images'])>1:
                    dict_[key]['images']=np.random.choice(dict_[key]['images'], size=num_choice, replace=False)
            return dict_
        else:
            val = data_frame[var].unique()
            for v in val:
                temp = data_frame[data_frame[var] == v]
                dict_[v]= self.rec_dict(temp, key_list[:],v)
            return dict_







    def nested_dict_from_df(self,dataframe):
        # create master dict

        start=time.time()
        keys=['patient_id', 'test_type', 'plane','SliceLocation']
        dict=self.rec_dict(self.data,keys)
        print(f'Time: {time.time() - start}')
        return dict





if __name__ == "__main__":
    from funcs import get_dict_from_class

    root = '/home/pooja/PycharmProjects/rsna_cnn_classification/'
    dataCreated = root + '/data/dataCreated/'
    base_loc = dataCreated + '/preprocessed2/'
    blank_loc = dataCreated + '/auxilary/'

    class rsna_loader_test():
        def __init__(self):
            self.dl =  rsna_loader( data_frame_path=str(dataCreated)+'/image_info/images7c.csv', label=str(root)+ '/data/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv', base_loc=base_loc,blank_loc=blank_loc)


    test=rsna_loader_test()
    dict_=test.dl.dict
    t1=test.dl.patient_dict_('00003')
    t2=0