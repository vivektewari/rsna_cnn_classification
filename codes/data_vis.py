import pandas as pd
import seaborn as sns
from config import root,dataCreated
import matplotlib.pyplot as plt
from commonFuncs import packing
import torch,cv2
if False:
    # finding what will be best input 4 channel,9, 12 channel?


    import seaborn as sns
    df0 = pd.read_csv(dataCreated / 'image_info' / 'images2.csv', dtype='str')
    c=df0.describe()
    pid='patient_id'
    for pl in ['test_type','plane','SamplesPerPixel','PhotometricInterpretation']:
        data = df0.groupby(pl)["image_name"].count()
        data2 = df0[[pl,pid]].sort_values(by=[pl,pid]).reset_index().drop_duplicates(subset=[pl,pid]).groupby(by=[pl])[pid].count()
        data.plot.pie(autopct="%.1f%%")
        plt.savefig(str(root)+"/diagnostics/data_plots/"+pl+".png")
        plt.close()
        data2.plot.bar()
        plt.savefig(str(root) + "/diagnostics/data_plots/" + pl + "_account_count.png")
        plt.close()
    data = df0[['test_type','plane',pid]].drop_duplicates(subset=['test_type','plane',pid]).groupby(['test_type','plane']).size().unstack(fill_value=0)
    #plt.imshow(data, cmap='hot', interpolation='nearest')
    sns.heatmap(data, annot=True, cmap='Blues', fmt='g')
    plt.savefig(str(root) + "/diagnostics/data_plots/" + "type_v_plane_heat.png")
    plt.close()
    plt.hist(df0[['area']],bins=10, alpha=0.5,histtype='stepfilled' )
    plt.savefig(str(root) + "/diagnostics/data_plots/" + "area.png")
    plt.close()
if False:
    #analysing best crop for images
    df0 = pd.read_csv(dataCreated / 'image_info' / 'images1.csv', dtype='str')
    temp = df0['coords'].apply(lambda x: packing.unpack(x))
    coords = torch.tensor(list(temp))
    df0['x_len'] = coords[:, 2] - coords[:, 0]
    df0['y_len']=coords[:, 3] - coords[:, 1]
    plt.hist(df0[['x_len','y_len']], bins=10,density=True , histtype='bar',cumulative=1)


    plt.savefig(str(root) + "/diagnostics/data_plots/" + "len.png")

if False:
    df0 = pd.read_csv(dataCreated / 'image_info' / 'images4.csv')
    image_vars = ['image_shape_x', 'image_shape_y', 'pixel_mean', 'pixel_std', 'pixel_max', 'pixel_min', 'pixel_0.75',
                  'Pixel_0.9']

    for var in image_vars:
        temp=df0[df0[var]!=0][[var]]
        plt.hist(temp, bins=100,density=True , histtype='bar',cumulative=1)
        plt.savefig(str(root) + "/diagnostics/data_plots/" + var+"_hist.png")
        plt.close()
        plt.hist(temp, bins=100, density=False, histtype='bar', cumulative=0)
        plt.savefig(str(root) + "/diagnostics/data_plots/" + var + "_hist2.png")
        plt.close()
if False: # checking in slice location and occupied perc distribution
    var='slice_loc_perc_'
    df0 = pd.read_csv(dataCreated + '/image_info/images2.csv')
    df1 = pd.read_csv(dataCreated+ '/image_info/images6.csv')
    df1=df1[df1['image_shape_x']==512]
    df1 = df1[df1['image_shape_y'] == 512]

    df2=pd.merge(df1 ,df0[['patient_id','test_type','image_name','SliceLocation']],on =['patient_id','test_type','image_name'])
    df2=df2[df2['SliceLocation']!='error']
    df2['SliceLocation']= df2['SliceLocation'].astype('float64')
    for plane in ['coronal','axial','sagittal']:
        sns.scatterplot(data=df2[df2['plane']==plane][["SliceLocation","occupied_perc","plane"]].fillna(0), x="SliceLocation", y="occupied_perc", hue="plane")
        plt.savefig(str(root) + "/data/diagnostics/data_plots/" + var +plane+ "_scatter.png")
        plt.close()
if 1:#checking slice location avaiblty for each value
    var='slice_loc'
    df0 = pd.read_csv(dataCreated + '/image_info/images2.csv')
    df0 = df0[df0['SliceLocation'] != 'error']
    df0=df0[df0['test_type']=='T1w']
    df0 = df0[df0['plane'] == 'axial']
    df0['SliceLocation_integer'] = df0['SliceLocation'].astype('float32').apply(lambda x:round(x))
    df0['SliceLocation_integer'] = df0['SliceLocation_integer'].apply(lambda x:x if x%5==0 else x-1 if (x-1)%5==0 else x+1 if (x+1)%5==0 else -1)
    df0=df0[df0['SliceLocation_integer']!=-1]
    #df0=df0.drop_duplicates(['patient_id','test_type','SliceLocation_integer'])
    df1=df0.groupby(['SliceLocation_integer'])['patient_id'].count()
    plt.hist(df0[['SliceLocation_integer']], density=False,bins=400, histtype='step', cumulative=0)

    plt.savefig(str(root) + "/data/diagnostics/data_plots/" + var +"_hist.png")
    plt.close()
