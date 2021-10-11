
class Model1:
    start_channel = 1
    channels = [32,64,128,64]
    input_image_dim = (21,256, 256)
    convs = [3, 3,2, 2, 2, 2, 2,2,2,2]  # 5,5,3
    pads = [0, 0, 0, 0, 0, 0, 0, 0,0,0,0]
    strides = [1, 1, 1, 1, 1, 1, 1,1,1,1]
    pools = [2, 2, 1, 1, 1, 1, 1,1,1,1]  # [2, 2,2,1]
    fc1_p = [32, 1]  # [256, 14]
class Model2:
    start_channel = 12
    channels = [24,48, 64, 32,16]
    input_image_dim = (256, 256)

    convs = [2, 2,2, 3, 2, 2, 2,2,1,1,1,1]  # 5,5,3
    pads = [0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0]
    strides = [1, 1, 1, 1, 1, 1, 1,1,1,1,1,]
    pools = [2, 2, 2, 2, 2, 2, 1,1,1,1,1]  # [2, 2,2,1]
    fc1_p = [None, 1]  # [256, 14]
class Model1_nii:
    start_channel = 1
    channels = [32,64,128,64]
    input_image_dim = (240,240, 155)
    convs = [4, 4,3, 3, 2, 2, 2,2,2,2]  # 5,5,3
    pads = [0, 0, 0, 0, 0, 0, 0, 0,0,0,0]
    strides = [1, 1, 1, 1, 1, 1, 1,1,1,1]
    pools = [3, 3, 2, 2, 1, 1, 1,1,1,1]  # [2, 2,2,1]
    fc1_p = [32, 1]  # [256, 14]

class DataLoad1:
    # data_frame = None
    label = 'label'
    reshape_pixel = (28, 28)
    pixel_col = ['pixel' + str(i) for i in range(reshape_pixel[0] * reshape_pixel[1])]
    path = ""
class rsna_param:
    data_frame_path = 'ni_images0_tumor.csv'
    label = 'train_labels.csv'
    base_loc = None
    blank_loc = None







