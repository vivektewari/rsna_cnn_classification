class Model1:
    channels = [31,65]
    input_image_dim = (28, 28)
    start_channel = 1
    convs = [5 ,5]#5,5,3
    pads = [0, 0,1]
    strides = [1,1, 1,1]
    pools = [2, 2,1]
    fc1_p = [256, 10]#1040

class Model2:
    channels = [31,65]
    input_image_dim = (28, 28)
    start_channel = 1
    convs = [5 ,5]#5,5,3
    pads = [0, 0,1]
    strides = [1,1, 1,1]
    pools = [2, 2,1]
    fc1_p = [256*5, 10]#1040

class Model3:
    channels = [31,65,128]
    input_image_dim = (28, 28)
    start_channel = 1
    convs = [5 ,5,3]#5,5,3
    pads = [0, 0,1]
    strides = [1,1, 1,1]
    pools = [2, 2,1]
    fc1_p = [256*5, 10]#1040

class Model4:
    channels = [31,65]
    input_image_dim = (28, 28)
    start_channel = 1
    convs = [3 ,3]#5,5,3
    pads = [0, 0,0]
    strides = [1,1, 1,1]
    pools = [2, 2,1]
    fc1_p = [None, 10]#1040
class Model5:
    channels = [31,65,128]
    input_image_dim = (28, 28)
    start_channel = 1
    convs = [5 ,5,3]#5,5,3
    pads = [0, 0,0]
    strides = [1,1, 1,1]
    pools = [2, 2,1]
    fc1_p = [None, 10]#1040

