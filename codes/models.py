import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
from funcs import vison_utils
#from multibox_loss import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size=(6, 6),
                 stride=(1, 1), padding=(5, 5), pool_size=(2, 2)):
        super().__init__()
        self.pool_size = pool_size
        self.in_channels = in_channels


        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=tuple(np.array(kernel_size) + np.array([0, 0])),
            stride=stride,
            padding=tuple(np.array(padding) + np.array([0, 0])),
            bias=False)

        # self.bn1 = nn.BatchNorm2d(out_channels)
        # self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input1, pool_size=None, pool_type='max'):
        if pool_size is None: pool_size = self.pool_size
        x = input1

        x = self.conv(input1)#F.relu_(self.conv(input1))
        # x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
            return x




class FeatureExtractor(nn.Module):
    def __init__(self, start_channel=4, input_image_dim=(28, 28), channels=[2],
                 convs=[4], strides=[1], pools=[2], pads=[1], fc1_p=[10, 10]):
        super().__init__()
        self.num_blocks = len(channels)
        self.start_channel = start_channel
        self.conv_blocks = nn.ModuleList()
        self.input_image_dim = input_image_dim
        self.fc1_p = fc1_p
        self.mode_train = 1
        self.activation_l = torch.nn.LeakyReLU()
        self.activation = torch.nn.Sigmoid()
        self.dropout = nn.Dropout(0.2)
        last_channel = start_channel
        for i in range(self.num_blocks):
            self.conv_blocks.append(ConvBlock(in_channels=last_channel, out_channels=channels[i],
                                              kernel_size=(convs[i], convs[i]), stride=(strides[i], strides[i]),
                                              pool_size=(pools[i], pools[i]), padding=pads[i]))
            last_channel = channels[i]

        # getting dim of output of conv blo
        conv_dim = self.get_conv_output_dim()
        if self.fc1_p[0] is not None:
            self.fc1 = nn.Linear(conv_dim[0], fc1_p[0], bias=True)
            self.fc2 = nn.Linear(fc1_p[0], fc1_p[1], bias=True)
        else :
            self.conv_blocks.append(ConvBlock(in_channels=last_channel, out_channels=fc1_p[1],
                                              kernel_size=(1, 1), stride=(1, 1),
                                              pool_size=(conv_dim[1][-2],conv_dim[1][-1]), padding=0))
            self.num_blocks+=1


        self.init_weight()


    def get_conv_output_dim(self):
        input_ = torch.Tensor(np.zeros((1,self.start_channel)+self.input_image_dim))
        x = self.cnn_feature_extractor(input_)
        return len(x.flatten()),x.shape

    @staticmethod
    def init_layer(layer):
        nn.init.xavier_uniform_(layer.weight*10)
        if hasattr(layer, "bias"):
            if layer.bias is not None:
                layer.bias.data.fill_(0.)

    def init_weight(self):
        for i in range(self.num_blocks):
            self.init_layer(self.conv_blocks[i].conv)

        if self.fc1_p[0] is not None:
            self.init_layer(self.fc1)
            self.init_layer(self.fc2)
        # init_layer(self.conv2)
        # init_bn(self.bn1)
        # init_bn(self.bn2)

    def cnn_feature_extractor(self, x):
        # input 501*64
        for i in range(self.num_blocks):
            x = self.conv_blocks[i](x)
            # x_70=torch.quantile(x, 0.7)
            # x_50 = torch.quantile(x, 1)
            # x=self.activation_l((x-x_50-1)/max(x_50,0.01))
            if i<(len(self.conv_blocks)-1): x = self.activation_l(x)
            if self.mode_train == 1:
                x = self.dropout(x)
            #x=torch.clamp(x,100,-100)

        return x

    def forward(self, input_):
        input_=input_*1.0
        mean=torch.mean(torch.where(input_>0.0,input_,torch.tensor(np.nan)),dim=(2,3)).reshape((input_.shape[0:2]+(1,1))).expand(input_.shape)
        mean=torch.nan_to_num(mean,nan=0)
        std = torch.std(torch.where(input_>0.0,input_,mean), dim=(2, 3)).reshape((input_.shape[0:2]+(1,1))).expand(input_.shape)
        std=torch.where(std==0.0,torch.tensor(0.0001),std)
        std = torch.nan_to_num(std, nan=0)

        x=(input_-mean)/std
        #x = input_/torch.nan_to_num(mean,nan=1)
        x = self.cnn_feature_extractor(x)
        if self.fc1_p is None:
            x = x.flatten(start_dim=1, end_dim=-1)
            return self.activation(x)

        x = x.flatten(start_dim=1, end_dim=-1)
        if self.fc1_p[0] is not None:
            x = self.activation_l(x)
            x = self.activation_l(self.fc1(x))
            if self.mode_train == 1:
                x = self.dropout(x)
            x = x / torch.max(x)
            x = self.fc2(x)

        x = self.activation(x)
        # x=F.softmax(x,dim=1)
        # x = F.softmax(x, dim=1)
        # x =x[:,0]

        return x.flatten()

class ConvBlock_3d(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, kernel_size=(6, 6,6),
                     stride=(1, 1), padding=(5, 5), pool_size=(2, 2)):
            super().__init__()
            self.pool_size = pool_size
            self.in_channels = in_channels

            self.conv = nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=tuple(np.array(kernel_size) + np.array([0, 0,0])),
                stride=stride,
                padding=tuple(np.array(padding) + np.array([0, 0,0])),
                bias=False)

            # self.bn1 = nn.BatchNorm2d(out_channels)
            # self.bn2 = nn.BatchNorm2d(out_channels)

        def forward(self, input1, pool_size=None, pool_type='max'):
            if pool_size is None: pool_size = self.pool_size
            x = input1

            x = self.conv(input1)  # F.relu_(self.conv(input1))
            # x = F.relu_(self.bn2(self.conv2(x)))
            if pool_type == 'max':
                x = F.max_pool3d(x, kernel_size=pool_size)
                return x

class modelling_3d(FeatureExtractor):

    def __init__(self, start_channel=4, input_image_dim=(28, 28), channels=[2],
                 convs=[4], strides=[1], pools=[2], pads=[1], fc1_p=[10, 10]):
        super(FeatureExtractor, self).__init__()

        self.num_blocks = len(channels)
        self.start_channel = start_channel
        self.conv_blocks = nn.ModuleList()
        self.input_image_dim = input_image_dim
        self.fc1_p = fc1_p
        self.mode_train = 1
        self.activation_l = torch.nn.LeakyReLU()
        self.activation = torch.nn.Sigmoid()
        self.dropout = nn.Dropout(0.2)
        last_channel = start_channel
        for i in range(self.num_blocks):
            self.conv_blocks.append(ConvBlock_3d(in_channels=last_channel, out_channels=channels[i],
                                              kernel_size=(convs[i], convs[i],convs[i]), stride=(strides[i], strides[i], strides[i]),
                                              pool_size=(pools[i], pools[i],pools[i]), padding=pads[i]))
            last_channel = channels[i]

        # getting dim of output of conv blo
        conv_dim = self.get_conv_output_dim()
        if self.fc1_p[0] is not None:
            self.fc1 = nn.Linear(conv_dim[0], fc1_p[0], bias=True)
            self.fc2 = nn.Linear(fc1_p[0], fc1_p[1], bias=True)
        else :
            self.conv_blocks.append(ConvBlock_3d(in_channels=last_channel, out_channels=fc1_p[1],
                                              kernel_size=(1, 1,1), stride=(1, 1,1),
                                              pool_size=(conv_dim[1][-3],conv_dim[1][-2],conv_dim[1][-1]), padding=0))
            self.num_blocks+=1


        self.init_weight()
    def preprocess(self,input_):
        input_=input_.reshape((input_.shape[0],1,)+tuple(input_.shape[1:]))*1.0
        # mean=torch.mean(torch.where(input_>0.0,input_,torch.tensor(np.nan)),dim=(2,3,4),keepdim=True)#.reshape((input_.shape[0:3]+(1,1))).expand(input_.shape)
        # mean=torch.nan_to_num(mean,nan=0)
        # std = torch.std(torch.where(input_>0.0,input_,mean),dim=(2,3,4),keepdim=True)
        # std=torch.where(std==0.0,torch.tensor(0.0001),std)
        # std = torch.nan_to_num(std, nan=0)
        #
        # x=(input_-mean)/std
        x=input_/5000.0
        return x
    def forward(self, input_):

        x=self.preprocess(input_)
        x = self.cnn_feature_extractor(x)
        if self.fc1_p is None:
            x = x.flatten(start_dim=1, end_dim=-1)
            return self.activation(x)

        x = x.flatten(start_dim=1, end_dim=-1)
        if self.fc1_p[0] is not None:
            x = self.activation_l(x)
            x = self.activation_l(self.fc1(x))
            if self.mode_train == 1:
                x = self.dropout(x)
            x = x / torch.max(x)
            x = self.fc2(x)

        x = self.activation(x)
        # x=F.softmax(x,dim=1)
        # x = F.softmax(x, dim=1)
        # x =x[:,0]

        return x.flatten()