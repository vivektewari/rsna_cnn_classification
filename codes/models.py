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
        self.activation_l = torch.nn.ReLU()
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

        input_ = input_ * 1.0
        mean = torch.mean(torch.where(input_ > 0.0, input_, torch.tensor(np.nan).to(device)), dim=(2, 3)).reshape(
            (input_.shape[0:2] + (1, 1))).expand(input_.shape)

        mean[torch.isnan(mean)] = torch.tensor(0.0).to(device)
        std = torch.std(torch.where(input_ > 0.0, input_, mean), dim=(2, 3)).reshape(
            (input_.shape[0:2] + (1, 1))).expand(input_.shape)
        std = torch.where(std == 0.0, torch.tensor(0.0001).to(device), std)

        x = (input_ - mean) / std
        # input_=input_[:,-21:,:,:]*1.0
        # mean=torch.mean(torch.where(input_>0.0,input_,torch.tensor(np.nan)),dim=(2,3)).reshape((input_.shape[0:2]+(1,1))).expand(input_.shape)
        # mean=torch.nan_to_num(mean,nan=0)
        # std = torch.std(torch.where(input_>0.0,input_,mean), dim=(2, 3)).reshape((input_.shape[0:2]+(1,1))).expand(input_.shape)
        # std=torch.where(std==0.0,torch.tensor(0.0001),std)
        # std = torch.nan_to_num(std, nan=0)

        # x=(input_-mean)/std
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




class FCLayered(FeatureExtractor):

    def __init__(self, num_blocks=1, start_channel=4, input_image_dim=(28, 28)):
        super().__init__()
        self.fc1 = nn.Linear(784, 30, bias=False)
        self.fc2 = nn.Linear(30, 2, bias=False)
        self.activation = torch.nn.Softmax()

    def forward(self, input_):
        x = input_

        x = self.fc1(x.flatten(start_dim=1, end_dim=-1))
        if self.mode == 'train': self.dropout = nn.Dropout(0.2)
        x = self.fc2(x)
        if self.mode == 'train': self.dropout = nn.Dropout(0.2)
        x = self.activation(x)
        return x


class FTWithLocalization(FeatureExtractor):

    def __init__(self, start_channel=4, input_image_dim=(28, 28), channels=[2],
                 convs=[4], strides=[1], pools=[2], pads=[1], fc1_p=[10, 10]):
        """

        :param : from super

        """
        super().__init__(start_channel, input_image_dim, channels,
                 convs, strides, pools, pads, fc1_p)
        self.activation =torch.nn.LeakyReLU()
        #self.activation_l =torch.nn.ReLU # torch.nn.LeakyReLU()
        self.dropout = nn.Dropout(0.2)
    def forward(self, input_):
        #x=super().forward(input_)
        x = self.cnn_feature_extractor(input_ / 255)
        x = F.normalize(x, dim=1)
        if self.fc1_p is None:
            x = x.flatten(start_dim=1, end_dim=-1)
            return self.activation(x)

        x = x.flatten(start_dim=1, end_dim=-1)
        if self.fc1_p[0] is not None:
            x = self.activation_l(x)
            x=self.fc1(x)
            x = self.activation_l(x)
            x = F.normalize(x, dim=1)

            if self.mode_train == 1:
                x = self.dropout(x)
            x = self.fc2(x)


        x = self.activation(x)
        #x = F.normalize(x, dim=1)



        first_slice = x[:, :10]
        first_slice = F.normalize(first_slice, dim=1)
        #first_slice = F.normalize(first_slice,dim=1)
        second_slice = x[:, 10:]
        tuple_of_activated_parts = (
            F.softmax(first_slice,dim=1),
            torch.clamp(second_slice,min=0, max=111))


        x = torch.cat(tuple_of_activated_parts, dim=1)

        return x

class FTWithLocalization_prior(FeatureExtractor):

    def __init__(self, start_channel=4, input_image_dim=(28, 28), channels=[2],
                 convs=[4], strides=[1], pools=[2], pads=[1], fc1_p=[10, 10]):
        """

        :param : from super

        """
        super().__init__(start_channel, input_image_dim, channels,
                         convs, strides, pools, pads, fc1_p)
        #self.activation =F.leaky_relu()# torch.nn.LeakyReLU()
        self.activation_l =torch.nn.ReLU() # torch.nn.Leaky ReLU()
        self.dropout = nn.Dropout(0.2)
        self.priors_cxcy = self.create_prior_boxes()
        self.priors_xy = torch.clamp(vison_utils.cxcy_to_xy(self.priors_cxcy), min=0, max=1)

    def model_outputs(self, model_output):
        """
        Decipher the 8732 locations and class scores (output of ths SSD300) to detect objects.
        For each class, perform Non-Maximum Suppression (NMS) on boxes that are above a minimum threshold.
        :param predicted_locs: predicted locations/boxes w.r.t the 8732 prior boxes, a tensor of dimensions (N, 8732, 4)
        :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 8732, n_classes)
        :param min_score: minimum threshold for a box to be considered a match for a certain class
        :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via NMS
        :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
        :return: detections (boxes, labels, and scores), lists of length batch_size
        """
        n_class=11
        batch_size = model_output.shape[0]
        boxes =model_output.shape[1] #int(model_output.shape[1] / (n_class + 4))
        scores = F.softmax(model_output[:,:,:n_class],dim=2)#.reshape((batch_size,boxes*n_class))#model_output[:, :boxes * n_class]
        locs = model_output[:,:,n_class:]#.reshape((batch_size,boxes*4))#[:, boxes * n_class:].reshape(batch_size,boxes,4)
        return_output=torch.zeros((batch_size,boxes*15))
        for i in range(batch_size):
            locs_i = vison_utils.gcxgcy_to_cxcy(locs[i],self.priors_cxcy)
            locs_i=torch.clamp(vison_utils.cxcy_to_xy(locs_i),0,1)
            output=torch.cat([scores[i].flatten(),locs_i.flatten()],dim=0)
            return_output[i]=output

        return return_output
    def forward(self, input_):
        #x=super().forward(input_)
        x = self.cnn_feature_extractor(input_/255)
        #print(torch.var(x))
        #x = F.sigmoid(x, dim=1)
        #x = torch.clamp(x, min=0, max=5)
        x = x.flatten(start_dim=1, end_dim=-1)
        #x=F.normalize(x,dim=1)
        # if self.mode_train == 1:
        #     x = self.dropout(x)
        #print(torch.var(x))
        if self.fc1_p[0] is None:
            pass #x= self.activation(x)
        if self.fc1_p[0] is not None:
            x = self.activation_l(x)
            x = self.fc1(x)
            x = self.activation_l(x)
            x = F.normalize(x, dim=1)


            x = self.fc2(x)


        #x = self.activation(x)
        #x = F.normalize(x,dim=1)
        x = x.reshape((x.shape[0], int(x.shape[1] / 15), 14 + 1)   )
        #x=x-x
        # first_slice = x[:,:, :11]
        # #first_slice=F.sigmoid(first_slice)
        # #first_slice = F.normalize(first_slice, dim=2)
        # second_slice = x[:, :, 11:]
        # #first_slice = F.normalize(first_slice, dim=2)
        # #second_slice = F.normalize(second_slice, dim=2)
        # #first_slice = F.normalize(first_slice,dim=1)
        #
        # tuple_of_activated_parts = (
        #     first_slice,
        #     torch.clamp(second_slice,min=0, max=1))
        #
        # tuple_of_activated_parts1=[x.flatten(start_dim=1, end_dim=-1) for x in tuple_of_activated_parts]
        # x = torch.cat(tuple_of_activated_parts1, dim=1)#.flatten(start_dim=1, end_dim=-1)
        # print("---")
        # print(torch.mean(torch.max(tuple_of_activated_parts[0], dim=2)[0]))
        # print(torch.mean(torch.max(x[:,:45*11].reshape((x.shape[0],45,11)), dim=2)[0]))
        return x

    @staticmethod
    def create_prior_boxes():
        """
        Create the 8732 prior (default) boxes for the SSD300, as defined in the paper.
        :return: prior boxes in center-size coordinates, a tensor of dimensions (8732, 4)
        """
        fmap_dims = {'conv4_3': 3,
                     'conv7': 9,
                     'conv8_2': 10,
                     'conv9_2': 5,
                     'conv10_2': 3,
                     'conv11_2': 1}

        obj_scales = {'conv4_3': 0.1,
                      'conv7': 0.2,
                      'conv8_2': 0.375,
                      'conv9_2': 0.55,
                      'conv10_2': 0.725,
                      'conv11_2': 0.9}

        aspect_ratios = {'conv4_3': [1., 2., 0.5],
                         'conv7': [1., 2., 3., 0.5, .333],
                         'conv8_2': [1., 2., 3., 0.5, .333],
                         'conv9_2': [1., 2., 3., 0.5, .333],
                         'conv10_2': [1., 2., 0.5],
                         'conv11_2': [1., 2., 0.5]}
        # fmap_dims = {'conv4_3': 10}
        # obj_scales = {'conv4_3': [0.324693658031057,0.13982694076936,0.254101116776089,0.121527277611988,0.379986683048495,0.162851193909395,0.171834183680792]}
        # aspect_ratios = {'conv4_3': [1.54719497498501, 1.49927167625187, 6.85407259539126,3.90017628205128,1.08222360161625,1.01875795512884,2.71517599378192]}

        #28*4
        fmap_dims = {'conv4_3': 10}
        obj_scales = {'conv4_3': [0.38, 0.14, 0.28, 0.11, 0.33,0.08,0.16,0.12,0.10,0.23,0.36]}
        aspect_ratios = {'conv4_3': [0.99, 1.33, 1.96, 2.13, 1.45,4.00,1.004,1.71,2.80,2.95,1.21]}

        #28
        # fmap_dims = {'conv4_3': 3}
        # obj_scales = {'conv4_3': [0.5063, 0.5874, 0.2848, 0.3808, 0.6632]}
        # aspect_ratios = {'conv4_3': [1.820, 1.4294, 5.754, 3.2327, 1.040]}

        fmaps = list(fmap_dims.keys())

        prior_boxes = []

        for k, fmap in enumerate(fmaps):
            for i in range(fmap_dims[fmap]):
                for j in range(fmap_dims[fmap]):
                    cx = (j + 0.5) / fmap_dims[fmap]
                    cy = (i + 0.5) / fmap_dims[fmap]
                    for loop in range(len(aspect_ratios[fmap])):
                        ratio = aspect_ratios[fmap][loop]
                        scale = obj_scales[fmap][loop]
                        prior_boxes.append(
                            [cx, cy, scale * np.sqrt(ratio), scale / np.sqrt(ratio)])

                        # For an aspect ratio of 1, use an additional prior whose scale is the geometric mean of the
                        # scale of the current feature map and the scale of the next feature map
                        # if ratio == 1.:
                        #     try:
                        #         additional_scale = np.sqrt(obj_scales[fmap] * obj_scales[fmaps[k + 1]])
                        #     # For the last feature map, there is no "next" feature map
                        #     except IndexError:
                        #         additional_scale = 1.
                        #     prior_boxes.append([cx, cy, additional_scale, additional_scale])

        prior_boxes = torch.FloatTensor(prior_boxes).to(device)  # (8732, 4)
        prior_boxes.clamp_(0, 1)  # (8732, 4)

        return prior_boxes



