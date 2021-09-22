import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np
from dataLoaders import DigitData
from config import *
from funcs import get_dict_from_class, updateMetricsSheet
from models import FeatureExtractor
from itertools import permutations,combinations
from tqdm import tqdm
from torch.utils.data import DataLoader
pixel=['pixel'+str(i) for i in range(784)]
import tensorflow as tf
import random
import matplotlib.pyplot as plt
model=tf.keras.applications.vgg16.VGG16(include_top=False,weights='imagenet',input_shape=(32,32,3))
model.summary()
def get_submodels(layer_name):
  return tf.keras.models.Model(model.input,model.get_layer(layer_name).output)
get_submodels('block1_conv2').summary()
def create_image(im):
  data_load = DigitData(**get_dict_from_class(DataLoad1))
  data = data_load.data
  image=np.array(data.iloc[im]['pixel0':'pixel783']).reshape((28,28))
  image=np.pad(image,[(0,4),(0,4)],mode='constant')
  image=np.stack((image,image,image)).swapaxes(0,2)
  # for i in range(784):
  return tf.constant(image,dtype='float32')
def plot_image(image,title='random'):
  image=image-tf.math.reduce_min(image)
  image=image/tf.math.reduce_max(image)
  plt.imshow(image)
  plt.show()
image =create_image(0)
#plot_image(image)
def visualize_filter(layer_name,f_index=None,iters=1):
  submodel=get_submodels(layer_name)
  num_filters=submodel.output.shape[-1]

  if f_index is None:
    f_index=random.randint(0,num_filters-1)
  assert num_filters>f_index,'f_index out of bounds'
  image=create_image(0)
  verbose_step=1#int(iters/10)
  for i in range(0,iters):
    with tf.GradientTape() as tape:
      tape.watch(image)
      out = submodel(tf.expand_dims(image,axis=0))[:,:,:,f_index]
      loss = tf.math.reduce_mean(out)
    grads=tape.gradient(loss,image)
    grads=tf.math.l2_normalize(grads)
    image+=grads*10
    if (i+1) % verbose_step == 0:
      print(f'iteration: {i+1},Loss:{loss.numpy():4f}')
  plot_image(image,f'{layer_name},{f_index}')
  print([layer.name for layer in model.layers if 'conv' in layer.name])
layer_name = 'block1_conv1'  # @param ['block1_conv1', 'block1_conv2', 'block2_conv1', 'block2_conv2', 'block3_conv1', 'block3_conv2]
visualize_filter(layer_name,iters=1000)