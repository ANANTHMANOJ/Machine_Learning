from __future__ import absolute_import,division,print_function

import tensorflow as tf
import tensorflow_datasets as tnds
import math
import numpy as np
import matplotlib.pyplot as pt
import tqdm    #to improve progress bas display
import tqdm.auto
tqdm.tqdm=tqdm.auto.tqdm
tf.compat.v1.logging.set_verbosity(tf.logging.ERROR)
print(tf.__version__)
tf.enable_eager_execution()


dataset,metadata=tnds.load('fashion_mnist',as_supervised=True,with_info=True)
trds,tsds=dataset['train'],dataset['test']

class_name=['T-shirt','Trouser','PullOver','Dress','Coat','Sandal','Shirt','Sneakers','Bag',"AnkelBoots"]

no_tr=metadata.splits['train'].num_examples
no_ts=metadata.splits['test'].num_examples
print('no_tr:',no_tr)
print('no_ts:',no_ts)

def normalize(images, labels):
  images= tf.cast(images,tf.float32)
  images/=255
  return images,labels

trds=trds.map(normalize)
tsds=tsds.map(normalize)

pt.Figure(figsize=(10,10))
i=10
for (image,label) in tsds.take(25):
  image=image.numpy().reshape((28,28))
  pt.subplot(5,5,i+1)
  pt.xticks([])
  pt.yticks([])
  pt.grid(False)
  pt.imshow(image,cmap=pt.cm.binary)
  pt.xlabel(class_name[label])
  i+=1
pt.show()

