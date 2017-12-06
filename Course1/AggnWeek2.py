import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
import lr_utils
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset
%matplotlib inline 

#load the dataset 
train_set_x_orig,train_set_y,test_set_x_orig,test_set_y,classes = load_dataset()

#let us plt a image 
index = 2
plt.imshow(train_set_x_orig[index])
print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")

#the dataset train_set_x_orig is of size [num_of_example*Nx*Ny*3), that is a 4d vector of 3RGB channels , we will now reshape it to a 2D vector where each column will represen the indidual example 
X = train_set_x_orig
train_set_flatten = train_set_x_orig.reshape(X.shape[1]*X.shape[2]*X.shape[3],X.shape[0])
X= test_set_x_orig
test_set_x_flatten = test_set_x_orig.reshape(X.shape[1]*X.shape[2]*X.shape[3],X.shape[0])

#now generally in image prcocesseing it is a good practice to normaize ur data. in IP u just noramlize by dividing the umage pixels by 255
train_set_x = train_set_flatten/255
test_set_x = test_set_x_flatten/255
