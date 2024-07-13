#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
import numpy as np
import skimage.io
from matplotlib import pyplot as plt
from patchify import patchify
from PIL import Image
np.random.seed(0) 


#CLAHE
def clahe_equalized(imgs):    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))    
    imgs_equalized = clahe.apply(imgs)
    return imgs_equalized


path1 = '../Data/train/image/' #training images directory
path2 = '../Data/train/mask/' #training masks directory

image_dataset = []
mask_dataset = [] 

patch_size = 256

images = sorted(os.listdir(path1)) 
for i, image_name in enumerate(images):  
   if image_name.endswith(".png"):                   
       image = skimage.io.imread(path1+"/"+image_name)  #Read image
       image = image[:,:,1] #selecting green channel
       image = clahe_equalized(image) #applying CLAHE
       SIZE_X = (image.shape[1]//patch_size)*patch_size #getting size multiple of patch size
       SIZE_Y = (image.shape[0]//patch_size)*patch_size #getting size multiple of patch size
       image = Image.fromarray(image)        
       image = image.resize((SIZE_X, SIZE_Y)) #resize image       
       image = np.array(image) 
       patches_img = patchify(image, (patch_size, patch_size), step=patch_size)  #create patches(patch_sizexpatch_sizex1)
            
       for i in range(patches_img.shape[0]):
           for j in range(patches_img.shape[1]):                        
               single_patch_img = patches_img[i,j,:,:]                 
               single_patch_img = (single_patch_img.astype('float32')) / 255.                    
               image_dataset.append(single_patch_img)

masks = sorted(os.listdir(path2))  
for i, mask_name in enumerate(masks):  
    if mask_name.endswith(".png"):                  
        mask = skimage.io.imread(path2+"/"+mask_name)   #Read masks
        SIZE_X = (mask.shape[1]//patch_size)*patch_size #getting size multiple of patch size
        SIZE_Y = (mask.shape[0]//patch_size)*patch_size #getting size multiple of patch size
        mask = Image.fromarray(mask)        
        mask = mask.resize((SIZE_X, SIZE_Y))  #resize image
        mask = np.array(mask)
        patches_mask = patchify(mask, (patch_size, patch_size), step=patch_size)  #create patches(patch_sizexpatch_sizex1)
            
        for i in range(patches_mask.shape[0]):
            for j in range(patches_mask.shape[1]):                            
                single_patch_mask = patches_mask[i,j,:,:]
                single_patch_mask = (single_patch_mask.astype('float32'))/255. 
                mask_dataset.append(single_patch_mask) 
 
image_dataset = np.array(image_dataset)
mask_dataset =  np.array(mask_dataset)
image_dataset = np.expand_dims(image_dataset,axis=-1)
mask_dataset =  np.expand_dims(mask_dataset,axis=-1)



# In[2]:


#importing models
# from model import unetmodel, residualunet, attentionunet, attention_residualunet
from model import unetmodel, residualunet, attentionunet, residual_attentionunet

from tensorflow.keras.optimizers import Adam
from evaluation_metrics import IoU_coef,IoU_loss

IMG_HEIGHT = patch_size
IMG_WIDTH = patch_size
IMG_CHANNELS = 1
input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

optimizer = Adam(learning_rate=0.001)

model = residual_attentionunet(input_shape)
model.compile(optimizer = optimizer, loss= IoU_loss, metrics= ['accuracy', IoU_coef])


#splitting data into 70-30 ratio to validate training performance
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(image_dataset, mask_dataset, test_size=0.3, random_state=0)

#train model
history = model.fit(x_train, y_train, 
                    verbose=1,
                    batch_size = 16,
                    validation_data=(x_test, y_test ), 
                    shuffle=False,
                    epochs=150)


# In[3]:


model.save('./retina_AttResUnet.hdf5')


# In[4]:


print(history.history)


# In[5]:


#training-validation loss curve
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.figure(figsize=(7,5))
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'y', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
# plt.show()
plt.savefig('loss_curve.png')


#training-validation accuracy curve
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.figure(figsize=(7,5))
plt.plot(epochs, acc, 'r', label='Training Accuracy')
plt.plot(epochs, val_acc, 'y', label='Validation Accuracy')
plt.title('Training and validation accuracies')
plt.xlabel('Epochs')
plt.ylabel('IoU')
plt.legend()
# plt.show()
plt.savefig('accuracy_curve.png')

#training-validation IoU curve
iou_coef = history.history['io_u_coef']
val_iou_coef = history.history['val_io_u_coef']
plt.figure(figsize=(7,5))
plt.plot(epochs, iou_coef, 'r', label='Training IoU')
plt.plot(epochs, val_iou_coef, 'y', label='Validation IoU')
plt.title('Training and validation IoU coefficients')
plt.xlabel('Epochs')
plt.ylabel('IoU')
plt.legend()
# plt.show()
plt.savefig('validation_curve.png')

