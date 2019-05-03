
# coding: utf-8

# In[1]:


# importing libraries
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Add
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.applications import VGG19
from keras.models import Sequential, Model
from keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
import sys
import numpy as np
import os
import cv2
from keras.layers import add
from keras.applications import VGG19
import keras.backend as K
from tqdm import tqdm


# In[2]:



def res_block_gen(model, kernal_size, filters, strides):
    gen = model
    model = Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
    model = BatchNormalization(momentum = 0.5)(model)
    # Using Parametric ReLU
    model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(model)
    model = Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
    model = BatchNormalization(momentum = 0.5)(model)

    model = add([gen, model])

    return model


def plotLoss(epoch):
    plt.figure(figsize=(10, 8))
    plt.plot(dLosses, label='Discriminitive loss')
    plt.plot(gLosses, label='Generative loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('images/gan_cnn_64x64/dcgan_%d_loss_epoch.png' % epoch)
    
def up_sampling_block(model, kernal_size, filters, strides):
    #model = Conv2DTranspose(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
    model = Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
    model = UpSampling2D(size = 2)(model)
    model = LeakyReLU(alpha = 0.2)(model)

    return model

def discriminator_block(model, filters, kernel_size, strides):

    model = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same")(model)
    model = BatchNormalization(momentum = 0.5)(model)
    model = LeakyReLU(alpha = 0.2)(model)

    return model


def get_gan_network(discriminator, shape, generator, optimizer, vgg_loss):
    discriminator.trainable = False
    gan_input = Input(shape=shape)
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(inputs=gan_input, outputs=[x,gan_output])
    gan.compile(loss=[vgg_loss, "binary_crossentropy"],
                loss_weights=[1., 1e-3],
                optimizer=optimizer)

    return gan


def datagen(batchSize,filesList,filePath):
    while(True):
        files = np.random.choice(filesList,batchSize,replace=False)
        X_train_HR = []
        X_train_LR = []
        for file in files:
            image = cv2.imread(filePath + "/" + file)
            print(filePath + "/" + file)
            print(image)
            image_HR = cv2.resize(image,(224,224),interpolation = cv2.INTER_CUBIC)
            image_HR = image_HR / 255.0
            X_train_HR.append(image_HR)
            
            image_LR = cv2.resize(image,(56,56),interpolation = cv2.INTER_CUBIC)
            image_LR = image_LR / 255.0
            
            X_train_LR.append(image_LR)
            
        X_train_HR = np.array(X_train_HR)
        X_train_LR = np.array(X_train_LR)
        yield X_train_LR,X_train_HR
        
      
    

def plotGeneratedImages(epoch,datagen,generator, examples=100, dim=(1, 1), figsize=(2, 2)):
    randomDim = 100
    low,hit = next(datagen)
    generatedImages = generator.predict(low)
    fig = plt.figure(figsize=(10,1))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    for i in range(10):
        ax = fig.add_subplot(1, 10, i + 1, xticks=[], yticks=[])
        ax.imshow(generatedImages[i])
    fol = 'images/gan_cnn_64x64/faces/'
    if not os.path.exists(fol):
        os.makedirs(fol)
    plt.savefig(fol+'random_{:05d}.png'.format(epoch))
    
    
def saveModels(epoch,generator,discriminator):
    fol = 'models/gan_cnn_64x64/'
    if not os.path.exists(fol):
        os.makedirs(fol)
    generator.save(fol+'dcgan_generator_epoch_%d.h5' % epoch)
    discriminator.save(fol+'dcgan_discriminator_epoch_%d.h5' % epoch)


# In[3]:


class Generator():
    def __init__(self, noise_shape):
        self.noise_shape = noise_shape
        
    def gen(self):
        gen_Input = Input(shape = self.noise_shape)
        
        
        model = Conv2D(filters = 64,kernel_size = 9 ,strides=1,padding = "same") (gen_Input)
        model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(model)
        
        for i in range(16):
            model = res_block_gen(model,3,64,1)
            
            
        for i in range(2):
            model = up_sampling_block(model,3,256,1)
        
        model = Conv2D(filters = 3,kernel_size=9,strides=1,padding="same")(model)
        model = Activation('tanh')(model)
        
        generator_model = Model(inputs = gen_Input, outputs = model)
        
        return generator_model
    

class Disciminator():
    def __init__(self, image_shape):
        
        self.image_shape = image_shape
        
        
    def disc(self):
        disc_Input = Input(shape = self.image_shape)
        
        model = Conv2D(filters = 64,kernel_size = 3 ,strides=1,padding = "same") (disc_Input)
        model = LeakyReLU(alpha = 0.2)(model)
        
        
        model = discriminator_block(model, 64, 3, 2)
        model = discriminator_block(model, 128, 3, 1)
        model = discriminator_block(model, 128, 3, 2)
        model = discriminator_block(model, 256, 3, 1)
        model = discriminator_block(model, 256, 3, 2)
        model = discriminator_block(model, 512, 3, 1)
        model = discriminator_block(model, 512, 3, 2)
        
        model = Flatten()(model)
        
        model = Dense(1024)(model)
        model = LeakyReLU(alpha = 0.2)(model)
        
        model = Dense(1)(model)
        model = Activation('softmax')(model)
        
        disciminator_model = Model(inputs = disc_Input,output = model)
        
        
        return disciminator_model
    

        
class VGG(object):

    def __init__(self, image_shape):
        
        self.image_shape = image_shape

    # computes VGG loss or content loss
    def vgg_loss(self, y_true, y_pred):
    
        vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=self.image_shape)
        vgg19.trainable = False
        # Make trainable as False
        for l in vgg19.layers:
            l.trainable = False
        model = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv4').output)
        model.trainable = False
    
        return K.mean(K.square(model(y_true) - model(y_pred)))
    


# In[5]:


# extracting the dataset
# filePath_train_LR = 'Desktop/spring2019/Aml/proj3/dataset/DIV2K_train_LR_x8'
# filePath_val_LR = 'Desktop/spring2019/Aml/proj3/dataset/DIV2K_valid_LR_x8'
filePath_train_HR = '/home/vidyach/DIV2K_train_HR'
#filePath_val_HR = 'Desktop/spring2019/Aml/proj3/dataset/DIV2K_valid_HR'


train_HR = [f for f in os.listdir(filePath_train_HR) ]
#valid_HR = [f for f in os.listdir(filePath_val_HR) ]


# In[12]:


image_HR_shape = (224,224,3)
image_SR_shape = (56,56,3)
optimizer = Adam(0.0002, 0.5)
no_of_epoch = 2
batch_count = 2
d_loss = []
g_loss = []


def train():
    
    loss = VGG(image_HR_shape)
     
    #making the generator network to predict image by supplying low resolution input
    
    generator = Generator(image_SR_shape).gen()
    discriminator = Disciminator(image_HR_shape).disc()
    
   # vgg_optimizer = loss.get_optimizer()
    generator.compile(loss=loss.vgg_loss, optimizer=optimizer)
    discriminator.compile(loss="binary_crossentropy", optimizer=optimizer)
    
    
    gan = get_gan_network(discriminator, image_SR_shape, generator, optimizer, loss.vgg_loss)
    datagenObj = datagen(batch_count,train_HR,filePath_train_HR)
    
    
    for e in range(1,no_of_epoch):
        
        for _ in tqdm(range(batch_count)):
           
            lr_img,hr_img = next(datagenObj)
            fake_img = generator.predict(lr_img)
            
            #inputDisc = np.concatenate([hr_img,fake_img])
            
            real_data_Y = np.ones(batch_count) - np.random.random_sample(batch_count)*0.2
            fake_data_Y = np.random.random_sample(batch_count)*0.2
            
            # training the disciminator
            discriminator.trainable = True
            
            d_loss_real = discriminator.train_on_batch(hr_img, real_data_Y)
            d_loss_fake = discriminator.train_on_batch(fake_img, fake_data_Y)
            dloss = 0.5 * np.add(d_loss_fake, d_loss_real)
            
           # dloss = discriminator.train_on_batch(inputDisc, target_value)
            
            #training the generator
            #generator is frozen
            lr_img,hr_img = next(datagenObj)
            target_value = np.ones(batch_count)
            discriminator.trainable = False
            gan_loss = gan.train_on_batch(lr_img, [hr_img,target_value])
        
        d_loss.append(dloss)
        g_loss.append(gan_loss)
        
        if e == 1 or e % 100 == 0:
            saveModels(e,generator,discriminator)
            plotGeneratedImages(e,datagenObj,generator)
           # plotGeneratedImages(e)
#             saveModels(e)


# In[ ]:


train()


# In[11]:


plt.figure(figsize=(10, 8))
plt.plot(d_loss, label='Discriminitive loss')
plt.plot(g_loss, label='Generative loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('images/gan_cnn_64x64/dcgan_%d_loss_epoch.png' % no_of_epoch)

