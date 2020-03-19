import sys
import numpy as np
from scipy.misc import imread
import os
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rng

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as keras_backend
import tensorflow.keras.utils as utils

# Other dependencies
import random
import sys
import time
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True

## Set logging dir for Tensorboard
logging_dir_n = 0

## The directory where data path is present
data_path = r"C:\Users\kusht\Documents\Other Courses\Fellowship AI"
train_path = os.path.join(data_path, 'images_background')
validation_path = os.path.join(data_path,'images_evaluation')

def load_images_from_directory(path,n=0):
    X=[]
    Y=[]
    ## We load every alphabet seperately and append that to one tensor
    for alphabet in os.listdir(path):
        if not alphabet.startswith('.'):
            print("loading alphabet: " + alphabet)
            alphabet_path = os.path.join(path,alphabet)

            ## Each character in alphabet is in a separate folder
            for letter in os.listdir(alphabet_path):
                if not letter.startswith('.'):
                    category_images=[]
                    letter_path = os.path.join(alphabet_path, letter)

                    if not os.path.isdir(letter_path):
                        continue
                    label = int(letter[-2:])
                    Y.append(label)

                    ## Read every image in this directory
                    for filename in os.listdir(letter_path):
                        if not filename.startswith('.'):
                            image_path = os.path.join(letter_path, filename)
                            image = imread(image_path)

                            ### Image preprocessing!
                            image = image/255
                            image = 1-image

                            category_images.append(image)

                    try:
                        X.append(np.stack(category_images))
                    #edge case  - last one
                    except ValueError as e:
                        print(e)
                        print("error - category_images:", category_images)
    
    X = np.stack(X)
    return X, Y

print("Loading training set")
Xtrain, Ytrain = load_images_from_directory(train_path)
print(Xtrain.shape)
print(len(Ytrain))

print("Now loading evaluation set")
Xval, Yval = load_images_from_directory(validation_path)
print(Xval.shape)


class OmniModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.inp = keras.layers.Conv2D(64, (3,3), activation='relu', strides = (3,3), input_shape=(105, 105,1)) #change to 28
        self.conv = keras.layers.Conv2D(64, (3,3), activation='relu', strides = (3,3))
        self.flat = keras.layers.Flatten()
        self.out = keras.layers.Dense(20, activation='softmax') #need to change dynamically!!
        
    def forward(self, x): #add bn
        x = self.inp(x)
        x = self.conv(x)
        x = self.conv(x)
        x = self.flat(x)
        x = self.out(x)
        return x


def loss_function(pred_y, y):
  pred_y = utils.to_categorical(pred_y, num_classes=y.shape[0])
  return keras.losses.categorical_crossentropy(y, pred_y) 

def np_to_tensor(list_of_numpy_objs):
    return (tf.convert_to_tensor(obj) for obj in list_of_numpy_objs)
    

def compute_loss(model, x, y, loss_fn=loss_function):
    logits = model.forward(x)
    ce = loss_fn(y, logits)
    return ce, logits


def compute_gradients(model, x, y, loss_fn=loss_function):
    with tf.GradientTape() as tape:
        loss, _ = compute_loss(model, x, y, loss_fn)
    return tape.gradient(loss, model.trainable_variables), loss


def apply_gradients(optimizer, gradients, variables):
    optimizer.apply_gradients(zip(gradients, variables))

    
def train_batch(x, y, model, optimizer):
    tensor_x, tensor_y = np_to_tensor((x, y))
    gradients, loss = compute_gradients(model, tensor_x, tensor_y)
    apply_gradients(optimizer, gradients, model.trainable_variables)
    return loss

def copy_model(model, x):
    '''Copy model weights to a new model.
    
    Args:
        model: model to be copied.
        x: An input example. This is used to run
            a forward pass in order to add the weights of the graph
            as variables.
    Returns:
        A copy of the model.
    '''
    copied_model = OmniModel()
    
    # If we don't run this step the weights are not "initialized"
    # and the gradients will not be computed.
    copied_model.forward(tf.convert_to_tensor(x))
    
    copied_model.set_weights(model.get_weights())
    return copied_model

def train_maml(model, epochs, xtrain, ytrain, lr_inner=0.01, batch_size=1, log_steps=1000):
    '''Train using the MAML setup.
    
    The comments in this function that start with:
        
        Step X:
        
    Refer to a step described in the Algorithm 1 of the paper.
    
    Args:
        model: A model.
        epochs: Number of epochs used for training.
        dataset: A dataset used for training.
        lr_inner: Inner learning rate (alpha in Algorithm 1). Default value is 0.01.
        batch_size: Batch size. Default value is 1. The paper does not specify
            which value they use.
        log_steps: At every `log_steps` a log message is printed.
    
    Returns:
        A strong, fully-developed and trained maml.
    '''
    optimizer = keras.optimizers.Adam()
    
    # Step 2: instead of checking for convergence, we train for a number
    # of epochs
    for _ in range(epochs):
        total_loss = 0
        losses = []
        start = time.time()
        # Step 3 and 4
        for i, x in enumerate(random.sample(list(xtrain), len(xtrain))):
            y = ytrain[i]
            x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1) #can also use expand dims!
            model.forward(x)  # run forward pass to initialize weights
            with tf.GradientTape() as test_tape:
                # test_tape.watch(model.trainable_variables)
                # Step 5: Calculate loss
                with tf.GradientTape() as train_tape:
                    train_loss, _ = compute_loss(model, x, y)
                # Step 6: Calculate gradient 
                gradients = train_tape.gradient(train_loss, model.trainable_variables)
                k = 0
                model_copy = copy_model(model, x)
                #Step 7: Compute adapted parameterr using gradient on copy_model
                for j in range(len(model_copy.layers) - 2):
                    model_copy.layers[j].kernel = tf.subtract(model.layers[j].kernel,
                                tf.multiply(lr_inner, gradients[k]))
                    model_copy.layers[j].bias = tf.subtract(model.layers[j].bias,
                                tf.multiply(lr_inner, gradients[k+1]))
                    k += 2
                # Step 8: Compute loss on model_copy
                test_loss, logits = compute_loss(model_copy, x, y)
            # Step 9: Compute gradient on model_copy and use optimizer
            gradients = test_tape.gradient(test_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            # Logs
            total_loss += test_loss
            loss = total_loss / (i+1.0)
            losses.append(loss)
            
            if i % log_steps == 0 and i > 0:
                print('Step {}: loss = {}, Time to run {} steps = {}'.format(i, loss, log_steps, time.time() - start))
                start = time.time()

maml = OmniModel()
train_maml(maml, 1, Xtrain, Ytrain)

#Need to one shot test for results!