#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import os
import librosa
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow_addons.optimizers import CyclicalLearningRate
from preprocess import generate_features, pre_trained_model


# In[6]:


batch_size = 16
epochs = 30
max_lr = 3e-5 ##
base_lr = max_lr/100
cycles = 4

iterations = round(3837/batch_size*epochs)
iterations = list(range(0,iterations+1))
step_size = len(iterations)/(cycles)

cyclical_lr = CyclicalLearningRate(
                     initial_learning_rate=base_lr,
                     maximal_learning_rate=max_lr,
                     step_size=step_size,
                     scale_fn=lambda x: 1 / (2.0 ** (x - 1)),
                     scale_mode='cycle')


# In[7]:


input_dir = './private_test'

def is_wav(x):
    return x.endswith('.wav')

file_filenames = [x for x in sorted(os.listdir(input_dir))
                   if is_wav(x)]
private_img = []
sr = 8000
for name in tqdm(file_filenames):

    file_path = os.path.join(input_dir, name)
    y,sr=librosa.load(file_path,sr=sr) 
    image = generate_features(y)
    private_img.append(image)


private_img = np.array(private_img)
private_img = private_img.astype('float32')
private_prob = np.zeros(shape=(private_img.shape[0],6))

for i in tqdm(range(0,5)):
    model = pre_trained_model()
    optimizer = tfa.optimizers.RectifiedAdam(learning_rate = cyclical_lr)
    loss      = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss,metrics=['accuracy'])
    model.load_weights('./weight/K-fold_%s_V11_b0' % i)
    prob = model.predict(private_img)
    private_prob += prob
    del model
    
private_prob = private_prob / 5
df = pd.read_csv('./public.csv')
df.loc[10000:30000,['Barking', 'Howling', 'Crying', 'COSmoke', 'GlassBreaking','Other']] = private_prob
df.to_csv('./final_submission.csv', index=False)  

