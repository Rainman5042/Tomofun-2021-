#!/usr/bin/env python
# coding: utf-8

# In[16]:


import os
import cv2
import random
import librosa
import warnings
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import models
from tensorflow.keras.models import Model
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize
from sklearn.model_selection import StratifiedKFold
from tensorflow_addons.optimizers import CyclicalLearningRate
from preprocess import generate_features, train_pre, label_pre, mixup, pre_trained_model
warnings.filterwarnings('ignore')


# In[27]:


df = pd.read_csv('./meta_train.csv')
sr = 8000
_filenames  = df.Filename.values.tolist()
_labels     = df.Label.values.tolist()

full_file = []
full_label= []

for i in tqdm(range(len(_filenames))):
    filename = _filenames[i] + str(".wav")
    if filename != "train_01046.wav":
        y, sr = librosa.load('./train/'+str(filename),sr=sr)
        full_file.append(y)
        full_label.append(_labels[i])
        
full_file = np.array(full_file)
full_label= np.array(full_label)


# In[18]:


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


# In[19]:


def submit_NN(train_data, train_label, n_splits=5):
    
    count = 0

    train_acc = [] 
    test_acc  = []
    val_acc   = []
    roc_acc   = []
    
    
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=10)
    

    for train, val in folds.split(train_data, train_label):
        
        X_train = train_pre((train_data[train,:]), train=True)
        y_train = label_pre((train_label[train]), train=True)
        X_mix, y_mix  = mixup((train_data[train,:]),train_label[train])
        X_train = np.vstack((X_train,X_mix))
        y_train = np.vstack((y_train,y_mix))
        
        X_val   = train_pre((train_data[val,:]), train=False)
        y_val   = label_pre((train_label[val]), train=False)
        
        
        model     = pre_trained_model()
        optimizer = tfa.optimizers.RectifiedAdam(learning_rate = cyclical_lr)
        loss      = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        model.compile(optimizer=optimizer, loss=loss,metrics=['accuracy'])
        
        checkpoint_filepath = ('./weight/K-fold_%s_V11_b0' % count)
        
        ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
                        filepath=checkpoint_filepath,
                        save_weights_only=True,
                        monitor='val_accuracy',
                        mode='max',
                        save_best_only=True)

        history = model.fit(X_train, y_train, epochs=30,validation_data= (X_val, y_val), batch_size=16, verbose=1, 
                                callbacks=[ckpt_callback])
        
        model.load_weights('./weight/K-fold_%s_V11_b0' % count)
             

        preds_prob      = model.predict(X_val)

        val_eval = model.evaluate(X_val, y_val, verbose=0)
        roc_auc_ovr = roc_auc_score(y_val, preds_prob, multi_class="ovr",average="weighted")
        val_acc.append(val_eval[1])

        roc_acc.append(roc_auc_ovr)
        count += 1
        
        print('\n')
        print('K_fold[%s/%s]' % (count, n_splits))
        print('K_fold_Val loss:', val_eval[0])
        print('K_fold_Val accuracy:', val_eval[1])
        print("K_fold_ROC_AUC:", roc_auc_ovr)
        print('\n')
        del model, X_train, X_val, y_train, y_val
        
    print('Finish !')
    print('Mean val accuracy:', np.mean(val_acc))
    print("Mean ROC AUC accurancy:", np.mean(roc_acc))
    
    return np.mean(test_acc), np.mean(roc_acc)


# In[20]:


test_acc,roc_acc = submit_NN(full_file,full_label, n_splits=5)


# In[38]:


input_dir  = './public_test'
input_dir2 = './private_test'
def is_wav(x):
    return x.endswith('.wav')

file_filenames = [x for x in sorted(os.listdir(input_dir))
                   if is_wav(x)]

file_filenames2 = [x for x in sorted(os.listdir(input_dir2))
                   if is_wav(x)]
public_img = []
sr = 8000

for name in tqdm(file_filenames):

    file_path = os.path.join(input_dir, name)
    y,sr=librosa.load(file_path,sr=sr) 
    image = generate_features(y, noise=False)
    public_img.append(image)

public_img = np.array(public_img)
public_img = public_img.astype('float32')
public_prob = np.zeros(shape=(public_img.shape[0],6))

for i in tqdm(range(0,5)):
    model     = pre_trained_model()
    optimizer = tfa.optimizers.RectifiedAdam(learning_rate = cyclical_lr)
    loss      = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss,metrics=['accuracy'])
    model.load_weights('./weight/K-fold_%s_V11_b0' % i)
    public_prob += model.predict(public_img)

public_prob = public_prob / 5
public_prob = pd.DataFrame(public_prob)

total_name = file_filenames + file_filenames2

for i in tqdm(range(0,len(total_name))):
    total_name[i] = total_name[i].replace('.wav','')




total_name = pd.DataFrame(total_name)
total_data = pd.concat([total_name, public_prob], axis=1, ignore_index = True)
total_data.columns = ['Filename','Barking','Howling','Crying','COSmoke','GlassBreaking','Other']
total_data.loc[10000:30000,['Barking','Howling','Crying','COSmoke','GlassBreaking','Other']] = 0.1666

total_data.to_csv('./public.csv', index=False)  


# In[ ]:




