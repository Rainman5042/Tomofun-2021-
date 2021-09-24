
import mp
from mp.mixup_generator import MixupGenerator
import pandas as pd
import os
import random
import librosa
import librosa.display
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from tensorflow.keras.applications.efficientnet import preprocess_input, EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Input, GlobalAveragePooling2D
import warnings
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tqdm import tqdm

warnings.filterwarnings('ignore')



def spec_augment(spec: np.ndarray, num_mask=2, 
                 freq_masking_max_percentage=0.3, time_masking_max_percentage=0.3):

    spec = spec.copy()
    for i in range(num_mask):
        all_frames_num, all_freqs_num = spec.shape
        freq_percentage = random.uniform(0.0, freq_masking_max_percentage)

        num_freqs_to_mask = int(freq_percentage * all_freqs_num)
        f0 = np.random.uniform(low=0.0, high=all_freqs_num - num_freqs_to_mask)
        f0 = int(f0)
        spec[:, f0:f0 + num_freqs_to_mask] = 0

        time_percentage = random.uniform(0.0, time_masking_max_percentage)
        num_frames_to_mask = int(time_percentage * all_frames_num)
        t0 = np.random.uniform(low=0.0, high=all_frames_num - num_frames_to_mask)
        t0 = int(t0)
        spec[t0:t0 + num_frames_to_mask, :] = 0
    
    return spec




def mono_to_color(X, mean=None, std=None, norm_max=None, norm_min=None, eps=1e-6):
    # Standardize
    mean = mean or X.mean()
    std = std or X.std()
    Xstd = (X - mean) / (std + eps)
    _min, _max = Xstd.min(), Xstd.max()
    norm_max = norm_max or _max
    norm_min = norm_min or _min
    if (_max - _min) > eps:
        # Normalize to [0, 255]
        V = Xstd
        V[V < norm_min] = norm_min
        V[V > norm_max] = norm_max
        V = 255 * (V - norm_min) / (norm_max - norm_min)
        V = V.astype(np.uint8)
    else:
        # Just zero
        V = np.zeros_like(Xstd, dtype=np.uint8)
    return V


def generate_features(y, SpecAugment=False, noise=False):
    n_mels       = 224
    n_fft        = 20*224 
    hop_length   = 179
    sr = 8000
    # pre_emphasis
    pre_emphasis = 0.95
    y = np.append(y[0], y[1:] - pre_emphasis * y[:-1])
    
    if noise == True:
        augment = Compose([
                    AddGaussianNoise(min_amplitude=0.015, max_amplitude=0.015, p=1),
                    TimeStretch(min_rate=1.25, max_rate=1.25, p=1),
                    PitchShift(min_semitones=4, max_semitones=4, p=0),
                    Shift(min_fraction=0.5, max_fraction=0.5, p=1)])

        y = augment(samples=y, sample_rate=sr)
    
    # log-mei_spec + log-mei_spec*delta + log-mei_spec*delta*delta
    melspec     = librosa.feature.melspectrogram(y, sr, n_fft=n_fft,fmin=20, fmax=8000, hop_length=hop_length, n_mels=n_mels).T
    logmelspec  = librosa.amplitude_to_db(melspec)

    log_delta   = librosa.feature.delta(logmelspec)
    log_delta_x2= librosa.feature.delta(log_delta)
    
    
    logmelspec    = logmelspec.astype(np.float32)
    log_delta     = log_delta.astype(np.float32)
    log_delta_x2  = log_delta_x2.astype(np.float32)
    
    logmelspec   = mono_to_color(logmelspec)
    log_delta    = mono_to_color(log_delta)
    log_delta_x2 = mono_to_color(log_delta_x2)

    if SpecAugment == True:
        logmelspec   = spec_augment(logmelspec)
        log_delta    = spec_augment(log_delta)
        log_delta_x2 = spec_augment(log_delta_x2)



    image = np.dstack((logmelspec,log_delta))
    image = np.dstack((image,log_delta_x2))
    
    #image =cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
    image = preprocess_input(image)
    
    return image



def train_pre(y, train=False):
    train_img = []

    for i in tqdm(range(len(y))):
        if train==True:
            noise =  generate_features(y[i],  noise = True,  SpecAugment=False)
            aug_image = generate_features(y[i],  SpecAugment=True )
            train_img.append(aug_image)
            train_img.append(noise)
            
        image = generate_features(y[i])
        train_img.append(image)
    X_train = np.array(train_img)

    return X_train




def label_pre(label, train= False):
    train_label = []
    if train == True:
        for i in tqdm(range(len(label))):
            train_label.append(label[i])
            train_label.append(label[i])
            train_label.append(label[i])
    else:
        for i in tqdm(range(len(label))):
            train_label.append(label[i])
    
    y_train = np.array(train_label)    
    y_train_One_Hot = tf.keras.utils.to_categorical(y_train)
    return y_train_One_Hot






def mixup(train_list, label_list, alpha=0.2):
    train = train_pre(train_list)
    label = label_pre(label_list)
    training_generator = MixupGenerator(train,label, batch_size=32, alpha=alpha)()
    X_mix = []
    y_mix = []
    count = train.shape[0] // 32
    for i in tqdm(range(0,count+1)):
        X_mix_train, y_mix_label = next(training_generator)
        X_mix.append(X_mix_train)
        y_mix.append(y_mix_label)


    X_mix = np.vstack(X_mix)
    y_mix = np.vstack(y_mix)
    return  X_mix, y_mix




def pre_trained_model() :
    weight_path = "./efficientnetb0_notop.h5"

    base_model  = EfficientNetB0(input_shape=(224,224,3), weights=weight_path, include_top=False)
    base_model.trainable = True

    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    x = Dropout(0.2)(x)
    predictions = Dense(6, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)


    return model







