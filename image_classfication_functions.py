# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 15:49:07 2024

@author: rishit.somvanshi
"""
# import libraries for file processing
import os
import random
import zipfile

# libraries for image processing
import cv2
import numpy as np
import pandas as pd

# pytorch libraries
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split

# skorch wrapper classes and functions
from skorch import NeuralNetClassifier
from skorch.dataset import ValidSplit
from skorch.callbacks import LRScheduler, Checkpoint
from skorch.callbacks import Freezer, EarlyStopping

# albumentations for image augmentation
import albumentations
from albumentations import pytorch

# for multiprocessing
import multiprocessing as mp

# plot graphs for data exploration
import matplotlib.pyplot as plt

from transformers import pipeline

## CLIP ZERO SHOT IMAGE CLASSIFICATION

model_checkpoint = "openai/clip-vit-large-patch14"
classifier = pipeline(model = model_checkpoint,
                      task = "zero-shot-image-classification", device = ['cuda' if torch.cuda.is_available() else 'cpu'][0])

def image_has_text(image):
    
    class_names = ['figure_has_text', 'figure_has_no_text']
    predictions = classifier(image, candidate_labels = class_names)
    
    label = predictions[0]['label']
    prob = predictions[0]['score']
    
    if(prob > 0.5) & (label == class_names[0]):
        return True
    
    return False

# DENSENET CLASSIFICATION

# declaration of constant variables
batch_size = 128
num_workers = mp.cpu_count()
img_size = 224
n_classes = 8

# callback functions for models

# DenseNet169
# callback for Reduce on Plateau scheduler
lr_scheduler = LRScheduler(policy='ReduceLROnPlateau',
                                    factor=0.5, patience=1)
# callback for saving the best on validation accuracy model
checkpoint = Checkpoint(f_params='best_model_densenet169.pkl',
                                 monitor='valid_acc_best')
# callback for freezing all layer of the model except the last layer
freezer = Freezer(lambda x: not x.startswith('model.classifier'))
# callback for early stopping
early_stopping = EarlyStopping(patience=5)

# class which uses DenseNet169 pretrained model
# + added custom classifier in the last layer
class DenseNet169(nn.Module):
    def __init__(self, output_features, num_units=512, drop=0.5,
                 num_units1=512, drop1=0.5):
        super().__init__()
        model = torchvision.models.densenet169(pretrained=True)
        n_inputs = model.classifier.in_features
        model.classifier = nn.Sequential(
                                nn.Linear(n_inputs, num_units),
                                nn.ReLU(),
                                nn.Dropout(p=drop),
                                nn.Linear(num_units, num_units1),
                                nn.ReLU(),
                                nn.Dropout(p=drop1),
                                nn.Linear(num_units1, output_features))
        self.model = model

    def forward(self, x):
        return self.model(x)

# NeuralNetClassifier for based on DenseNet169 with custom parameters
densenet = NeuralNetClassifier(
    # pretrained DenseNet169 + custom classifier
    module=DenseNet169,
    module__output_features=n_classes,
    # criterion
    criterion=nn.CrossEntropyLoss,
    # batch_size = 128
    batch_size=batch_size,
    # number of epochs to train
    max_epochs=100,
    # optimizer Adam used
    optimizer=torch.optim.Adam,
    optimizer__lr = 0.001,
    optimizer__weight_decay=1e-6,
    # shuffle dataset while loading
    iterator_train__shuffle=True,
    # load in parallel
    iterator_train__num_workers=num_workers,
    # stratified kfold split of loaded dataset
    train_split=ValidSplit(cv=5, stratified=True),
    # callbacks declared earlier
    callbacks=[lr_scheduler, checkpoint, freezer, early_stopping],
    # use GPU or CPU
    device="cuda:0" if torch.cuda.is_available() else "cpu"
)
    
# Load model and test
densenet.initialize()
densenet.load_params(f_params='best_model_densenet169.pkl')

def preprocess_image(image, img_size):
    # Load the image
    image = image.convert('RGB')
    image = np.array(image)
    # Display the image
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    # Apply the same transformations as during training
    transform = albumentations.Compose([
        albumentations.Resize(img_size, img_size),
        albumentations.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        pytorch.ToTensorV2()
    ])

    # Apply the transformations
    augmented = transform(image=image)
    image_tensor = augmented['image']

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor
    
def chart_classification(image):
    
    class_names = ['just_image', 'bar_chart', 'diagram', 'flow_chart', 'graph',
               'growth_chart', 'pie_chart', 'table']
    
    img_size = 224
    image_tensor = preprocess_image(image, img_size)
    
    densenet.module_.eval()
    
    with torch.no_grad():
        prediction = densenet.predict(image_tensor)[0]
        
    return class_names[prediction]

def pie_chart_classification(image):
    
    class_names = ['pie_chart', 'venn_diagram', 'round_figure']
    predictions = classifier(image, candidate_labels = class_names)
        
    label = predictions[0]['label']
    prob = predictions[0]['score']
    
    return label