#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import numpy as np
import pandas as pd
import os

import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

with open('COCO_images_captions.json') as f:
    data = json.load(f)


sub1file = pd.read_csv('./stim_list/stim_lists/CSI01_stim_lists.txt', sep='\n',header=None)

text_imagenet = open('./BOLD5000_Stimuli/Image_Labels/imagenet_final_labels.txt', 'r')
lines = text_imagenet.readlines()
text_imagenet_data = {}
for line in lines:
    if line.split(' ',1)[0].strip() not in text_imagenet_data:
        text_imagenet_data[line.split(' ',1)[0].strip()] = line.split(' ',1)[1].strip()

text_sent = []
for i in sub1file[0]:
    i = i.replace('rep_','')
    if 'COCO_train' in i or 'rep_COCO_train' in i:
        text_sent.append(data[i])
    elif 'n0' in i or ('n1' in i and 'n1.' not in i and 'n11.' not in i):
        #print(i.split('_')[0])
        text_sent.append(text_imagenet_data[i.split('_')[0]])
    else:
        text_sent.append(i.split('.')[0][:-1])

img_feat = []
text_feat = []
count=0
for i in sub1file[0]:
    i = i.replace('rep_','')
    if 'COCO_train' in i : 
        image = preprocess(Image.open("BOLD5000_Stimuli/Scene_Stimuli/Presented_Stimuli/COCO/"+str(i))).unsqueeze(0).to(device)
        text = clip.tokenize(data[i]).to(device)
    elif 'n0' in i or ('n1' in i and 'n1.' not in i and 'n11.' not in i):
        image = preprocess(Image.open("BOLD5000_Stimuli/Scene_Stimuli/Presented_Stimuli/ImageNet/"+str(i))).unsqueeze(0).to(device)
        text = clip.tokenize(text_sent[count]).to(device)
        count+=1
    else:
        image = preprocess(Image.open("BOLD5000_Stimuli/Scene_Stimuli/Presented_Stimuli/Scene/"+str(i))).unsqueeze(0).to(device)
        text = clip.tokenize(text_sent[count]).to(device)
        count+=1
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        img_feat.append(image_features.detach().numpy())
        text_feat.append(text_features.detach().numpy())

img_feat = np.array(img_feat)
np.save('img_feat_bold5000_clip',np.reshape(img_feat,(img_feat.shape[0],img_feat.shape[2])))