#!/usr/bin/env python
# coding: utf-8

import json
import numpy as np
import pandas as pd
import os
from torch import nn

from transformers import ViTFeatureExtractor, ViTModel, DeiTFeatureExtractor, DeiTModel, BeitFeatureExtractor, BeitModel
from PIL import Image


feature_extractor = BeitFeatureExtractor.from_pretrained('microsoft/beit-base-patch16-224-pt22k-ft22k')
model = BeitModel.from_pretrained('microsoft/beit-base-patch16-224-pt22k-ft22k')

sub1file = pd.read_csv('./stim_list/stim_lists/CSI01_stim_lists.txt', sep='\n',header=None)


img_feat = []
img_avg1 = []
for i in sub1file[0]:
    i = i.replace('rep_','')
    if 'COCO' in i:
        image = Image.open("BOLD5000_Stimuli/Scene_Stimuli/Presented_Stimuli/COCO/"+str(i))
        inputs = feature_extractor(images=image, return_tensors="pt")
        outputs = model(**inputs, output_hidden_states=True)
    elif 'n0' in i or ('n1' in i and 'n1.' not in i and 'n11.' not in i):
        image = Image.open("BOLD5000_Stimuli/Scene_Stimuli/Presented_Stimuli/ImageNet/"+str(i))
        inputs = feature_extractor(images=image, return_tensors="pt")
        outputs = model(**inputs, output_hidden_states=True)
    else:
        image = Image.open("BOLD5000_Stimuli/Scene_Stimuli/Presented_Stimuli/Scene/"+str(i))
        inputs = feature_extractor(images=image, return_tensors="pt")
        outputs = model(**inputs, output_hidden_states=True)
        
    img_feat.append(outputs['last_hidden_state'].detach().numpy())
    img_avg1.append(outputs['pooler_output'].detach().numpy())   

img_avg = np.array(img_avg1)
print(img_avg.shape)

np.save('beit_img_feat',np.reshape(img_avg,(img_avg.shape[0],img_avg.shape[2])))


img_feat = np.array(img_feat)
img_feat = np.mean(img_feat,axis=2)
np.save('beit_img_feat_final',img_feat)