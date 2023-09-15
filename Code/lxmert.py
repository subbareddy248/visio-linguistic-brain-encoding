#!/usr/bin/env python
# coding: utf-8
import json
import numpy as np
import pandas as pd
import os
from transformers import BertTokenizer, VisualBertModel
from transformers import ViTFeatureExtractor, ViTModel, LxmertTokenizer, LxmertModel
import requests
from torch import nn

import torch
import clip
from PIL import Image

tokenizer = LxmertTokenizer.from_pretrained('unc-nlp/lxmert-base-uncased')
model = LxmertModel.from_pretrained('unc-nlp/lxmert-base-uncased')


import pandas as pd
sub1file = pd.read_csv('./stim_list/stim_lists/CSI01_stim_lists.txt', sep='\n',header=None)


with open('COCO_images_captions.json') as f:
    data = json.load(f)

text_sent = []
for i in sub1file[0]:
    if 'COCO_train' in i or 'rep_COCO_train' in i:
        i = i.replace('rep_','')
        text_sent.append(data[i][0])

text_imagenet = open('./BOLD5000_Stimuli/Image_Labels/imagenet_final_labels.txt', 'r')
lines = text_imagenet.readlines()
text_imagenet_data = {}
for line in lines:
    if line.split(' ',1)[0].strip() not in text_imagenet_data:
        text_imagenet_data[line.split(' ',1)[0].strip()] = line.split(' ',1)[1].strip()


img_feat = np.load('./coco_frcnn.npy')
img_feat1 = np.load('./imagenet_bold_frcnn.npy')
img_feat_boxes = np.load('./coco_frcnn_boxes.npy')
img_feat_boxes1 = np.load('./imagenet_bold_frcnn_boxes.npy')


text_sent = []
img_feat2 = []
img_feat_boxes2 = []
count = 0
count1 = 0
for i in sub1file[0]:
    i = i.replace('rep_','')
    if 'COCO_train' in i or 'rep_COCO_train' in i:
        text_sent.append(data[i][0])
        img_feat2.append(img_feat[count])
        img_feat_boxes2.append(img_feat_boxes[count])
        count+=1
    elif 'n0' in i or ('n1' in i and 'n1.' not in i and 'n11.' not in i):
        #print(i.split('_')[0])
        text_sent.append(text_imagenet_data[i.split('_')[0]])
        img_feat2.append(img_feat1[count1])
        img_feat_boxes2.append(img_feat_boxes1[count1])
        count1+=1
    else:
        text_sent.append(i.split('.')[0][:-1])
        img_feat2.append(img_feat1[count1])
        img_feat_boxes2.append(img_feat_boxes1[count1])
        count1+=1

img_feat2 = np.array(img_feat2)
img_feat_boxes2 = np.array(img_feat_boxes2)
print(img_feat2.shape, img_feat_boxes2.shape)


language_output = []
language_avg_output = []
vision_output = []
vision_avg_output = []
pooled_output = []
language_hidden_states = []
vision_hidden_states = []
for i in np.arange(img_feat2.shape[0]):
    inputs = tokenizer(text_sent[i], return_tensors="pt",padding=True)
    visual_embeds = torch.Tensor(img_feat2[i].reshape(1,img_feat2[i].shape[0],img_feat2[i].shape[1]))
    visual_token_type_ids = torch.Tensor(img_feat_boxes2[i].reshape(1,img_feat_boxes2[i].shape[0],img_feat_boxes2[i].shape[1]))
    visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)
    inputs.update({
     "visual_feats": visual_embeds,
     "visual_pos": visual_token_type_ids,
     "visual_attention_mask": visual_attention_mask
     })
    outputs = model(**inputs, output_attentions=True, output_hidden_states=True)
    language_output.append(outputs['language_output'].detach().numpy())
    language_avg_output.append(nn.functional.adaptive_avg_pool2d(outputs['language_output'], (1,768)).detach().numpy())
    vision_output.append(outputs['vision_output'].detach().numpy())
    vision_avg_output.append(nn.functional.adaptive_avg_pool2d(outputs['vision_output'], (1,768)).detach().numpy())
    pooled_output.append(outputs['pooled_output'].detach().numpy())
    language_hidden_states.append(list(outputs['language_hidden_states']))
    vision_hidden_states.append(list(outputs['vision_hidden_states']))

language_avg_output = np.array(language_avg_output)
print(language_avg_output.shape)

np.save('lxmert_bold5000_language_avg',language_avg_output.reshape(language_avg_output.shape[0],language_avg_output.shape[3]))

vision_avg_output = np.array(vision_avg_output)
print(vision_avg_output.shape)

np.save('lxmert_bold5000_vision_avg',vision_avg_output.reshape(vision_avg_output.shape[0],vision_avg_output.shape[3]))

pooled_output = np.array(pooled_output)
pooled_output = pooled_output.reshape(pooled_output.shape[0], pooled_output.shape[2])

np.save('lxmert_bold5000_common',pooled_output)

np.save('lxmert_imagenet_language',np.array(language_output))
np.save('lxmert_imagenet_vision',np.array(vision_output))