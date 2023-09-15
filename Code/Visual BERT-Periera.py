#!/usr/bin/env python
# coding: utf-8

import json
import numpy as np
import pandas as pd
import os
from transformers import BertTokenizer, VisualBertModel
from transformers import ViTFeatureExtractor, ViTModel
import requests
import torch.nn as nn
import torch

import torch
import clip
from PIL import Image


# In[3]:


model = VisualBertModel.from_pretrained("uclanlp/visualbert-vqa-coco-pre")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


img_feat = np.load('../vilbert-multi-task/periera_img1.npy')
img_feat2 = np.load('../vilbert-multi-task/periera_img2.npy')
img_feat = np.concatenate([img_feat,img_feat2], axis=0)
print(img_feat.shape)

import json
  
# Opening JSON file
f = open('concept2caption.json')
  
# returns JSON object as 
# a dictionary
data = json.load(f)


text_sent = []
for eachword in sorted(data['concept2caption'].keys()):
    #print(eachword)
    if len(data['concept2caption'][eachword])!=6:
        print(len(data['concept2caption'][eachword]))
        print(eachword)
    for eachsent in data['concept2caption'][eachword]:
        #print(eachsent)
        text_sent.append(eachsent)

remove_indices = [264, 379, 558, 610, 674, 675, 692, 758, 780, 782, 866, 897, 1005, 1009, 1013]


img_feat = np.delete(img_feat, remove_indices, axis=0)
print(img_feat.shape)


# In[39]:


language_output = []
language_avg_output = []
vision_output = []
vision_avg_output = []
pooled_output = []
language_hidden_states = []
vision_hidden_states = []
for i in np.arange(img_feat.shape[0]):
    inputs = tokenizer(text_sent[i], return_tensors="pt",padding=True)
    visual_embeds = torch.Tensor(img_feat[i].reshape(1,img_feat[i].shape[0],img_feat[i].shape[1]))
    visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
    visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)
    inputs.update({
     "visual_embeds": visual_embeds,
     "visual_token_type_ids": visual_token_type_ids,
     "visual_attention_mask": visual_attention_mask
     })
    outputs = model(**inputs, output_attentions=True, output_hidden_states=True)
    language_output.append(outputs['language_output'].detach().numpy())
    language_avg_output.append(np.mean(outputs['language_output'].detach().numpy(),axis=1))
    vision_output.append(outputs['pooler_output'].detach().numpy())
    vision_avg_output.append(np.mean(outputs['last_hidden_state'].detach().numpy(), axis=1))
    pooled_output.append(outputs['pooled_output'].detach().numpy())
    language_hidden_states.append(list(outputs['language_hidden_states']))
    vision_hidden_states.append(list(outputs['vision_hidden_states']))



vision_embeddings = []
i = 0
for eachword in sorted(data['concept2caption'].keys()):
    lt = len(data['concept2caption'][eachword])
    vision_embeddings.append(np.mean(vision_output[i:i+lt],axis=0))
    i+=lt

vision_embeddings = np.array(vision_embeddings)
print(vision_embeddings.shape)

vision_avg_embeddings = []
i = 0
for eachword in sorted(data['concept2caption'].keys()):
    lt = len(data['concept2caption'][eachword])
    vision_avg_embeddings.append(np.mean(vision_avg_output[i:i+lt],axis=0))
    i+=lt

vision_avg_embeddings = np.array(vision_avg_embeddings)
print(vision_avg_embeddings.shape)

vision_avg_output = np.array(vision_avg_output)
print(vision_avg_output.shape)

np.save('visualbert_coco_vision_pool',vision_avg_output.reshape(vision_avg_output.shape[0],vision_avg_output.shape[2]))

vision_output = np.array(vision_output)
print(vision_output.shape)

np.save('visualbert_coco_vision_avg',vision_output.reshape(vision_output.shape[0],vision_output.shape[2]))

np.save('visualbert_periera_lastlayer',vision_embeddings.reshape(vision_embeddings.shape[0],vision_embeddings.shape[2]))

np.save('visualbert_periera_lastlayer_avgpatch',vision_avg_embeddings.reshape(vision_avg_embeddings.shape[0],vision_avg_embeddings.shape[2]))