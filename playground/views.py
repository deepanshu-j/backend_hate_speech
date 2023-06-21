# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.shortcuts import render
from django.http import HttpResponse
# Create your views here.
import requests
import pickle

import re
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.preprocessing import StandardScaler    
from sentence_transformers import SentenceTransformer

input_shape = 384
file_path = 'playground\clf5.pth'
filename = 'playground\scaler2.sav'
scaler = pickle.load(open(filename, 'rb'))

def clean_text(txt):
  txt = txt.lower() #lowercase
  txt = re.sub(r"[^a-zA-Z0-9' ]", ' ', txt) #remove special characters
  txt = re.sub(r' +', ' ', txt) #remove extra spaces
  return txt

class Embedding():
  def __init__(self):
    pass
  
  def CreateSentenceEmbeddings(self, corpus, transformer = 'all-MiniLM-L6-v2'):
    model = SentenceTransformer(transformer)
    embedded_vector = model.encode(corpus)
    return embedded_vector

class ANN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            
            nn.Linear(16,8),
            nn.ReLU(),
            
            nn.Linear(8,8),
            nn.ReLU(),
            
            nn.Dropout(p=0.1),
            nn.Linear(8,1),
            nn.Sigmoid(),
        )
        
    def forward(self, x):
        x = self.model(x)
        return x

model = ANN(input_shape)
model.load_state_dict(torch.load(file_path))
encoder = Embedding()

def sentiment_analysis(text):
    text = clean_text(text)
    encoded_corpus = encoder.CreateSentenceEmbeddings([text], transformer = 'all-MiniLM-L6-v2')
    encoded_corpus = scaler.transform(encoded_corpus)
    X = torch.tensor(encoded_corpus).float()
    print("model(X):" , model(X))
    output = (model(X) > 0.7).int().squeeze()
    if output.item() == 1:
        return '{"val": "hate speech"}'
    return '{"val": "non hate speech"}'

def say_hello(req):
    st = req.GET.get("q", "")

    if "kidnap" in st:
        return HttpResponse('{"val": "hate speech"}')
    
    if "kill" in st:
        return HttpResponse('{"val": "hate speech"}')

    txt = sentiment_analysis(st)
    print(st, txt)
    return HttpResponse(txt)


# Dataset used to train this model 
# https://www.kaggle.com/datasets/usharengaraju/dynamically-generated-hate-speech-dataset?resource=download


# url = "https://api.apilayer.com/sentiment/analysis"

# payload = "I will kill you".encode("utf-8")
# headers= {
#   "apikey": "DDpyFGteym0eHdU7NwnI4uqnKkyU6jkl"
# }

# response = requests.request("POST", url, headers=headers, data = payload)

# status_code = response.status_code
# result = response.text