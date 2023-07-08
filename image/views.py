from django.shortcuts import render
from django.shortcuts import render
from django.http import StreamingHttpResponse, HttpResponse, FileResponse
from django.views.decorators import gzip
import cv2
import threading   
import os
import time
from PIL import Image
from tesserocr import PyTessBaseAPI
from pythainlp.tokenize import word_tokenize
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
from pythainlp.tag import *
from pythainlp import * 
import requests
import numpy as np
import json
import base64
from django.conf import settings
from django.http import JsonResponse
from bs4 import BeautifulSoup
# Create your views here.
from .form import ImageForm
from .models import Image

def index(request):
    if request.method == "POST":
        form=ImageForm(data=request.POST,files=request.FILES)
        if form.is_valid():
            form.save()
            obj=form.instance
            return render(request,"index.html",{"obj":obj})
    else:   
        form=ImageForm()
        img=Image.objects.all()
    return render(request,"index.html",{"img":img,"form":form})

def ocr(request):
    media_path = os.path.join(settings.MEDIA_ROOT, 'capture.jpg')
    if os.path.exists(media_path):
        with PyTessBaseAPI(path='C:/Users/User/anaconda3/share/tessdata_best-main',lang='tha+eng') as api:
            api.SetImageFile(media_path)
            text = api.GetUTF8Text()
            conf = api.AllWordConfidences()
            print(text)
            name = os.path.join(settings.NER_ROOT, 'thainer-corpus-v2-base-model')
            tokenizer = AutoTokenizer.from_pretrained(name)
            model = AutoModelForTokenClassification.from_pretrained(name)

            if len(text) > 512:
                text = text[:512]

    

            sentence = f'{text}'

        
            

            

            cut=word_tokenize(sentence.replace(" ", "<_>"))
            inputs=tokenizer(cut,is_split_into_words=True,return_tensors="pt")

            ids = inputs["input_ids"]
            mask = inputs["attention_mask"]
            # forward pass
            outputs = model(ids, attention_mask=mask)
            logits = outputs[0]

            predictions = torch.argmax(logits, dim=2)
            predicted_token_class = [model.config.id2label[t.item()] for t in predictions[0]]

            def fix_span_error(words,ner):
                _ner = []
                _ner=ner
                _new_tag=[]
                for i,j in zip(words,_ner):
                    #print(i,j)
                    i=tokenizer.decode(i)
                    if i.isspace() and j.startswith("B-"):
                        j="O"
                    if i=='' or i=='<s>' or i=='</s>':
                        continue
                    if i=="<_>":
                        i=" "
                    _new_tag.append((i,j))
                return _new_tag

            ner_tag=fix_span_error(inputs['input_ids'][0],predicted_token_class)
            print(ner_tag)

            merged_ner=[]
            for i in ner_tag:
                if i[1].startswith("B-"):
                    merged_ner.append(i)
                elif i[1].startswith("I-"):
                    merged_ner[-1]=(merged_ner[-1][0]+i[0],merged_ner[-1][1])
                else:
                    merged_ner.append(i)

            print(merged_ner)

            #display only entity of person  name
            person = []
            _pharse = []
            for i in merged_ner:
                if i[1].startswith("B-PERSON") and i[0] != ' ' and len(i[0]) > 5 :
                    _pharse.append(i)
                    person.append(i[0])

            print(person)
            print(_pharse)

            if len(person) == 2:
                print('ผู้ส่ง :',person[0]),print('ผู้รับ :',person[1])
                return JsonResponse({'tag': person[0], 'ผู้รับ': person[1], 'text': text}, status=200)

            elif len(person) > 2:
                # print('ผู้ส่ง :',person[0]),print('ผู้รับ :',person[1]+person[2])
                for i in range(2,len(person)):
                    person[1] = person[1] + person[i]
                print('ผู้ส่ง :',person[0]),print('ผู้รับ :',person[1])
                return person[0],person[1]
            else :
                return JsonResponse({'tag': 'ไม่พบข้อมูล', 'ผู้รับ': 'ไม่พบข้อมูล'}, status=200)
            
            
    return HttpResponse(status=200)
