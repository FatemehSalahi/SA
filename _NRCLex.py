#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
get_ipython().system('pip install NRCLex')
from nrclex import NRCLex


# In[ ]:


text_df = pd.read_csv("/...path..../test.csv", encoding="utf-8", engine='python')
j=text_df.count(axis=0)[1]
top=fea=ang=anti=tru=sur=pos=neg=sad=dis=joy=0

text_nrclex=[]

for i in range(j):
    fea=ang=ant=tru=sur=pos=neg=sad=dis=joy=0
    emotion=NRCLex(str(text_df['text'].iloc[i]))
    for m in emotion.raw_emotion_scores:
      if m=='fear':
        fea=emotion.raw_emotion_scores['fear']
      elif m=='anger':
        ang=emotion.raw_emotion_scores['anger']
      elif m=='anticipation':
        ant=emotion.raw_emotion_scores['anticipation']
      elif m=='trust':
        tru=emotion.raw_emotion_scores['trust']
      elif m=='surprise':
        sur=emotion.raw_emotion_scores['surprise']
      elif m=='positive':
        pos=emotion.raw_emotion_scores['positive']
      elif m=='negative':
        neg=emotion.raw_emotion_scores['negative']
      elif m=='sadness':
        sad=emotion.raw_emotion_scores['sadness']
      elif m=='disgust':
        dis=emotion.raw_emotion_scores['disgust']
      elif m=='joy':
        joy=emotion.raw_emotion_scores['joy']
    top=emotion.top_emotions
    text_nrclex.append([text_df['tweet'].iloc[i],top,fea,ang,ant,tru,sur,pos,neg,sad,dis,joy])
emotin_df=pd.DataFrame(text_nrclex,columns=['text','top','fear','anger','anticipation','trust','surprise','positive','negative','sadness','disgust','joy'])
emotin_df.to_csv('/...path.../test1.csv',encoding="utf-8")

