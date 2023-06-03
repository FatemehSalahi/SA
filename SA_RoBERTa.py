#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install transformers')
from transformers import AutoModelForSequenceClassification , BertForSequenceClassification
from transformers import (XLMRobertaConfig, XLMRobertaTokenizer, TFXLMRobertaModel)            
from transformers import AutoTokenizer, AutoConfig, TFAutoModel    
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd



# In[ ]:


text_df = pd.read_csv("/...path.../test.csv", encoding="utf-8", engine='python')
j=text_df.count(axis=0)[1]
roberta = "cardiffnlp/twitter-roberta-base-sentiment"
model = AutoModelForSequenceClassification.from_pretrained(roberta)
tokenizer = AutoTokenizer.from_pretrained(roberta)
labels = ['Negative', 'Neutral', 'Positive']
tweeter_emotion=[]
for i in range(j): 
    encoded_tweet = tokenizer(str(text_df['tweet'].iloc[i]),padding=True ,truncation=True ,return_tensors='pt')
    output = model(**encoded_tweet)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    emotion=NRCLex(str(text_df['tweet'].iloc[i]))
    tweeter_emotion.append([text_df['tweet'].iloc[i],scores[0],scores[1],scores[2],text_df['Location'].iloc[i],text_df['Date'].iloc[i],text_df['keyword'].iloc[i]])
emotin_df=pd.DataFrame(tweeter_emotion,columns=['tweet','negative','neutral','positive','location','date','keyword'])
emotin_df.to_csv('/...path.../test12.csv',encoding="utf-8")

