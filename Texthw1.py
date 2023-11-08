#!/usr/bin/env python
# coding: utf-8

# Q1 Review the source code of the webpage using the inspector tool of your browser. 
# List three types of HTML tags you found on the page and explain what elements of the page are created with those tags.
# I found the following tags head, meta, title. Title contains the title of the page. Meta contains information about the text, like the font type. Head formats the area as a header.

# In[1]:


import requests
from bs4 import BeautifulSoup
import pandas as pd
import sklearn


# Q2 Issue a request for the webpage using the requests library and display the HTTP response code for the request. 
# What is the response status code and what does it indicate? What do status codes in the 400s mean? What do status codes in the 500s mean?
# 
# Response status 200 = request completed okay
# Response status 400s = client error
# Response status 500s = server error

# In[2]:


requests.get("https://en.wikipedia.org/wiki/Georgetown_University")


# Q3 Extract all the content with p tag and save as a list
# Extract all raw text contained in paragraph tags on the web page. Store the results in single cell in a pandas DataFrame.

# In[3]:


# 1 Pass URL
html = requests.get("https://en.wikipedia.org/wiki/Georgetown_University").text

# 2 Make it soup object
soup_msba = BeautifulSoup(html, 'lxml')

msba_p = [each.text for each in soup_msba.find_all("p")]

# Convert list of content to dataframe
df = pd.DataFrame(msba_p,
                 columns=["text"])
df = pd.DataFrame({"text":" ".join(df["text"])},
                 index=[0])
df


# Q4 Load and instantiate a spaCy NLP pipeline. Apply the pipeline to the scraped text to extract the named entities via named entity recognition (NER). Store the results in a new dataframe called ner_df with two columns: the extracted entities/text and its corresponding entity label.

# In[4]:


#pip install -U spacy


# In[5]:


import spacy
#import en_core_web_sm


# In[6]:


nlpspacy = spacy.load("en_core_web_sm")


# In[7]:


doc = nlpspacy(df["text"][0])

#for each in doc.ents:
 #   print(each.text, each.label_)
    
ner_df = pd.DataFrame({
    "text":[each.text for each in doc.ents],
    "label":[each.label_ for each in doc.ents]
})


# Q5 Print in descending order the top 10 most frequently mentioned people and the number of times each is mentioned. Print in descending order the top 10 most frequently mentioned organizations and the number of times each is mentioned. 
# 
# Most mentioned person- Jesus
# Org- Georgetown
# 

# In[38]:


ner_df


# In[43]:


ner_df_orgs= ner_df.query('label =="ORG"')


# In[45]:


orgs = ner_df_orgs['text'].value_counts(ascending=False)
orgs.nlargest(10)


# In[46]:


ner_df_people= ner_df.query('label =="PERSON"')
people = ner_df_people['text'].value_counts(ascending=False)
people.nlargest(10)


# Q6 Do #3, #4, and #5 again but instead extract all text contained within anchor (<a>) tags on the web page. Do the results differ from the results in #5?
#     
#     Yes! The results do change from #5. 
#     Top "person" Articles  followed by John Carroll 
#     Top org Georgetown University
# 

# In[49]:


msba_p = [each.text for each in soup_msba.find_all("a")]

# Convert list of content to dataframe
df = pd.DataFrame(msba_p,
                 columns=["text"])
df = pd.DataFrame({"text":" ".join(df["text"])},
                 index=[0])
df
doc = nlpspacy(df["text"][0])

#for each in doc.ents:
 #   print(each.text, each.label_)
    
ner_df = pd.DataFrame({
    "text":[each.text for each in doc.ents],
    "label":[each.label_ for each in doc.ents]
})
ner_df_orgs= ner_df.query('label =="ORG"')
orgs = ner_df_orgs['text'].value_counts(ascending=False)
orgs.nlargest(10)


# In[50]:


ner_df_people= ner_df.query('label =="PERSON"')
people = ner_df_people['text'].value_counts(ascending=False)
people.nlargest(10)


# Q7 Briefly explain web scraping and named entity recognition/extraction in a way that a non-technical co-worker would understand. 
# 
# Webscraping is a process that reads the base code for a website and converts it into a usable file. Entity recognition takes web scraping a step further and attempts to classify items into categories like names or locations.
