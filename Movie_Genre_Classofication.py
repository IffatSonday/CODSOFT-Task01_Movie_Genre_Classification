#!/usr/bin/env python
# coding: utf-8

# # Movie Genre Classification

# ## Importing Libraries

# In[1]:


import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Initialize NLTK resources
#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('omw-1.4')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


# ### Data Loading

# In[2]:


train= pd.read_csv("C:/Users/Effat/Desktop/Internships/CODSOFT/Genre Classification Dataset/train_data.txt",sep=':::',names=['title', 'genre', 'description'],engine="python")

test= pd.read_csv("C:/Users/Effat/Desktop/Internships/CODSOFT/Genre Classification Dataset/test_data.txt",sep=':::',names=['title', 'genre', 'description'],engine="python")


# In[3]:


print("Train Dataset")
train.head()


# In[4]:


print("Test Dataset")
test.head()


# ## Data Analysis

# In[5]:


train.info()


# In[6]:


train.describe().T


# **Checking Null values**

# In[7]:


train.isnull().sum()


# In[8]:


train.duplicated().sum()


# **Different Types of Genre**

# In[9]:


categories= train['genre'].unique()
categories


# **Count Of Genre**

# In[10]:


values=train['genre'].value_counts()
values


# **Genre Distribution**

# In[11]:


plt.figure(figsize=(20,15))
sns.countplot(data=train, y="genre", 
              order=train["genre"].value_counts().index, 
              palette="twilight_shifted")

plt.title('Number Of Movies By Gener',fontsize=20)
plt.ylabel('Genre',fontsize=16)
plt.xlabel('Number Of Movies',fontsize=16)
ax = plt.gca()
ax.set_xticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
for p in ax.patches:
    ax.annotate(f'{int(p.get_width())}', 
                (p.get_width(), p.get_y() + p.get_height() / 2), 
                ha='center', va='center', 
                fontsize=12, color='black', 
                xytext=(20, 0), textcoords='offset points')
plt.show()

plt.show()


# **Movies in the 'Drama' genre are the most frequent in the dataset, with a count of 113,613, followed by 'Documentary' with 13,096 and 'Comedy' with 7,447.**
# 
# **Movies in the 'War' genre have the lowest frequency in the dataset, with a count of 132, followed by 'News' with 181 and 'Game-Show' with 194.**

# In[12]:


train['length']=train['description'].apply(len)
train.head()


# **Distribution of Length of Genre Description**

# In[13]:


plt.figure(figsize=(12, 7))

sns.histplot(data=train, x='length', bins=20, kde=True, color='red')

plt.xlabel('Length', fontsize=14, fontweight='bold')
plt.ylabel('Frequency', fontsize=14, fontweight='bold')
plt.title('Distribution of Lengths', fontsize=16, fontweight='bold')

plt.show()


# ## Data Preprocessing 

# In[14]:


def data_processing(text):
    text = text.lower()
    text = re.sub(r'@\S+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = "".join([ch for ch in text if ch not in string.punctuation])
    words = nltk.word_tokenize(text)
    text = " ".join([lemmatizer.lemmatize(word) for word in words if word not in stop_words and len(word) > 2])
    return text.strip()


# In[15]:


train["Cleaned_Text"] = train["description"].apply(data_processing)
test["Cleaned_Text"] = test["description"].apply(data_processing)


# In[16]:


train['Cleaned_Text_len'] = train['Cleaned_Text'].apply(len)


# **Cleaned Dataset**

# In[17]:


train.head()


# In[18]:


train.columns


# In[19]:


plt.figure(figsize=(12, 6))
sns.kdeplot(train['length'], label='Description length',alpha=0.5)
sns.kdeplot(train['Cleaned_Text_len'], label='Description cleaned length',alpha=0.5)
ax = plt.gca()
#ax.set_xticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#ax.spines['bottom'].set_visible(False)
plt.title('Distribution of Text Lengths')
plt.xlabel('Text Length')
plt.ylabel('Frequency')
plt.legend()

           


# **NormalizeText**

# In[20]:


vector = TfidfVectorizer()

x_train = vector.fit_transform(train["Cleaned_Text"])
x_test = vector.transform(test["Cleaned_Text"])


# **Splitting Data**

# In[21]:


x = x_train
y = train["genre"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state=42)


# In[22]:


print("Size of x_train: ", (x_train.shape))
print("Size of y_train: ", (y_train.shape))
print("Size of x_test: ", (x_test.shape))
print("Size of y_test: ", (y_test.shape))


# ## Model Building 

# In[23]:


l_reg=LogisticRegression(solver='saga', max_iter=500)
l_reg.fit(x_train, y_train)


# In[24]:


y_pred = l_reg.predict(x_test)


# ## Model Evaluation

# In[25]:


accuracy = accuracy_score(y_pred,y_test)
print("-----Model Evaluation on Test Data-----")
accuracy


# In[26]:


print(classification_report(y_test, y_pred))


# ## Saving Classified Data 

# In[28]:


print("Length of x_test:", x_test.shape[0])
print("Length of y_pred:", len(y_pred))
print("Length of test:", test.shape[0])


# In[29]:


test_1 = test.iloc[:len(y_pred)].copy()  # Adjust to match the length of predictions


# In[30]:


test_1["Classified_Genre"] = y_pred

# Save the dataframe to a CSV file
test.to_csv("Classified_Genre_dataset.csv", index=False)

print("Predictions saved to 'Classified_Genre_dataset.csv'")
test_1.head()


# ### Thank You
