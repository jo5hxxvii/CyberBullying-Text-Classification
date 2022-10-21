#!/usr/bin/env python
# coding: utf-8

# ## Loading Datasets 

# In[1]:


import pandas as pd
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Kaggle Dynamically generated dataset

# In[2]:


kaggle = pd.read_csv('dynamical kaggle/2020-12-31-DynamicallyGeneratedHateDataset-entries-v0.1.csv')


# In[3]:


kaggle = pd.concat(
    [
        kaggle[kaggle['label'] == 'hate'].sample(5000),
        kaggle[kaggle['label'] == 'nothate'].sample(5000)
    ]
)


# In[4]:


kaggle = kaggle.reset_index().drop('index',axis = 1)


# In[6]:


kaggle = kaggle[['text','label']]


# ### Bayzick Bullying Dataset

# In[9]:


path = "C:\\Users\\josh\\Documents\\work\\Cyber Bullying\\Cyber Bullying\\BayzickBullyingData"
filenames = glob.glob(path + "\\Human Concensus\\*.xlsx")
#print('File names:', filenames)
b_labels = pd.DataFrame()
for file in filenames:
    print(file)
    df = pd.concat(pd.read_excel(file, sheet_name=None, skiprows=2),ignore_index=True, sort=False)
    b_labels = b_labels.append(df, ignore_index=True)


# In[10]:


b_labels.drop(['Unnamed: 2','Unnamed: 3'], inplace = True, axis = 1)
b_labels.columns = ['filename', 'label']


# In[11]:


b_labels.filename = b_labels.filename.astype(str)


# In[12]:


import xml.etree.ElementTree as ET


# In[13]:


def loadXML(p):
    dirs = [x[0] for x in os.walk(p)]
    data = []
    for d in dirs[1:]:
        filenames = glob.glob(d + "\*.xml")
        for file in filenames:
            try:
                #print(file)
                tree = ET.parse(file)
                f = file.split('\\')[-1].replace('.xml','')
                root = tree.getroot()
                texts = []
                for post in root.findall(".//post"):
                    for body in post.findall(".//body"):
                        data.append([f,body.text])
            except:
                continue
    return data


# In[14]:


b_texts = loadXML(path+'\\packets\\')


# In[15]:


b_texts = pd.DataFrame(b_texts)


# In[16]:


b_texts.columns = ['filename', 'text']


# In[17]:


bayzick = pd.merge(b_texts,b_labels, on = 'filename', how= 'left')


# In[18]:


bayzick.dropna(inplace = True)
bayzick.drop('filename', axis =1, inplace = True)


# In[20]:


bayzick = bayzick.reset_index().drop('index',axis = 1)


# ### Kaggle Youtube Parsed Dataset

# In[22]:


youtube = pd.read_csv('various/youtube_parsed_dataset.csv')


# In[25]:


youtube.drop(['index', 'UserIndex', 'Number of Comments',
       'Number of Subscribers', 'Membership Duration', 'Number of Uploads',
       'Profanity in UserID', 'Age' ], axis = 1,  inplace=True)


# In[27]:


youtube.columns = ['text',  'label']


# ### Kaggle Attack Parsed Dataset

# In[28]:


aggression = pd.read_csv('various/aggression_parsed_dataset.csv')


# In[29]:


aggression = pd.concat(
    [
        aggression[aggression['oh_label'] == 0].sample(5000),
        aggression[aggression['oh_label'] == 1].sample(5000)
    ]
)


# In[30]:


aggression = aggression.reset_index().drop('index',axis = 1)


# In[32]:


aggression.drop(['ed_label_0','ed_label_1', 'level_0'], axis = 1,  inplace=True)


# In[34]:


aggression.columns = ['text', 'label']


# #  Preprocessing

# In[35]:


datasets = {'kaggle':kaggle, 'bayzick':bayzick, 'aggression':aggression, 'youtube':youtube}


# In[36]:


import nltk
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


# In[37]:


nltk.download('stopwords')


# In[38]:


REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    """
        text: a string
        
        return: modified initial string
    """
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # delete stopwors from text
    return text


# In[39]:


#Clean Datasets
for i, (n,df) in enumerate(datasets.items()):
    df['text'] = df['text'].apply(clean_text)


# In[40]:


preprocess = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ])


# # Training and Classification

# ### Model Definitions

# In[41]:


from sklearn.naive_bayes import GaussianNB as NB
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import LogisticRegression as LR 
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier as NN
from sklearn.ensemble import GradientBoostingClassifier as GB
from sklearn.ensemble import AdaBoostClassifier as AB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.ensemble import VotingClassifier as VC
import disarray as da


# In[ ]:


voting_estimators = [
    ('Naive Bayes', NB()),
    ('K Nearest Neigbhor', KNN()),
    ('Logistic Regression', LR()),
    ('Decision Tree', DT()),
    ('SVM', SVC()),
]


# In[42]:


models = {
    'Naive Bayes': NB(),
    'Random Forest': RFC(),
    'K Nearest Neigbhor': KNN(),
    'Logistic Regression': LR(),
    'Decision Tree': DT(),
    'SVM': SVC(),
    'Neural Network': NN(),
    'Gradient Boosting': GB(),
    'Ada Boost': AB(),
    'Quadratic Discriminant Analysis': QDA(),
    'Max Voting': VC(estimators=voting_estimators, voting='hard')
}


# In[43]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, cohen_kappa_score, matthews_corrcoef


# In[44]:


def train_classify(df, model):
    x = df.drop([0,1],axis = 1)
    y = df.iloc[:, 1]
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.30, random_state = 42)
    model.fit(xtrain,ytrain)
    ypred = model.predict(xtest)
    accuracy = accuracy_score(ytest, ypred)
    cls_report = classification_report(ytest,ypred)
    cm = confusion_matrix(ytest,ypred)

    
    return ytest, ypred, accuracy, cls_report, model, pd.DataFrame(cm, dtype=int)


# In[45]:


def run_models(df, models):
    processed_text = preprocess.fit_transform(df.text).toarray()
    df = pd.concat([df, pd.DataFrame(processed_text)],axis = 1, ignore_index = True)
    results = {}
    for i, (mn,model) in enumerate(models.items()):
        print('Running '+mn)
        ytest, ypred, accuracy, report, mod, cm = train_classify(df,model)
        results[mn] = [ytest, ypred, accuracy, report, [mn,mod], cm]
    return results


# In[46]:


def run_experiments(datasets, models):
    experiment_results = {}
    for i, (n,df) in enumerate(datasets.items()):
        print('Running for: '+n)
        results = run_models(df,models)
        #experiment_results[n] = results
    #return experiment_results
    return results


# In[47]:


k = {'kaggle':datasets['kaggle']}
b = {'bayzick':datasets['bayzick']}
a = {'aggression':datasets['aggression']}
y = {'youtube':datasets['youtube']}


# In[48]:


#run models on kaggle dataset
kag = run_experiments(k, models)


# In[131]:


#run models on bayzick dataset
bay = run_experiments(b, models)


# In[50]:


#run models on aggression dataset
agg = run_experiments(a, models)


# In[51]:


#run models on youtube dataset
you = run_experiments(y, models)


# In[139]:


#convert labels to numerical for result collation (AUC Function can only use numerical classes)
def convertLabels(l, val, rep):
    for i in l.keys():
        l[i][0].replace(val,rep,inplace=True)
        yp = l[i][1]
        yp[yp == val] = rep
        l[i][1] = yp
    return l


# In[111]:


kag = convertLabels(kag, 'nothate', int(0)) 
kag = convertLabels(kag, 'hate', int(1))
bay = convertLabels(bay, 'N', int(0)) 
bay = convertLabels(bay, 'Y', int(1))


# # Results and Comparison

# In[118]:


def collateResults(ytest,ypred, cm, model):
    mcc = matthews_corrcoef(ytest, ypred)
    kappa = cohen_kappa_score(ytest, ypred)
    auc = roc_auc_score(ytest,ypred)
    metrics = cm.da.export_metrics(metrics_to_include=[
        'accuracy',
        'f1',
        'false_discovery_rate',
        'false_negative_rate',
        'false_positive_rate',
        'negative_predictive_value',
        'precision',
        'recall',
        'specificity',
    ])[['micro-average']]
    metrics.columns = model
    metrics = metrics.transpose().reset_index()
    metrics['MCC'] = [mcc]
    metrics['KAPPA'] = [kappa]
    metrics['AUC'] = [auc]
    
    return metrics


# In[159]:


def collate(results):
    res = {}
    for i, (data,result) in enumerate(results.items()):
        r = pd.DataFrame(columns=['index', 'accuracy', 'f1', 'false_discovery_rate',
       'false_negative_rate', 'false_positive_rate',
       'negative_predictive_value', 'precision', 'recall', 'specificity',
       'MCC', 'KAPPA', 'AUC'])
        for i, (model,performance) in enumerate(result.items()):
            temp = collateResults(performance[0], performance[1].astype(float), performance[5], [model])
            r = pd.concat([r,temp], ignore_index=True)
        res[data] = r
    return res


# In[160]:


final_results = collate({'kaggle':kag,'bayzick':bay,'aggression':agg,'youtube':you})


# ### Export Results

# In[191]:


for i in final_results.keys():
    final_results[i].to_excel(i+'.xlsx', index=False)


# In[ ]:




