#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
from scipy import stats
from scipy.stats import norm,skew
import ast
from sklearn.preprocessing import MultiLabelBinarizer,LabelEncoder
from scipy.special import boxcox1p,inv_boxcox1p,boxcox
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import lightgbm as lgb
import catboost as cat
from sklearn.base import BaseEstimator,TransformerMixin,RegressorMixin,clone
from sklearn.metrics import mean_squared_log_error


# In[2]:


#读入数据
train=pd.read_csv("C:/Users/jynkris/Desktop/train.csv")
test=pd.read_csv("C:/Users/jynkris/Desktop/test.csv")


# In[3]:


#训练集和测试集合并
test.index=test.index+3000
df=pd.concat([train,test]).drop("id", axis=1)
df1=pd.concat([train.drop("revenue",axis=1),test]).drop("id", axis=1)
train1=train
y_train=train["revenue"]


# In[4]:


#修正json格式的数据
fixones=["belongs_to_collection", "genres", "production_companies", "production_countries",                 "Keywords"]

for feature in fixones:
    df.loc[df[feature].notnull(),feature]=    df.loc[df[feature].notnull(),feature].apply(lambda x : ast.literal_eval(x))    .apply(lambda x : [y["name"] for y in x])


# In[5]:


#增添新的数据
release_date=pd.to_datetime(df["release_date"])
df["release_year"]=release_date.dt.year
df["release_month"]=release_date.dt.month
df["release_day"]=release_date.dt.day
df["release_weekday"]=release_date.dt.dayofweek
df["release_quarter"]=release_date.dt.quarter


# In[6]:


#修正json格式的数据
df.loc[df["cast"].notnull(),"cast"]=df.loc[df["cast"].notnull(),"cast"].apply(lambda x : ast.literal_eval(x))
df.loc[df["crew"].notnull(),"crew"]=df.loc[df["crew"].notnull(),"crew"].apply(lambda x : ast.literal_eval(x))


# In[7]:


#只提取cast和crew中的前五个人
df.loc[df["cast"].notnull(),"cast"]=df.loc[df["cast"].notnull(),"cast"].apply(lambda x : [y["name"] for y in x if y["order"]<5]) 


# In[8]:


#增添新的数据
df["Director"]=[[] for i in range(df.shape[0])]
df["Producer"]=[[] for i in range(df.shape[0])]
df["Executive Producer"]=[[] for i in range(df.shape[0])]
df["Director"]=df.loc[df["crew"].notnull(),"crew"].apply(lambda x : [y["name"] for y in x if y["job"]=="Director"])
df["Producer"]=df.loc[df["crew"].notnull(),"crew"].apply(lambda x : [y["name"] for y in x if y["job"]=="Producer"])
df["Executive Producer"]=df.loc[df["crew"].notnull(),"crew"].apply(lambda x : [y["name"] for y in x if y["job"]=="Executive Producer"])


# In[9]:


#将空数据记为零
emptyones=["belongs_to_collection","Keywords","production_companies","production_countries","Director","Producer","Executive Producer","cast","genres"]
for feature in emptyones:
    df[feature] = df[feature].apply(lambda d: d if isinstance(d, list) else [])


# In[10]:


#将空数据记为零
zerones=["runtime","release_month","release_year","release_weekday","release_quarter","release_day"]
for feat in zerones:
    df[feat]=df[feat].fillna(0)


# In[11]:


#去掉用不到的数据
df=df.drop(["imdb_id","original_title","overview","poster_path","tagline","status","title","spoken_languages","release_date","crew"],axis=1)


# In[12]:


#热编码
lbl=LabelEncoder()
lbl.fit(df["release_year"].values)
df["release_year"]=lbl.transform(df["release_year"].values)
lbl.fit(df["original_language"].values)
df["original_language"]=lbl.transform(df["original_language"].values)
lbl.fit(df["Director"].values)
df["Director"]=lbl.transform(df["Director"].values)
lbl.fit(df["Producer"].values)
df["Producer"]=lbl.transform(df["Producer"].values)
lbl.fit(df["Executive Producer"].values)
df["Executive Producer"]=lbl.transform(df["Executive Producer"].values)
lbl.fit(df["genres"].values)
df["genres"]=lbl.transform(df["genres"].values)
lbl.fit(df["production_companies"].values)
df["production_companies"]=lbl.transform(df["production_companies"].values)
lbl.fit(df["production_countries"].values)
df["production_countries"]=lbl.transform(df["production_countries"].values)
lbl.fit(df["Keywords"].values)
df["Keywords"]=lbl.transform(df["Keywords"].values)
lbl.fit(df["belongs_to_collection"].values)
df["belongs_to_collection"]=lbl.transform(df["belongs_to_collection"].values)


# In[13]:


a=df.corr()[u'revenue']
a=a.drop(["revenue"],axis=0)


# In[14]:


name_list = ['Keywords','belongs_to_collection','budget','genres','original_language','popularity','production_companies','production_countries','runtime','release_year','release_month','release_day','release_weekday','release_quarter','Director','Producer','Executive Producer']  
plt.barh(range(len(a)), a,color='#87CEFA',tick_label=name_list)
plt.xlabel('Influence')
plt.ylabel('Factor')
plt.title("Influence of some factors")
plt.show()  


# In[15]:


dict_columns = ['belongs_to_collection', 'genres', 'production_companies','production_countries', 'spoken_languages', 'Keywords', 'cast', 'crew']

def text_to_dict(df):
    for column in dict_columns:
        df[column] = df[column].apply(lambda x: {} if pd.isna(x) else ast.literal_eval(x) )
    return df
        
train = text_to_dict(train)


# In[16]:


list_of_genres = list(train['genres'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)
list_of_Keywords = list(train['Keywords'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)
list_of_production_companies = list(train['production_companies'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)
list_of_production_countries = list(train['production_countries'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)


# In[17]:


a=Counter([i for j in list_of_genres for i in j]).most_common()
b=[]
c=[]
for i in a:
    b.append(i[0])
    c.append(i[1])


# In[18]:


plt.barh(range(len(c)), c,color='#BF3EFF',tick_label=b)
plt.xlabel('Number')
plt.ylabel('Genres')
plt.show() 
d=[]
e=[]
for i in range(len(c)):
    if c[i]>300:
        d.append(c[i])
        e.append(b[i])
labels = e
fracs = d
plt.axes(aspect=1)  # set this , Figure is round, otherwise it is an ellipse
#autopct ，show percet
explode = [0.1,0,0,0,0,0,0,0]
plt.pie(x=fracs, labels=labels,autopct='%3.1f %%',
        shadow=True, labeldistance=1.1, explode=explode,startangle = 90,pctdistance = 0.6)
plt.title("The number of genres of movies")
plt.show()
 


# In[19]:


a=Counter([i for j in list_of_Keywords for i in j]).most_common()
b=[]
c=[]
for i in a:
    b.append(i[0])
    c.append(i[1])


# In[20]:


d=[]
e=[]
for i in range(10):
        d.append(c[i])
        e.append(b[i])
plt.barh(range(len(d)), d,color='#BF3EFF',tick_label=e)
plt.xlabel('Number')
plt.ylabel('Keywords')
plt.show() 
labels = e
fracs = d
plt.axes(aspect=1)# set this , Figure is round, otherwise it is an ellipse
#autopct ，show percet
explode = [0.1,0,0,0,0,0,0,0,0,0]
plt.pie(x=fracs, labels=labels,autopct='%3.1f %%',
        shadow=True, labeldistance=1.1, explode=explode,startangle = 90,pctdistance = 0.6)
plt.title("The number of the keywords of movies")
plt.show()


# In[21]:


from collections import Counter
a=Counter([i for j in list_of_production_companies for i in j]).most_common()
b=[]
c=[]
for i in a:
    b.append(i[0])
    c.append(i[1])


# In[22]:


d=[]
e=[]
for i in range(10):
        d.append(c[i])
        e.append(b[i])
plt.barh(range(len(d)), d,color='#BF3EFF',tick_label=e)
plt.xlabel('Number')
plt.ylabel('Production_companies')
plt.show() 
labels = e
fracs = d
plt.axes(aspect=1)# set this , Figure is round, otherwise it is an ellipse
#autopct ，show percet
explode = [0.1,0,0,0,0,0,0,0,0,0]
plt.pie(x=fracs, labels=labels,autopct='%3.1f %%',
        shadow=True, labeldistance=1.1, explode=explode,startangle = 90,pctdistance = 0.6)
plt.title("The number of  production countries")
plt.show()


# In[23]:


a=Counter([i for j in list_of_production_countries for i in j]).most_common()
b=[]
c=[]
for i in a:
    b.append(i[0])
    c.append(i[1])


# In[24]:


d=[]
e=[]
for i in range(5):
        d.append(c[i])
        e.append(b[i])
plt.barh(range(len(d)), d,color='#BF3EFF',tick_label=e)
plt.xlabel('Number')
plt.ylabel('Production_companies')
plt.show() 
labels = e
fracs = d
plt.axes(aspect=1)# set this , Figure is round, otherwise it is an ellipse
#autopct ，show percet
explode = [0.1,0,0,0,0]
plt.pie(x=fracs, labels=labels,autopct='%3.1f %%',
        shadow=True, labeldistance=1.1, explode=explode,startangle = 90,pctdistance = 0.6)
plt.title("The number of production companies")
plt.show()


# In[25]:


train['num_genres'] = train['genres'].apply(lambda x: len(x) if x != {} else 0)
plt.scatter(train['num_genres'], train['revenue'])
plt.xlabel('num_genres')
plt.ylabel('revenue')
plt.title('Revenue for different number of genres in the film');


# In[26]:


train['num_Keywords'] = train['Keywords'].apply(lambda x: len(x) if x != {} else 0)
plt.xlim([0,40])
plt.scatter(train['num_Keywords'], train['revenue'])
plt.xlabel('num_Keywords')
plt.ylabel('revenue')
plt.title('Revenue for different number of keywords in the film');


# In[27]:


train['num_production_companies'] = train['production_companies'].apply(lambda x: len(x) if x != {} else 0)
plt.scatter(train['num_production_companies'], train['revenue'])
plt.xlabel('num_production_companies')
plt.ylabel('revenue')
plt.title('Revenue for different number of production companies in the film');


# In[28]:


train['num_production_countries'] = train['production_countries'].apply(lambda x: len(x) if x != {} else 0)
plt.scatter(train['num_production_countries'], train['revenue'])
plt.xlabel('num_production_countries')
plt.ylabel('revenue')
plt.title('Revenue for different number of production countries in the film');


# In[29]:


plt.hist(df['runtime'].fillna(0),bins=40);
plt.xlim([50,200])
plt.xlabel('runtime')
plt.ylabel('Influence')
plt.title('Influence of length of film in hours');


# In[30]:


plt.scatter(df['budget'], df['revenue'])
plt.xlabel('budget')
plt.ylabel('revenue')
plt.title('Influence of budget');


# In[31]:


plt.scatter(train['popularity'], train['revenue'])
plt.xlabel('popularity')
plt.ylabel('revenue')
plt.xlim([0,50])
plt.title('Influence of popularity');


# In[32]:


plt.scatter(df['belongs_to_collection'], df['revenue'])
plt.xlabel('belongs_to_collection')
plt.ylabel('revenue')
plt.title('Influence of belongs_to_collection');


# In[33]:


plt.scatter(df['release_weekday'], df['revenue'])
plt.xlabel('release_weekday')
plt.ylabel('revenue')
plt.title('Influence of release_weekday');


# In[34]:


plt.scatter(df['Executive Producer'], df['revenue'])
plt.xlabel('Executive Producer')
plt.ylabel('revenue')
plt.title('Influence of Executive Producer');


# In[35]:


release_date=pd.to_datetime(train["release_date"])
train["release_year"]=release_date.dt.year
import collections     
ctr = collections.Counter(train["release_year"])  
a=dict(ctr)
b=[]
for k in sorted(a):
    b.append(a[k])
a=sorted(a)
plt.xlim([1970,2019])
plt.plot(a, b)
plt.xlabel('release_year')
plt.ylabel('number')
plt.title('Influence of release year');
plt.show()


# In[13]:


#重新处理df1 用boxcox检验
y_train=boxcox1p(y_train,0.2)
features_to_fix=["belongs_to_collection", "genres", "production_companies", "production_countries",                 "Keywords"]

for feature in features_to_fix:
    df1.loc[df1[feature].notnull(),feature]=    df1.loc[df1[feature].notnull(),feature].apply(lambda x : ast.literal_eval(x))    .apply(lambda x : [y["name"] for y in x])
df1.loc[df1["cast"].notnull(),"cast"]=df1.loc[df1["cast"].notnull(),"cast"].apply(lambda x : ast.literal_eval(x))
df1.loc[df1["crew"].notnull(),"crew"]=df1.loc[df1["crew"].notnull(),"crew"].apply(lambda x : ast.literal_eval(x))
df1["cast_len"] = df1.loc[df1["cast"].notnull(),"cast"].apply(lambda x : len(x))
df1["crew_len"] = df1.loc[df1["crew"].notnull(),"crew"].apply(lambda x : len(x))

df1["production_companies_len"]=df1.loc[df1["production_companies"].notnull(),"production_companies"].apply(lambda x : len(x))

df1["production_countries_len"]=df1.loc[df1["production_countries"].notnull(),"production_countries"].apply(lambda x : len(x))

df1["Keywords_len"]=df1.loc[df1["Keywords"].notnull(),"Keywords"].apply(lambda x : len(x))
df1["genres_len"]=df1.loc[df1["genres"].notnull(),"genres"].apply(lambda x : len(x))

release_date=pd.to_datetime(df1["release_date"])
df1["release_year"]=release_date.dt.year
df1["release_month"]=release_date.dt.month
df1["release_day"]=release_date.dt.day
df1["release_wd"]=release_date.dt.dayofweek
df1["release_quarter"]=release_date.dt.quarter

df1.loc[df1["cast"].notnull(),"cast"]=df1.loc[df1["cast"].notnull(),"cast"].apply(lambda x : [y["name"] for y in x if y["order"]<6])

df1=df1.drop(["imdb_id","original_title","overview","poster_path","tagline","status","title",           "spoken_languages","release_date","crew"],axis=1)
mis_val=((df1.isnull().sum()/df1.shape[0])*100).sort_values(ascending=False)
mis_val=mis_val.drop(mis_val[mis_val==0].index)
to_empty_list=["belongs_to_collection","Keywords","production_companies","production_countries",             "cast","genres"]

for feature in to_empty_list:
    df1[feature] = df1[feature].apply(lambda d: d if isinstance(d, list) else [])
to_zero=["runtime","release_month","release_year","release_wd","release_quarter","release_day","Keywords_len","production_companies_len","production_countries_len","crew_len","cast_len","genres_len"]

for feat in to_zero:
    df1[feat]=df1[feat].fillna(0)
df1['_budget_popularity_ratio'] = df1['budget']/df1['popularity']
df1['_releaseYear_popularity_ratio'] = df1['release_year']/df1['popularity']
df1['_releaseYear_popularity_ratio2'] = df1['popularity']/df1['release_year']
mis_val=((df.isnull().sum()/df.shape[0])*100).sort_values(ascending=False)
mis_val=mis_val.drop(mis_val[mis_val==0].index)
numeric=[feat for feat in df1.columns if df1[feat].dtype!="object"]
#使用boxcox检测
skewness=df1[numeric].apply(lambda x : skew(x)).sort_values(ascending=False)
skew=skewness[skewness>2.5]
high_skew=skew[skew>10].index
medium_skew=skew[skew<=10].index

for feat in high_skew:
    df1[feat]=np.log1p(df1[feat])

for feat in medium_skew:
    df1[feat]=df1[feat]=boxcox1p(df1[feat],0.15)
skew=df1[skew.index].skew()
lbl=LabelEncoder()
lbl.fit(df1["release_year"].values)
df1["release_year"]=lbl.transform(df1["release_year"].values)


# In[14]:


#设置限制
limits=[4,0,0,35,10,40,10] 
to_dummy = ["belongs_to_collection","genres","original_language","production_companies","production_countries",           "Keywords","cast"]#,"Director","Producer","Executive Producer"
for i,feat in enumerate(to_dummy):
    mlb = MultiLabelBinarizer()
    s=df1[feat]
    x=pd.DataFrame(mlb.fit_transform(s),columns=mlb.classes_, index=df1.index)
    y=pd.DataFrame(mlb.fit_transform(s),columns=mlb.classes_, index=df1.index).sum().sort_values(ascending=False)
    rare_entries=y[y<=limits[i]].index
    x=x.drop(rare_entries,axis=1)
    df1=df1.drop(feat,axis=1)
    df1=pd.concat([df1, x], axis=1, sort=False)


# In[15]:


#将训练集与测试集分开
ntrain=train1.shape[0]

train1=df1.iloc[:ntrain,:]
test=df1.iloc[ntrain:,:]
print("The shape of train DataFrame is {} and the shape of the test DataFrame is {}".format(train1.shape,test.shape))


# In[7]:


#建立模型
model_xgb=xgb.XGBRegressor(max_depth=5,
                           learning_rate=0.1, 
                           n_estimators=2000, 
                           objective='reg:linear', 
                           gamma=1.45, 
                           verbosity=3,
                           subsample=0.7, 
                           colsample_bytree=0.8, 
                           colsample_bylevel=0.50)


# In[38]:


model_lgb=lgb.LGBMRegressor(n_estimators=10000, 
                             objective="regression", 
                             metric="rmse", 
                             num_leaves=20, 
                             min_child_samples=100,
                             learning_rate=0.01, 
                             bagging_fraction=0.8, 
                             feature_fraction=0.8, 
                             bagging_frequency=1, 
                             subsample=.9, 
                             colsample_bytree=.9,
                             use_best_model=True)


# In[39]:


model_cat = cat.CatBoostRegressor(iterations=10000,learning_rate=0.01,depth=5,eval_metric='RMSE',                              colsample_bylevel=0.7,
                              bagging_temperature = 0.2,
                              metric_period = None,
                              early_stopping_rounds=200)


# In[40]:


n_folds=5

def cross_val(model):
    cr_val=np.sqrt(-cross_val_score(model,train1.values,y_train.values,scoring="neg_mean_squared_log_error",cv=5))
    return cr_val


# In[41]:


def msle(y,y_pred):
    return np.sqrt(mean_squared_log_error(y,y_pred))


# In[42]:


ti=time.time()
model_lgb.fit(train1.values,y_train)
print("Number of minutes of training of model_lgb = {:.2f}".format((time.time()-ti)/60))

lgb_pred_train=model_lgb.predict(train1.values)
print("Mean square logarithmic error of lgb model on whole train = {:.4f}".format(msle(y_train,lgb_pred_train)))


# In[43]:


#这里耗时有点长
ti=time.time()
model_xgb.fit(train1.values,y_train)
print("Number of minutes of training of model_xgb = {:.2f}".format((time.time()-ti)/60))

xgb_pred_train=model_xgb.predict(train1.values)
print("Mean square logarithmic error of xgb model on whole train = {:.4f}".format(msle(y_train,xgb_pred_train)))


# In[44]:


#这里耗时有点长
ti=time.time()
model_cat.fit(train1.values,y_train,verbose=False)
print("Number of minutes of training of model_cal = {:.2f}".format((time.time()-ti)/60))

cat_pred_train=model_cat.predict(train1.values)
cat_pred_train[cat_pred_train<0]=0
print("Mean square logarithmic error of cat model on whole train = {:.4f}".format(msle(y_train,cat_pred_train)))


# In[45]:


c = np.array([0.333334,0.333333,0.333333])

print("The sum of the entries of c is {}".format(c.sum()))

train_pred=xgb_pred_train*c[0]+lgb_pred_train*c[1]+cat_pred_train*c[2]
print("Mean square logarithmic error of chosen model on whole train = {:.4f}".format(msle(y_train,train_pred)))


# In[47]:


lgb_pred=model_lgb.predict(test)
xgb_pred=model_xgb.predict(test.values)
cat_pred=model_cat.predict(test)


# In[48]:


#将结果写入表格
pred=inv_boxcox1p((xgb_pred*c[0]+lgb_pred*c[1]+cat_pred*c[2]),0.2)

sub=pd.DataFrame({"id":np.arange(test.shape[0])+3001,"revenue":pred})
sub.to_csv("C:/Users/jynkris/Desktop/sample_submission.csv",index=False)

