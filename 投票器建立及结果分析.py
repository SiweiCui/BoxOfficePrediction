import pandas as pd
import numpy as np


from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import tree


from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import hinge_loss


rawdata=pd.read_excel('总电影数据.xlsx')

##票房分类
rawdata.iloc[(rawdata.iloc[:,1]<=5000000),1]=4
rawdata.iloc[(rawdata.iloc[:,1]<=10000000)&(rawdata.iloc[:,1]>5000000),1]=3
rawdata.iloc[(rawdata.iloc[:,1]<=100000000)&(rawdata.iloc[:,1]>10000000),1]=2
rawdata.iloc[(rawdata.iloc[:,1]<=300000000)&(rawdata.iloc[:,1]>100000000),1]=1
rawdata.iloc[rawdata.iloc[:,1]>300000000,1]=0


##IP信息

b=pd.get_dummies(rawdata['IP信息'])
rawdata=pd.concat([rawdata,b],axis=1)


##利用四个学习器的结果建立硬投票器
def find_m_often(array):
    l=list(array)
    for i in l:
        if l.count(i)==len(l):
            return i
    count=[l.count(j) for j in l]
    if len(set(count))==len(count):
        return False
    else:
        return l[count.index(max(count))]
    
def voting(m1,m2,m3,m4,E):
    ac_record=[]
    pred=[]
    result=[]
    X_train,X_test,y_train,y_test=E[0],E[1],E[2],E[3]
    for model in (m1,m2,m3,m4):
        model.fit(X_train.values,y_train.values)
        y_pred=model.predict(X_test.values)
        pred.append(list(y_pred))
        ac_record.append(metrics.accuracy_score(y_test, y_pred))
    best_model_index=ac_record.index(max(ac_record))
    pred=np.array(pred)
    # print(pred)
    rows,columns=pred.shape[0],pred.shape[1]
    for i in range(columns):
        if find_m_often(pred[:,i]):
            # print(pred[:,i],find_m_often(pred[:,i]))
            result.append(find_m_often(pred[:,i]))
        else:
            # print(list(pred[:,i]))
            result.append(pred[best_model_index,i])
    return result
        
def voting_k_fold_validation(k,X,y):
    kf=KFold(n_splits=k,shuffle=True,random_state=15)
    a=[]
    for train,test in kf.split(X):
        X_train=X.iloc[train,:]
        y_train=y.iloc[train]
        X_test=X.iloc[test,:]
        y_test=y.iloc[test]
        E=[X_train,X_test,y_train,y_test]
        voting_result=voting(model1,model2,model3,model4,E)
        a.append(metrics.accuracy_score(y_test,voting_result))
    return np.array(a)

def k_fold_validation(k):
    kf=KFold(n_splits=k,shuffle=True,random_state=15)
    L=[]
    for j in (model1,model2,model3,model4):
        a=[]
        for train,test in kf.split(X):
            X_train=X.iloc[train,:]
            y_train=y.iloc[train]
            X_test=X.iloc[test,:]
            y_test=y.iloc[test]
            j.fit(X_train.values,y_train)
            y_pred=j.predict(X_test.values)
            a.append(metrics.accuracy_score(y_test,y_pred))
        L.append(a)  
    return np.array(L)

def voting_k_fold_validation_evaluation(k,X,y):
    kf=KFold(n_splits=k,shuffle=True,random_state=15)
    L=[]
    a=[]
    for train,test in kf.split(X):
        X_train=X.iloc[train,:]
        y_train=y.iloc[train]
        X_test=X.iloc[test,:]
        y_test=y.iloc[test]
        E=[X_train,X_test,y_train,y_test]
        voting_result=voting(model1,model2,model3,model4,E)
        a.append(cohen_kappa_score(y_test,voting_result))
        a.append(hamming_loss(y_test,voting_result))
        # a.append(hinge_loss(y_test,voting_result))
    L.append(a)
        
    return np.array(L)

#模型建立       

model1 = RandomForestClassifier(n_estimators = 15, criterion="entropy")
model2= xgb.XGBClassifier(learning_rate=0.1,n_estimators=15,max_depth=7,
                          use_label_encoder=False,
                          eval_metric=['logloss','auc','error'])

model3= GradientBoostingClassifier(
    learning_rate=0.1,n_estimators=15,max_depth=7,
    max_features='sqrt', random_state=10
)

model4= tree.DecisionTreeClassifier(max_depth=7,criterion="entropy")
X=rawdata.iloc[:,[2,4,5,7,-6]]
X=pd.concat([X,b],axis=1)
y=rawdata.iloc[:,1]

# In[1]
#交叉验证
#10折交叉验证
arr=voting_k_fold_validation(10,X,y)
print('投票器结果')
print(arr.mean())
print(arr.var())   
print(arr.max()) 
print(arr.max()-arr.min())  

# In[2]
#留一法
arr=voting_k_fold_validation(152,X,y) 
print(arr.mean())
print(arr.var())    

# In[3]    
#子分类器
#10折交叉验证
print('子分类器结果')
arr=k_fold_validation(10)
print(arr.mean(axis=1))
print(arr.var(axis=1))
print(arr.max(axis=1))
print(arr.max(axis=1)-arr.min(axis=1))     

# In[4]
#留一法
arr=k_fold_validation(152) 
print(arr.mean(axis=1))
print(arr.var(axis=1))        

# In[5] 准确率随折数的变化
l=[]
for i in range(130,153):
    arr=voting_k_fold_validation(i,X,y)
    l.append(arr.mean())

# In[6] 指标重要性
feature_importances=np.array([model1.feature_importances_,
    model2.feature_importances_,
    model3.feature_importances_,
    model4.feature_importances_])


# In[7] 根据是否在疫情后分类预测
noprodata=pd.read_excel('总原始数据.xlsx')
judge=noprodata['是否在疫情后']
post_name=noprodata['电影名称'][judge==1]
previous_name=noprodata['电影名称'][judge==0]
post_index=[]
previous_index=[]
for i in range(152):
    if rawdata.iloc[i,0] in list(post_name):
        post_index.append(i)
    elif rawdata.iloc[i,0] in list(previous_name):
        previous_index.append(i)
post_data=rawdata.iloc[post_index,:]
previous_data=rawdata.iloc[previous_index,:]
post_result=voting_k_fold_validation(len(post_data.index),
                                      post_data.iloc[:,[2,4,5,7,-6]],
                                      post_data.iloc[:,1]
)
previous_result=voting_k_fold_validation(len(previous_data.index),
                                          previous_data.iloc[:,[2,4,5,7,-6]],
                                          previous_data.iloc[:,1]
)
print(post_result.mean())
print(previous_result.mean())
rawdata=pd.read_excel('总电影数据.xlsx')
post_data=rawdata.iloc[post_index,:]
previous_data=rawdata.iloc[previous_index,:]
print(post_data['总票房'][1:57].mean(),post_data['总票房'].max())
print(previous_data['总票房'][1:93].mean(),previous_data['总票房'].max())

# In[8] kappa 系数
arr=voting_k_fold_validation_evaluation(10,X,y)
print(arr) 

# In[9]
# 决策树可视化
import pybaobabdt
import matplotlib.pyplot as plt
rawdata=pd.read_excel('总电影数据.xlsx')

##票房分类
rawdata.iloc[(rawdata.iloc[:,1]<=5000000),1]=4
rawdata.iloc[(rawdata.iloc[:,1]<=10000000)&(rawdata.iloc[:,1]>5000000),1]=3
rawdata.iloc[(rawdata.iloc[:,1]<=100000000)&(rawdata.iloc[:,1]>10000000),1]=2
rawdata.iloc[(rawdata.iloc[:,1]<=300000000)&(rawdata.iloc[:,1]>100000000),1]=1
rawdata.iloc[rawdata.iloc[:,1]>300000000,1]=0



X=rawdata.iloc[:,[2,4,5,7,-1,-4]]
def processindex(rawdata,columns):
    names=pd.value_counts(rawdata.iloc[:,columns]).index
    value=[i for i in range(len(names))]
    for i in range(len(rawdata.iloc[:,columns])):
        rawdata.iloc[i,columns]=int(value[list(names).index(rawdata.iloc[i,columns])])
processindex(X,-1)
X['IP信息']=list(map(int,X['IP信息']))

# In[]
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签 
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
features=['首日票房','档期','热度指数','是否续集','影片分类','IP信息']
model4.fit(X,y)
ax = pybaobabdt.drawTree(model4, size=10, dpi=72, features=features)

    
    
   
    
   
    

    