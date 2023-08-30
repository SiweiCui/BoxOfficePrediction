import pandas as pd
import numpy as np

# 处理原始数据

## 将中文表示的数字转为浮点
def processrev(numstring):
    if '亿' in numstring:
        num=float(numstring[:-1])*100000000
    else:
        num=float(numstring[:-1])*10000
    return num     

## 将日期转化为对应的档期
def processdate(datestring):
    s=datestring[:10]
    if s[5:7]=='12' and int(s[-2:])>=20:
        return '元旦档期'
    elif s[5:7]=='10' and int(s[-2:])<=7:
        return '国庆档期'
    elif (s[5:7]=='01' or s[5:7]=='02'):
        return '春节档期'
    elif 7<=int(s[5:7])<=9:
        return '暑期档期'
    elif s[5:7]=='05' and int(s[-2:])<=5:
        return '五一档期'
    else:
        return '其他档期'

## 计算用于填充缺失热度指数的平均值
def call_avg(clas):
    s=0
    c=0
    for i in range(len(data['票房所属类别'])):
        if data['票房所属类别'][i]==clas and data['热度指数'][i]!='暂无':
            if data['票房所属类别'][i]!='E':
                s+=float(data['热度指数'][i])
                c+=1
            else:
                return 1000
    return s/c

## 利用平均值填充缺失的热度指数
def processindex(ind):
    clas=data['票房所属类别'][i]
    av=call_avg(clas)
    return int(av)

    
data=pd.read_excel('总原始数据.xlsx')

# In[0] 票房去量纲以及首日票房的缺失值处理
reven=data['总票房']
dreven=data['首日票房']
for i in range(len(reven)):
    try:
        reven.iloc[i]=processrev(reven.iloc[i])
        dreven.iloc[i]=processrev(dreven.iloc[i])
    except:
        dreven.iloc[i]=dreven.iloc[i-1]
        
# In[1] 档期处理
clas=[]
date=data['上映日期']
for i in range(len(date)):
    clas.append(processdate(date.iloc[i]))
data['档期']=clas
c=list(data['档期'])
numb=[]
s1,s2,s3,s4=0,0,0,0,
for j in range(len(c)):
    if c[j]=='春节档期':
        numb.append(1)
        s1+=int(reven[j])/c.count('春节档期')
    elif c[j]=='暑期档期':
        numb.append(1)
        s2+=int(reven[j])/c.count('暑期档期')
    elif c[j]=='国庆档期' or c[j]=='元旦档期' or c[j]=='五一档期':
        numb.append(1)
        s3+=int(reven[j])/(c.count('国庆档期')+c.count('元旦档期')+c.count('五一档期'))
    else:
        numb.append(0)
        s4+=int(reven[j])/c.count('其他档期')
# print('不同档期平均每部电影收入如下(春节，暑期，国庆元旦五一，其他)')
# print(re)
# print('不同档期电影数目如下')
# print(c.count('春节档期'),c.count('暑期档期'),c.count('国庆档期'),c.count('元旦档期'),c.count('五一档期'),c.count('其他档期'))


data['档期']=numb

# In[2]票房类型
def estimate(num):
    if num>500000000:
        return 'A'
    elif 100000000<=num<=500000000:
        return 'B'
    elif 10000000<=num<=100000000:
        return 'C'
    elif 5000000<=num<=10000000:
        return 'D'
    elif num<=5000000:
        return 'E'

data['票房所属类别']=list(map(estimate,data['总票房']))



# In[3] 指数缺失值填充
for i in range(len(data['热度指数'])):
    if data['热度指数'][i]=='暂无' or float(data['热度指数'][i])<=1000:
        data.iloc[i,5]=processindex(i)
    else:
        data.iloc[i,5]=float(data.iloc[i,5])


# In[4] 计算每种类型的价值以及将价值归一化
all_class=set()
def processclas(clas):
    l=clas.split('/')
    for i in range(len(l)):
        l[i]=l[i].replace(' ','')
        all_class.add(l[i])
    return l

data.iloc[:,3]=data['影片分类'].map(processclas)
cla=['喜剧','奇幻','动作','冒险','剧情','科幻','家庭','爱情','悬疑']
count={}
dic={}
for i in range(len(data['影片分类'])):
    for j in data['影片分类'][i]:
        if j in cla:
            if j in dic.keys():
                dic[j]+=data.iloc[i,1]
            if j in count.keys():
                count[j]+=1
            else:
                dic[j]=data.iloc[i,1]
                count[j]=1
Gijmax=max(dic.values())
Gijmin=min(dic.values())

def trans(l):
    l=list(map(lambda x: (np.log(dic[x]/Gijmin))/(np.log(Gijmax/Gijmin)) if x in cla else 0,l
               )
               )
    return sum(l)

data.iloc[:,3]=data['影片分类'].map(trans)
data[data.iloc[:,3]==0]
data.sort_values(by='总票房',ascending=False,inplace=True)
