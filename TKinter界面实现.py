import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
import xgboost
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import tree


from sklearn import metrics
from sklearn.model_selection import KFold

rawdata=pd.read_excel('数据/总电影数据.xlsx')

##票房分类
rawdata.iloc[(rawdata.iloc[:,1]<=5000000),1]=4
rawdata.iloc[(rawdata.iloc[:,1]<=10000000)&(rawdata.iloc[:,1]>5000000),1]=3
rawdata.iloc[(rawdata.iloc[:,1]<=100000000)&(rawdata.iloc[:,1]>10000000),1]=2
rawdata.iloc[(rawdata.iloc[:,1]<=300000000)&(rawdata.iloc[:,1]>100000000),1]=1
rawdata.iloc[rawdata.iloc[:,1]>300000000,1]=0


##IP信息

b=pd.get_dummies(rawdata['IP信息'])
rawdata=pd.concat([rawdata,b],axis=1)


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
    
def voting(modellist,sample):
    #result的结果是按m1,m2,m3,m4排序的
    rank=[model2,model3,model1,model4]
    result=[model.predict(sample) for model in modellist]
    modellist.sort(key=lambda x:rank.index(x))
    # print(modellist)
    pred=[]
    for model in modellist:
        pred.append(model.predict(sample))

    if find_m_often(pred):
        result.append(find_m_often(pred))
    else:
        result.append(pred[0])
    
    return result
        

#模型建立       

model1 = RandomForestClassifier(n_estimators = 15, criterion="entropy")
model2= xgboost.XGBClassifier(learning_rate=0.1,n_estimators=15,max_depth=7,
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
for i in (model1,model2,model3,model4):
    i.fit(X,y)

# In[1] GUI模块
from tkinter import *
import random
def step1():
    def model_preparation():
        check_buttons=[rVar1,rVar2,rVar3,rVar4]
        check_situation=np.array(list(map(lambda x:x.get(),check_buttons)))
        model_name=np.array(['随机森林','XGBoost','GDBT','决策树'])
        text='已选择：'
        model_selected='、'.join(model_name[check_situation==1])
        l2.config(text=text+model_selected)
    def yes():
        check_buttons=[rVar1,rVar2,rVar3,rVar4]
        check_situation=np.array(list(map(lambda x:x.get(),check_buttons)))
        modellist=[model1,model2,model3,model4]
        final_models=[modellist[i] for i in range(len(check_situation)) if check_situation[i]==1]
        window1.destroy()
        step2(check_situation,final_models)       
        
        
        
    window1=Tk()
    window1.title('动漫电影票房预测')
    h=window1.winfo_screenmmheight()
    w=window1.winfo_screenmmwidth()
    window1.geometry('500x500+%d+%d'%(w,h))
    l1=Label(window1,text='请选择用于参加投票的子分类器：')
    
    rVar1=IntVar()
    rVar2=IntVar()
    rVar3=IntVar()
    rVar4=IntVar()
    chk1=Checkbutton(window1,text='随机森林',variable=rVar1,onvalue=1,offvalue=0,command=model_preparation)
    chk2=Checkbutton(window1,text='XGBoost',variable=rVar2,onvalue=1,offvalue=0,command=model_preparation)
    chk3=Checkbutton(window1,text='GDBT',variable=rVar3,onvalue=1,offvalue=0,command=model_preparation)
    chk4=Checkbutton(window1,text='决策树',variable=rVar4,onvalue=1,offvalue=0,command=model_preparation)
    
    
    
    l2=Label(window1,text='未选择')
    
    button=Button(window1,text='确定',command=yes)
    
    l1.grid()
    chk1.grid()
    chk2.grid()
    chk3.grid()
    chk4.grid()
    l2.grid(column=1,row=2)
    button.grid()
    window1.mainloop()
    
def step2(check_situation,final_models):
    def predict():
        name=e1.get()
        freven=int(e2.get())
        dangqi=(0 if rVar1.get()==0 else 1)
        baiduindex=int(e4.get())
        xuji=(0 if rVar2.get()==0 else 1)
        typ=e6.get().split(' ')
        dic={'喜剧': 11265475000.0, '奇幻': 9631600000.0, '动作': 4513893000.0, '冒险': 11770987000.0, '剧情': 1833813000.0, '科幻': 1262854000.0, '家庭': 2685246000.0, '爱情': 997498000.0, '悬疑': 605684000.0}
        cla=['喜剧','奇幻','动作','冒险','剧情','科幻','家庭','爱情','悬疑']
        Gijmax=max(dic.values())
        Gijmin=min(dic.values())
        def trans(l):
            l=list(map(lambda x: (np.log(dic[x]/Gijmin))/(np.log(Gijmax/Gijmin)) if x in cla else 0,l
               )
               )
            return sum(l)
        typ=trans(typ)
        X=pd.DataFrame(index=[0],columns=['首日票房','档期','热度指数','是否续集','影片分类','动画','无','日漫','知名公司','知名故事'])
        X.iloc[0,0],X.iloc[0,1],X.iloc[0,2],X.iloc[0,3],X.iloc[0,4]=freven,dangqi,baiduindex,xuji,typ
        X.loc[0,e7.get()]=1
        X.fillna(value=0,inplace=True)
        # print(X)
        pred=voting(final_models,X)#这里的结果是按m1,m2,m3,m4排序的
        cate=['3亿以上','1亿到3亿','1千万到1亿','5百万到1千万','5百万以下']
        pred=list(map(lambda x:cate[int(x[0])],pred))
        window3=Toplevel()
        h=window3.winfo_screenmmheight()
        w=window3.winfo_screenmmwidth()
        window3.geometry('300x300+%d+%d'%(w,h))
        show='''
        电影名称：{}
        预测票房区间：{}
        '''.format(name,pred[-1])
        ll1=Label(window3,text=show)
        
        model_name=['随机森林','XGBoost','GDBT','决策树']
        model_name_selected=[model_name[i] for i in range(len(check_situation)) if check_situation[i]==1]
        show_name='\n'.join(model_name_selected)
        show_re='\n'.join(pred[:-1])
        # print(pred)
        # print(show_name)
        ll2=Label(window3,text='子投票器结果：')
        ll3=Label(window3,text=show_name)
        ll4=Label(window3,text=show_re)
        
        ll1.grid()
        ll2.grid()
        ll3.grid(rowspan=len(model_name_selected))
        ll4.grid(row=2,column=5,rowspan=len(model_name_selected))
        window3.mainloop()

    def back():
        window2.destroy()
        step1()
    
    window2=Tk()
    window2.title('动漫电影票房预测')
    h=window2.winfo_screenmmheight()
    w=window2.winfo_screenmmwidth()
    window2.geometry('900x550+%d+%d'%(w,h))
    pl=['图片/模型训练中.gif','图片/哪吒.gif','图片/熊出没.gif','图片/千与千寻.gif']
    photoindex=random.randint(0,3)
    photo=PhotoImage(file=pl[photoindex])
    image=Label(image=photo)
    hint='''
    温馨提示：
    1、电影名称非必需项
    2、如电影类型有多个，则填入时以空格分隔，电影类型从【喜剧、奇幻、动作、冒险、剧情、科幻、家庭、爱情、悬疑】中选取
    3、IP类型只能填一个，从【知名故事，知名公司，动画，日漫、无】中选取
    4、热门档期是指在暑期、春节、国庆、五一、元旦档期
    '''
    l0=Label(window2,text=hint)
    
    l1=Label(window2,text='请输入电影名称')
    e1=Entry(window2)
    
    l2=Label(window2,text='请输入电影首日票房')
    e2=Entry(window2)
    
    
    l3=Label(window2,text='是否在热门档期')
    rVar1=IntVar()
    r1=Radiobutton(window2,text='是',variable=rVar1,value=1)
    r2=Radiobutton(window2,text='否',variable=rVar1,value=0)
    
    
    l4=Label(window2,text='请输入百度指数峰值')
    e4=Entry(window2)
    
    l5=Label(window2,text='是否是续集')
    rVar2=IntVar()
    r3=Radiobutton(window2,text='是',variable=rVar2,value=1)
    r4=Radiobutton(window2,text='否',variable=rVar2,value=0)
    
    l6=Label(window2,text='请输入影片分类')
    e6=Entry(window2)
    
    l7=Label(window2,text='请输入IP类型')
    e7=Entry(window2)
    
    b1=Button(window2,text='     预测     ',command=predict)
    b2=Button(window2,text='返回上一级',command=back)
    
    
    l1.grid(sticky=W)
    e1.grid(sticky=W)
    l2.grid(sticky=W)
    e2.grid(sticky=W)
    l3.grid(sticky=W)
    r1.grid(sticky=W)
    r2.grid(row=5,column=1,sticky=W)
    l4.grid(sticky=W)
    e4.grid(sticky=W)
    l5.grid(sticky=W)
    r3.grid(sticky=W)
    r4.grid(row=9,column=1,sticky=W)
    l6.grid(sticky=W)
    e6.grid(sticky=W)
    l7.grid(sticky=W)
    e7.grid(sticky=W)
    image.grid(row=5,column=10,rowspan=10)
    b1.grid(row=15,column=8)
    b2.grid(row=15,column=9,sticky=E)
    l0.grid(row=0,column=3,rowspan=5,sticky=N,columnspan=20)
    window2.mainloop()
    
step1()

