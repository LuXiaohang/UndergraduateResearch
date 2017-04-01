#coding=utf8
import MySQLdb
import random
import os
import matplotlib.pyplot as plt
import numpy as np

from collections import defaultdict


from sklearn import cross_validation
import numpy as np
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential,model_from_json
from keras.layers import Dense, Dropout, Activation, Embedding,TimeDistributed,Flatten
from keras.layers import LSTM, SimpleRNN, GRU

'''

这个function是传统的滑动窗口
一方面前置窗口固定，另外一个方面其对于后端的窗口也进行预测，是一个多分类的问题
，加入LSTM


'''

'''滑动窗口'''
WINDOWS_WIDTH = 3

WINDOWS_WIDTH2 = 3
THETA=1
FEA_DIR='fea2'
CLICK_DATA='click_data'

def week2feature(week,now,data_dir):
    '''按照窗口得到前面的feature'''
    assert now-week<WINDOWS_WIDTH
    if int(week)<=0: return []
    now_week=os.path.join(data_dir,'week%s.txt'%str(week))
    now_week=file(now_week)
    now_week_line=[i.strip() for i in now_week.readlines()]
    g=lambda x:x.split(',')[0]#返回用户id
    f=lambda x:[float(i)*(THETA**(now-week)) for i in x.split(',')[1:]]
    before_len=WINDOWS_WIDTH-(now-week)-1
    after_len=now-week
    fea_len=len(f(now_week_line[0]))
    before= [0. for i in range(fea_len*before_len)]
    after = [0. for i in range(fea_len*after_len)]
    return [(g(i),before+f(i)+after) for i in now_week_line]

def week2feature2(week,data_dir):
    now_week=os.path.join(data_dir,'week%s.txt'%str(week))
    now_week=file(now_week)
    now_week_line=[i.strip() for i in now_week.readlines()]
    g=lambda x:x.split(',')[0]#返回用户id
    f=lambda x:[float(i) for i in x.split(',')[1:]]
    return [(g(i),f(i)) for i in now_week_line]



def read_y(course_name):
    course_f_name=os.path.join(FEA_DIR,course_name+'.fea2')
    course_f=file(course_f_name)
    people_fea=defaultdict(list)
    for line in course_f.readlines():
        line=line.strip()
        if len(line)<1:continue
        l_list=line.split()
        k=l_list[0]
        v=l_list[1].split(',')
        people_fea[k]=[float(i) for i in v]
    return people_fea

def read_x(course_name):
    course_f_name=os.path.join(FEA_DIR,course_name+'.fea')
    course_f=file(course_f_name)
    people_fea=defaultdict(list)
    for line in course_f.readlines():
        line=line.strip()
        if len(line)<1:continue
        l_list=line.split()
        k=l_list[0]
        v=l_list[1].split(',')
        people_fea[k]=[float(i) for i in v]
    return people_fea


def data_test(course_name,begin_week,end_len):
    '''
    测试数据
    '''
    x1=[]
    y1=[]
    data_dir=os.path.join(CLICK_DATA,course_name)
    for week in range(WINDOWS_WIDTH+begin_week,end_len):
        week_feature=dict() # 字典型
        now_week=os.path.join(data_dir,'week%s.txt'%str(week))
        next_week=os.path.join(data_dir,'week%s.txt'%str(week+1))
        if not os.path.exists(now_week):break
        if not os.path.exists(next_week):break
        for tt in range(WINDOWS_WIDTH):
            cur_week=week-tt
            cur_week=week2feature(cur_week,week,data_dir)
            for i in cur_week:
                who=i[0]
                if who not in week_feature:
                    week_feature[who]=i[1]
                else:
                    week_feature[who]=[x+y for x,y in zip(week_feature[who],i[1])]
        
        before_week_feature=[]
        for i in week_feature:
            before_week_feature.append((i,week_feature[i]))

        # befere_week_feature的结构是('usr_id',vector) 
        # print before_week_feature[1]
        people_list=[i[0] for i in before_week_feature]

        next_week_alive=[i[0] for i in week2feature2(week+1,data_dir)]
        next_week_alive=map(lambda x:1 if x in next_week_alive else 0,people_list)

        people_fea=read_y(course_name)

        people_fea2=read_x(course_name)


        next_week_alive2=[people_fea[i][week] for i in people_list]

        for p,a,b in zip(people_list,next_week_alive,next_week_alive2):
            if float(a)-b>0:
                print p,a,b
                break
        for index,p in enumerate(people_list):
            a=people_fea2[p][(week-WINDOWS_WIDTH)*19:week*19]
            b=before_week_feature[index][1]
            assert sum(a),sum(b)

        print week,sum(next_week_alive),sum(next_week_alive2)


def y2y(y):
    '''将序列y变为类别y'''
    sum=0
    for index,i in enumerate(y):
        sum+=2**index*i
    return sum

def model_test(course_name,begin_week,end_len):
    '''做了week2feature后的数据的样子是
     '5178338': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 34.0, 22.0, 10.0, 0.0, 4.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    '''
    rs_f=file(course_name+'_model2.tmp','w')
    x1=[]
    y1=[]
    test_y=[]
    data_dir=os.path.join(CLICK_DATA,course_name)
    for week in range(WINDOWS_WIDTH+begin_week,end_len):
        week_feature=dict() # 字典型
        now_week=os.path.join(data_dir,'week%s.txt'%str(week))
        next_week=os.path.join(data_dir,'week%s.txt'%str(week+1))
        if not os.path.exists(now_week):break
        if not os.path.exists(next_week):break
        for tt in range(WINDOWS_WIDTH):
            cur_week=week-tt
            cur_week=week2feature(cur_week,week,data_dir)
            for i in cur_week:
                who=i[0]
                if who not in week_feature:
                    week_feature[who]=i[1]
                else:
                    week_feature[who]=[x+y for x,y in zip(week_feature[who],i[1])]
        
        before_week_feature=[]
        for i in week_feature:
            before_week_feature.append((i,week_feature[i]))

        # befere_week_feature的结构是('usr_id',vector) 
        # print before_week_feature[1]

        X=[i[1] for i in before_week_feature]

        people_list=[i[0] for i in before_week_feature]
        people_fea=read_y(course_name)

        Y=[people_fea[i][week:week+WINDOWS_WIDTH2] for i in people_list]
        from sklearn.cross_validation import KFold
        kf=KFold(len(X),n_folds=5,shuffle=True)
        rs0=[]
        rs1=[]
        for train_index,test_index in kf:
            train_X=[]
            train_Y=[]
            test_X=[]
            test_Y=[]
            for train_i in train_index:
                train_X.append(X[train_i])
                train_Y.append(Y[train_i])
            for test_i in train_index:
                test_X.append(X[test_i])
                test_Y.append(Y[test_i])

            # m2=logistic_reg_test(train_X,train_Y,test_X,test_Y)
            # print m2
            # m2=svc_class_test(train_X,train_Y,test_X,test_Y)
            # m3=naive_bayes_test(train_X,train_Y,test_X,test_Y)
            # assert m1[1]==m2[1],'Something Wrong'
            # assert m1[1]==m3[1],'Something Wrong'
            m1,m1_ = lstm_test(train_X,train_Y,test_X,test_Y)
            m2,m2_ = svm_test(train_X,train_Y,test_X,test_Y)
            print>>rs_f,week,m1,m1_,m2,m2_
            rs0.append(m1*1./m1_)
            rs1.append(m2*1./m2_)
            tmp_y=[y2y(tt) for tt in test_Y]
            test_y.append(test_Y)
            break #由于神经网络有点慢，所以就没有再循环求了。
        
        x1.append(week)
        v0=sum(rs0)/len(rs0)
        v1=sum(rs1)/len(rs1)
        y1.append([v0,v1])

    return x1,y1,test_y

def svm_test(train_X,train_Y,test_X,test_Y):
    '''使用逻辑回归，得到训练数据和测试数据的正确性'''
    from sklearn.svm import SVC

    

    train_Y=[y2y(i) for i in train_Y]
    test_Y=[y2y(i) for i in test_Y]
    clf = SVC()
    clf.fit(train_X, train_Y)
    rs=clf.predict(test_X)  
    correct=0
    for i,j in zip(rs,test_Y):
        if i==j:
            correct+=1
    return correct,len(test_Y)

def lstm_test(train_X,train_Y,test_X,test_Y):
    '''LSTM'''
    # print len(train_X),sum(train_Y)
    input_len=len(train_X[0])

    train_X=np.array(train_X)
    test_X=np.array(test_X)
    # print train_X.shape
    train_X = np.reshape(train_X, (train_X.shape[0], 19, train_X.shape[1]/19))
    test_X = np.reshape(test_X, (test_X.shape[0], 19, test_X.shape[1]/19))
    # print train_X.shape
    model = Sequential()
    model.add(LSTM(64, input_dim=3))
   # model.add(Embedding(DICT_SIZE, EMBED_SIZE, input_length=MAX_SENTENCE_LEN))
    # model.add(Embedding(19, 10, input_length=input_len, mask_zero=True))
    # # model.add(TimeDistributed(Dense(NUM_CLASS, activation='softmax'))) 
    model.add(Dropout(0.5))
    model.add(Dense(WINDOWS_WIDTH2,activation='sigmoid'))
    model.compile(loss='binary_crossentropy',                                   
                  optimizer='rmsprop',                                               
                  metrics=['accuracy'])
    model.fit(train_X, train_Y, batch_size=100,verbose=1, nb_epoch=200)

    rs = model.predict(test_X)
    rs =np.array([[1 if i>0.5 else 0 for i in j] for j in rs]) 
    rs_all=0
    rs_pre=0
    rs_right=0
    line=0
    correct=0

    for i,j in zip(test_Y,rs):
        i1=y2y(i)
        i2=y2y(j)
        if i1==i2:correct+=1

    return correct,len(test_Y)



def plot_the_result(x,y):
    print len(x),'*'*50
    # plt.axes([0.14,0.11,1.2,1.])
    plt.figure(figsize=(5,5),dpi=80)

    line1,=plt.plot(x,y[:,0],color="black",linewidth=2.0,linestyle='-',label='baseline',marker='*')
    line2,=plt.plot(x,y[:,1],color="blue",linestyle='--',linewidth=2.0,label='log',marker='*')
    legend=plt.legend([line1,line2],[r'SVM',r'LSTM'],\
        loc=2, borderaxespad=0.,bbox_to_anchor=(1.01, 1))
    plt.xticks(x,[str(i) for i in x])
    plt.xlim(x.min()-1,x.max()+1)
    plt.ylim(0.5,1.)
    plt.xlabel(r'Week')
    plt.ylabel(r'Accuracy')


course_list=[
("methodologysocial-001",1,14),
("methodologysocial2-001",1,10),
("pkubioinfo-001",3,20),# 2
("pkubioinfo-002",2,10), #这个不要，
("pkubioinfo-003",0,16)
]


def main():
    for item in course_list:
        i=item[0]
        beg=item[1]
        end=item[2]
        x,y,t_y=model_test(i,beg,end)
        np.save(i+'_model2_x1',np.array(x))
        np.save(i+'_model2_y1',np.array(y))
        np.save(i+'_model2_t_y',np.array(t_y))
        print i+' finished!'

if __name__ == '__main__':
    main()

