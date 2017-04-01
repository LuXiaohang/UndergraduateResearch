#coding=utf8
import MySQLdb
import random
import os
import matplotlib.pyplot as plt
import numpy as np


from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential,model_from_json
from keras.layers import Dense, Dropout, Activation, Embedding,TimeDistributed,Flatten
from keras.layers import LSTM, SimpleRNN, GRU



'''
这个function是传统的滑动窗口，而且是不同的时间周期的滑动窗口

这个模型的特点是前面的窗口需要固定，而不是变长的,并且该模型的后置窗口为1，只是对于课程阶段性的解读和建模求解问题
其指在解决课程进程中如何对于课程动态的进行监控。

讨论课程的学习周期的情况，前置的窗口说明的是如何将窗口视为一个

'''

'''滑动窗口'''
WINDOWS_WIDTH = 3
THETA=1


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



def data_test(data_dir,begin_week,end_len):
    '''做了week2feature后的数据的样子是
     '5178338': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 34.0, 22.0, 10.0, 0.0, 4.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    '''
    x1=[]
    y1=[]
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
        next_week_alive=[i[0] for i in week2feature2(week+1,data_dir)]
        next_week_alive=map(lambda x:1 if x[0] in next_week_alive else 0,before_week_feature)
        X=[i[1] for i in before_week_feature]
        Y=next_week_alive
        if sum(Y)<10:break
        from sklearn.cross_validation import KFold
        kf=KFold(len(X),n_folds=5,shuffle=True)
        rs0=[]
        rs1=[]
        rs2=[]
        rs3=[]
        rs4=[]
        rs5=[]
        print len(X)
        lstm_flag=1
        for train_index,test_index in kf:
            print len(train_index),len(test_index)
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

            m1=logistic_reg_test(train_X,train_Y,test_X,test_Y)
            m2=svc_class_test(train_X,train_Y,test_X,test_Y)
            # m3=naive_bayes_test(train_X,train_Y,test_X,test_Y)B
            if lstm_flag:
                print 'mlp'
                m3=mlp_nn(train_X,train_Y,test_X,test_Y)
                print 'lstm'
                m4=lstm_test(train_X,train_Y,test_X,test_Y)
                lstm_flag=0
                rs3.append(m3[0])
                rs4.append(m4[0])

            assert m1[1]==m2[1],'Something Wrong'

            rs0.append(m1[1])
            rs1.append(m1[0])
            rs2.append(m2[0])
            break

        x1.append(week)
        v0=sum(rs0)/len(rs0)
        v1=sum(rs1)/len(rs1)
        v2=sum(rs2)/len(rs2)
        v3=sum(rs3)/len(rs3)
        v4=sum(rs4)/len(rs4)
        y1.append([v0,v1,v2,v3,v4])

    return x1,y1
   


def mlp_nn(train_X,train_Y,test_X,test_Y):
    '''LSTM'''
    # print train_X.shape
    input_len=len(train_X[0])
    model = Sequential()
    model.add(Dense(64, input_dim=input_len))
    # model.add(Embedding(DICT_SIZE, EMBED_SIZE, input_length=MAX_SENTENCE_LEN))
    # model.add(Embedding(19, 10, input_length=input_len, mask_zero=True))
    # # model.add(TimeDistributed(Dense(NUM_CLASS, activation='softmax'))) 
    model.add(Dropout(0.2))
    model.add(Dense(32))
    model.add(Dropout(0.5))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy',                                   
                  optimizer='rmsprop',                                               
                  metrics=['accuracy'])
    model.fit(train_X, train_Y, batch_size=100,verbose=1, nb_epoch=20)

    rs = model.predict_classes(test_X)

    correct=0
    i_guess_wrong=0
    for i,j in zip(rs,test_Y):
        if j==0:i_guess_wrong+=1
        if i==j:
            correct+=1
    return correct*1.0/len(test_Y),i_guess_wrong*1.0/len(test_Y)


def lstm_test(train_X,train_Y,test_X,test_Y):
    '''LSTM'''
    # print len(train_X),sum(train_Y)
    input_len=len(train_X[0])

    train_X=np.array(train_X)
    test_X=np.array(test_X)
    train_X = np.reshape(train_X, (train_X.shape[0], 19, train_X.shape[1]/19))
    test_X = np.reshape(test_X, (test_X.shape[0], 19, test_X.shape[1]/19))
    # print train_X.shape
    model = Sequential()
    model.add(LSTM(64, input_dim=WINDOWS_WIDTH))
   # model.add(Embedding(DICT_SIZE, EMBED_SIZE, input_length=MAX_SENTENCE_LEN))
    # model.add(Embedding(19, 10, input_length=input_len, mask_zero=True))
    # # model.add(TimeDistributed(Dense(NUM_CLASS, activation='softmax'))) 
    model.add(Dropout(0.5))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy',                                   
                  optimizer='rmsprop',                                               
                  metrics=['accuracy'])
    model.fit(train_X, train_Y, batch_size=100,verbose=1, nb_epoch=200)

    rs = model.predict_classes(test_X)
    correct=0
    i_guess_wrong=0
    for i,j in zip(rs,test_Y):
        if j==0:i_guess_wrong+=1
        if i==j:
            correct+=1
    return correct*1.0/len(test_Y),i_guess_wrong*1.0/len(test_Y)


def logistic_reg_test(train_X,train_Y,test_X,test_Y):
    '''使用逻辑回归，得到训练数据和测试数据的正确性'''
    from sklearn import linear_model

    ols = linear_model.LogisticRegression()
    ols.fit(train_X, train_Y)
    
    rs=ols.predict(test_X)
    correct=0
    i_guess_wrong=0
    for i,j in zip(rs,test_Y):
        if j==0:i_guess_wrong+=1
        if i==j:
            correct+=1
    return correct*1.0/len(test_Y),i_guess_wrong*1.0/len(test_Y)

def svc_class_test(train_X,train_Y,test_X,test_Y):
    '''使用SVC，其中核函数为default=’rbf'''
    from sklearn.svm import SVC
    clf = SVC()
    clf.fit(train_X, train_Y)
    rs=clf.predict(test_X)  
    correct=0
    i_guess_wrong=0
    for i,j in zip(rs,test_Y):
        if j==0:i_guess_wrong+=1
        if i==j:
            correct+=1
    return correct*1.0/len(test_Y),i_guess_wrong*1.0/len(test_Y)

# def naive_bayes_test(train_X,train_Y,test_X,test_Y):
#     '''朴素贝叶斯，使用高斯'''
#     from sklearn.naive_bayes import GaussianNB
#     gnb = GaussianNB()

#     gnb.fit(train_X, train_Y)
 
#     rs=gnb.predict(test_X)
#     correct=0
#     i_guess_wrong=0
#     for i,j in zip(rs,test_Y):
#         if j==0:i_guess_wrong+=1
#         if i==j:
#             correct+=1
#     return correct*1.0/len(test_Y),i_guess_wrong*1.0/len(test_Y)




def plot_the_result(x,y):
    print len(x),'*'*50
    # plt.axes([0.14,0.11,1.2,1.])

    line1,=plt.plot(x,y[:,0],color="black",linewidth=2.0,linestyle='-',label='baseline',marker='*')
    line2,=plt.plot(x,y[:,1],color="blue",linestyle='--',linewidth=2.0,label='log',marker='*')
    line3,=plt.plot(x,y[:,2],color="green",linestyle='--',linewidth=2.0,label='svm',marker='*')
    line4,=plt.plot(x,y[:,3],color='c',linestyle='--',linewidth=2.0,label='MLP',marker='*')
    line5,=plt.plot(x,y[:,4],color='r',linestyle='--',linewidth=2.0,label='LSTM',marker='*')
    plt.legend([line1,line2,line3,line4,line5],[r'Baseline',r'LR',r'SVM','MLP','LSTM'],\
        loc=2,bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    # legend=plt.legend([line1,line2,line3],[r'Baseline',r'LR',r'SVM'],\
    #     loc=2, borderaxespad=0.,bbox_to_anchor=(1.01, 1))
    plt.xticks(x,[str(i) for i in x])
    plt.xlim(x.min()-1,x.max()+1)
    plt.ylim(0.5,1.)
    plt.xlabel(r'Week')
    plt.ylabel(r'Accuracy')



def plot_from_file():
    x1=np.load("methodologysocial-001_modelx1.npy")
    y1=np.load("methodologysocial-001_modely1.npy")
    plot_the_result(x1,y1)
    plt.title("methodologysocial-001")
    plt.show()

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
        data_dir=os.path.join('click_data',i)
        x1,y1=data_test(data_dir,beg,end)
        x1=np.array(x1)
        y1=np.array(y1)

        np.save(i+'_modelx1',x1)
        np.save(i+'_modely1',y1)
        plot_the_result(x1,y1)
        plt.title(i)
        plt.savefig(i+'png')

if __name__ == '__main__':
    pass
    main()
    # plot_from_file()

