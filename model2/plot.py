#coding=utf8

import os

import numpy as np

import matplotlib.pyplot as plt
from scipy.stats import mode

course_list=[

"methodologysocial-001",
"methodologysocial2-001",
"pkubioinfo-001",# 2
"pkubioinfo-002", #这个不要，
"pkubioinfo-003"

]

def y2y(y):
    '''将序列y变为类别y'''
    sum=0
    for index,i in enumerate(y):
        sum+=2**index*i
    return sum



def main():
    for course_name in course_list:
        x1=np.load(course_name+'_model2_x1.npy')
        y1=np.load(course_name+'_model2_y1.npy')
        t_y=np.load(course_name+'_model2_t_y.npy')
        X=[]
        svm_y=[]
        lstm_y=[]
        baseline1=[]
        baseline2=[]
        for x,y,t_y in zip(x1,y1,t_y):
            X.append(x)
            lstm_y.append(y[0])
            svm_y.append(y[1])
            t_y2=[y2y(t) for t in t_y]
            v_0=sum([1 if i==0. else 0 for i in t_y2])
            v_7=sum([1 if i==7. else 0 for i in t_y2])
            baseline1.append(v_0*1./len(t_y2))
            baseline2.append(v_7*1./len(t_y2))

        print X,svm_y,lstm_y,baseline1,baseline2
        week_x=np.array(X)
        plt.figure(figsize=(5,5),dpi=80)
        baseline1,=plt.plot(week_x,baseline1,color="black",linewidth=2.0,linestyle='-',label='baseline',marker='^')
        baseline2,=plt.plot(week_x,baseline2,color="black",linewidth=2.0,linestyle='-.',label='baseline2',marker='*')
        svm_line,=plt.plot(week_x,svm_y,color="green",linestyle='--',linewidth=2.0,label='svm',marker='*')
        lstm_line,=plt.plot(week_x,lstm_y,color='r',linestyle='--',linewidth=2.0,label='LSTM',marker='*')
        # plt.legend([baseline1,baseline2,svm_line,lstm_line],[r'Baseline1',r'Baseline2',r'SVM','LSTM'],\
        #     loc=2, bbox_to_anchor=(1.01, 1),borderaxespad=0.)
        legend=plt.legend([baseline1,baseline2,svm_line,lstm_line],[r'Baseline1',r'Baseline2',r'SVM',r'LSTM'],\
            loc=2, borderaxespad=0.,bbox_to_anchor=(1.01, 1))
        plt.xticks(week_x,[str(i) for i in week_x])
        plt.xlim(week_x.min()-1,week_x.max()+1)
        plt.ylim(0.,1.)
        plt.xlabel(r'Week')
        plt.ylabel(r'Accuracy')
        plt.title(course_name)
        # plt.savefig(course_name)
        plt.show()
        break

if __name__ == '__main__':
    main()