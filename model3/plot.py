#coding=utf8

import os

import numpy as np

import matplotlib.pyplot as plt
from scipy.stats import mode

course_list=[

("methodologysocial-001",6,14),
("methodologysocial2-001",4,10),
("pkubioinfo-001",7,20),# 2
("pkubioinfo-002",4,10), #这个不要，
("pkubioinfo-003",3,16)

]

def y2y(y):
    '''将序列y变为类别y'''
    sum=0
    for index,i in enumerate(y):
        sum+=2**index*i
    return sum



def main():
    for item in course_list:
        course_name=item[0]
        point=item[1]
        point2=item[2]
        x=np.load(course_name+'_model3_rs_x.npy')
        y=np.load(course_name+'_model3_rs_y.npy')
        y_min=y.min()
        x_max=x.max()
        print x_max
        print course_name
        print x
        print y
        baseline1,=plt.plot(x,y,linewidth=2.0,linestyle='--',label='baseline',marker='*')
        x_line,=plt.plot([x[point],x[point]],[0.4,y[point]],linestyle='--',color='g',alpha=0.5,linewidth=2.0)
        y_line,=plt.plot([0,x_max+1],[y[point],y[point]],linestyle='--',color='r',alpha=0.5,linewidth=2.0)
        plt.annotate(r'$week={}\ auc={:.2f}$'.format(x[point],y[point]), xy=(x[point],y[point]), xytext=(x[point]+1,y[point]-0.05),
            arrowprops=dict(facecolor='black', shrink=0.01),
            )

        x_line,=plt.plot([x[point2-2],x[point2-2]],[0.4,y[point2-2]],linestyle='--',color='g',alpha=0.5,linewidth=2.0)
        y_line,=plt.plot([0,x_max+1],[y[point2-2],y[point2-2]],linestyle='--',color='r',alpha=0.5,linewidth=2.0)
        plt.annotate(r'$week={}\ auc={:.2f}$'.format(x[point2-2],y[point2-2]), xy=(x[point2-2],y[point2-2]), xytext=(x[point2-2]-4,y[point2-2]-0.1),
            arrowprops=dict(facecolor='black', shrink=0.01),
            )

        plt.grid()
        plt.xticks(x,[str(i) for i in x])
        plt.xlim(x.min()-1,x.max()+1)
        plt.ylim(0.4,1.)
        plt.xlabel('week')
        plt.ylabel('AUC')
        plt.title(course_name)
        plt.savefig(course_name)
        plt.show()

if __name__ == '__main__':
    main()