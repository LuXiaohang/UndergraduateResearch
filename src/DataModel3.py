#coding=utf8
import MySQLdb
import random
import os
import matplotlib.pyplot as plt
import numpy as np

from itertools import cycle

from sklearn.model_selection  import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from collections import defaultdict
'''
实现对于以下课程的定长向量的抽取

'''

'''滑动窗口'''


course_list=[

("methodologysocial-001",1,14),
("methodologysocial2-001",1,10),
("pkubioinfo-001",4,20),# 2
("pkubioinfo-002",2,10), #这个不要，
("pkubioinfo-003",0,16)

]

FEA_DIR='fea2'
CLICK_DATA='click_data'


def read_score(score_filename):
    '''读取分数'''
    f=file(score_filename)
    score_dic=dict()
    for i in f.readlines():
        i=i.strip()
        i_list=i.split(' ')
        score_dic[i_list[0]]=i_list[1]
    return score_dic



def data_test():
    for item in course_list:

        course_name=item[0]
        course_begin=item[1]
        course_end=item[2]
        week_list=range(1,course_end)
        course_filename=os.path.join(FEA_DIR,course_name+'.fea')
        course_f=file(course_filename)
        people=defaultdict(list)

        score_filename=os.path.join(CLICK_DATA,course_name+'_score.data')
        people_score=read_score(score_filename)
        for line in course_f.readlines():
            line=line.strip()
            l_list=line.split()
            people[l_list[0]]=[float(i) for i in l_list[1].split(',')]
        
        rs_x=[]
        rs_y=[]
        for week in week_list:
            X=[]
            Y=[]
            for p in people:
                if p in people_score:
                    X.append(people[p][:19*week])
                    Y.append(1 if float(people_score[p])>60.0 else 0)

            print course_name,'data prepared',len(X),sum(Y),
            cv = StratifiedKFold(n_splits=5)
            X=np.array(X)
            y=np.array(Y)
            clf = SVC(probability=True)

            mean_tpr = 0.0
            mean_fpr = np.linspace(0, 1, 100)
            colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange'])
            lw = 2

            i = 0
            for (train, test), color in zip(cv.split(X, y), colors):
                probas_ = clf.fit(X[train], y[train]).predict_proba(X[test])
                # Compute ROC curve and area the curve
                fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
                mean_tpr += np.interp(mean_fpr, fpr, tpr)
                mean_tpr[0] = 0.0
                roc_auc = auc(fpr, tpr)
                # plt.plot(fpr, tpr, lw=lw, color=color,
                #          label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

                i += 1
            # plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k',
            #          label='Luck')

            mean_tpr /= cv.get_n_splits(X, y)
            mean_tpr[-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr)
            # plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--',
            #          label='Mean ROC (area = %0.2f)' % mean_auc, lw=lw)
            print mean_auc
            rs_x.append(week)
            rs_y.append(mean_auc)
        # plt.xlim([-0.05, 1.05])
        # plt.ylim([-0.05, 1.05])
        np.save(course_name+'_model3_rs_x',np.array(rs_x))
        np.save(course_name+'_model3_rs_y',np.array(rs_y))
        print course_name,'finished!'
        # plt.plot(rs_x,rs_y)
        # plt.xlabel('week')
        # plt.ylabel('AUC')
        # plt.legend(loc="lower right")
        # plt.show()

def data_test2():
    '''画出选定的week的5fold的auc的图'''


    course_list1=[

    ("methodologysocial-001",7,14),
    ("methodologysocial2-001",5,10),
    ("pkubioinfo-001",8,20),# 2
    ("pkubioinfo-002",5,10), #这个不要，
    ("pkubioinfo-003",4,16)

    ]
    for item in course_list1:

        course_name=item[0]
        course_week=item[1]
        course_end=item[2]
        week_list=range(1,course_end)

        course_filename=os.path.join(FEA_DIR,course_name+'.fea')
        course_f=file(course_filename)
        people=defaultdict(list)

        score_filename=os.path.join(CLICK_DATA,course_name+'_score.data')
        people_score=read_score(score_filename)
        for line in course_f.readlines():
            line=line.strip()
            l_list=line.split()
            people[l_list[0]]=[float(i) for i in l_list[1].split(',')]

        week=course_week
        
        X=[]
        Y=[]
        for p in people:
            if p in people_score:
                X.append(people[p][:19*(week-1)])
                Y.append(1 if float(people_score[p])>60.0 else 0)

        print course_name,'data prepared',len(X),sum(Y)
        cv = StratifiedKFold(n_splits=5)
        X=np.array(X)
        y=np.array(Y)
        clf = SVC(probability=True)

        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)
        colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange'])
        lw = 2

        i = 0
        for (train, test), color in zip(cv.split(X, y), colors):
            probas_ = clf.fit(X[train], y[train]).predict_proba(X[test])
            # Compute ROC curve and area the curve
            fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
            mean_tpr += np.interp(mean_fpr, fpr, tpr)
            mean_tpr[0] = 0.0
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=lw, color=color,
                     label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

            i += 1
        plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k',
                 label='Luck')

        mean_tpr /= cv.get_n_splits(X, y)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--',
                 label='Mean ROC (area = %0.2f)' % mean_auc, lw=lw)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(course_name+' '+"(week{})".format(week))
        plt.legend(loc="lower right")
        plt.savefig(course_name)
        plt.close()

  
def main():
    data_test()

if __name__ == '__main__':
    data_test2()

