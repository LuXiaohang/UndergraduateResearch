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
不同课程的预测的问题，对于同一门课程不同时间点，
用

'''

'''滑动窗口'''


course_case1=[

("methodologysocial-001",1,14),
("methodologysocial2-001",1,10),

]

course_case2=[
("pkubioinfo-001",4,20),# 2
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


def case1(w1,w2,course_name):
    '''适用于
    ("methodologysocial-001",1,14),
    ("methodologysocial2-001",1,10),
    ("pkubioinfo-003",0,16)
    即不需要合并，只需要进行返回即可
    '''
    course_filename=os.path.join(FEA_DIR,course_name+'.fea')
    course_f=file(course_filename)
    people=defaultdict(list)

    score_filename=os.path.join(CLICK_DATA,course_name+'_score.data')
    people_score=read_score(score_filename)
    for line in course_f.readlines():
        line=line.strip()
        l_list=line.split()
        people[l_list[0]]=[float(i) for i in l_list[1].split(',')]   
    X=[]
    Y=[]
    for p in people:
        if p in people_score:
            X.append(people[p][w1*19:19*(w2)])
            Y.append(1 if float(people_score[p])>60.0 else 0)

    return X,Y

def case2(w1,w2,course_name):
    '''
    ("pkubioinfo-001",4,20),# 2
    需要将前面的合并为一个，然后截取后面的
    '''
    course_filename=os.path.join(FEA_DIR,course_name+'.fea')
    course_f=file(course_filename)
    people=defaultdict(list)

    for line in course_f.readlines():
        line=line.strip()
        l_list=line.split()
        people[l_list[0]]=[float(i) for i in l_list[1].split(',')]   

    score_filename=os.path.join(CLICK_DATA,course_name+'_score.data')
    people_score=read_score(score_filename)

    for p in people:
        p_fea=people[p]
        
        p_w1_later=p_fea[(w1+1)*19:]
        p_w1=p_fea[w1*19:(w1+1)*19]
        #w1之前的加起来
        for i in range(w1):
            p_now=p_fea[i*19:(i+1)*19]
            for index,v in enumerate(p_now):
                p_w1[index]=p_w1[index]+v

        people[p]=p_w1+p_w1_later
    X=[]
    Y=[]
    # w1~w2部分
    for p in people:
        if p in people_score:
            X.append(people[p][0:19*(w2-w1)])
            Y.append(1 if float(people_score[p])>60.0 else 0)

    return X,Y


def problem1():
    '''
    对于第一个问题，用10 预测10 和用 6预测 6 看结果

    为了突出说明6的必要性，来个3
    '''
    course_case1=[
    ("methodologysocial-001",1,14),
    ("methodologysocial2-001",1,10),
    ]

    #PART1
    train_x,train_y=case1(1,10,"methodologysocial-001")
    test_x,test_y=case1(1,10,"methodologysocial2-001")
    print np.array(train_x).shape,np.array(train_y).shape,np.array(test_x).shape,np.array(test_y).shape


    clf1 = SVC(probability=True)

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange'])
    lw = 2

    probas_ = clf1.fit(train_x, train_y).predict_proba(test_x)
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(test_y, probas_[:, 1])
    mean_tpr += np.interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    print roc_auc
    plt.plot(fpr, tpr, lw=lw, color="r",
             label='ROC using 10 week(area = %0.2f)' % (roc_auc))

    #PART2
    train_x,train_y=case1(1,6,"methodologysocial-001")
    test_x,test_y=case1(1,6,"methodologysocial2-001")
    
    print np.array(train_x).shape,np.array(train_y).shape,np.array(test_x).shape,np.array(test_y).shape


    clf2 = SVC(probability=True)

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    lw = 2

    probas_ = clf2.fit(train_x, train_y).predict_proba(test_x)
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(test_y, probas_[:, 1])
    mean_tpr += np.interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    print roc_auc
    plt.plot(fpr, tpr, lw=lw, color="g",
             label='ROC using week6(area = %0.2f)' % (roc_auc))



   

     #PART3
    train_x,train_y=case1(0,2,"methodologysocial-001")
    test_x,test_y=case1(0,2,"methodologysocial2-001")
    
    print np.array(train_x).shape,np.array(train_y).shape,np.array(test_x).shape,np.array(test_y).shape


    clf3 = SVC(probability=True)

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    lw = 2

    probas_ = clf3.fit(train_x, train_y).predict_proba(test_x)
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(test_y, probas_[:, 1])
    mean_tpr += np.interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    print roc_auc
    plt.plot(fpr, tpr, lw=lw, color="c",
             label='ROC using week2(area = %0.2f)' % (roc_auc))


    #结尾工作
    plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='black',
         label='Luck')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('methodologysocial')
    plt.legend(loc="lower right")
    plt.savefig("methodologysocial")
    plt.close()

def problem2():
    '''
     对于第二个用例，将前面4周合并为1周，然后用16 预测 16，和 用7预测7
    '''
    
    course_case2=[
    ("pkubioinfo-001",4,20),# 2
    ("pkubioinfo-003",0,16)
    ]

    #PART1
    train_x,train_y=case2(4,20,"pkubioinfo-001")
    test_x,test_y=case1(0,16,"pkubioinfo-003")
    print np.array(train_x).shape,np.array(train_y).shape,np.array(test_x).shape,np.array(test_y).shape

    clf = SVC(probability=True)

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange'])
    lw = 2

    probas_ = clf.fit(train_x, train_y).predict_proba(test_x)
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(test_y, probas_[:, 1])
    mean_tpr += np.interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    print roc_auc
  
    #PART2
    train_x,train_y=case2(4,11,"pkubioinfo-001")
    test_x,test_y=case1(0,7,"pkubioinfo-003")
    
    print np.array(train_x).shape,np.array(train_y).shape,np.array(test_x).shape,np.array(test_y).shape


    clf = SVC(probability=True)

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    lw = 2

    probas_ = clf.fit(train_x, train_y).predict_proba(test_x)
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(test_y, probas_[:, 1])
    mean_tpr += np.interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    print roc_auc
    plt.plot(fpr, tpr, lw=lw, color="g",
             label='ROC using 7 week(area = %0.2f)' % (roc_auc))


    #PART3
    train_x,train_y=case2(4,8,"pkubioinfo-001")
    test_x,test_y=case1(0,4,"pkubioinfo-003")
    
    print np.array(train_x).shape,np.array(train_y).shape,np.array(test_x).shape,np.array(test_y).shape


    clf = SVC(probability=True)

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    lw = 2

    probas_ = clf.fit(train_x, train_y).predict_proba(test_x)
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(test_y, probas_[:, 1])
    mean_tpr += np.interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    print roc_auc
    plt.plot(fpr, tpr, lw=lw, color="b",
             label='ROC using 4 week(area = %0.2f)' % (roc_auc))

   

     #PART4
    train_x,train_y=case2(4,6,"pkubioinfo-001")
    test_x,test_y=case1(0,2,"pkubioinfo-003")
    
    print np.array(train_x).shape,np.array(train_y).shape,np.array(test_x).shape,np.array(test_y).shape


    clf = SVC(probability=True)

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    lw = 2

    probas_ = clf.fit(train_x, train_y).predict_proba(test_x)
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(test_y, probas_[:, 1])
    mean_tpr += np.interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    print roc_auc
    plt.plot(fpr, tpr, lw=lw, color="c",
             label='ROC using 2 week(area = %0.2f)' % (roc_auc))

    # 其他的
    plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='black',
         label='Luck')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("pkubioinfo")
    plt.legend(loc="lower right")
    plt.savefig("pkubioinfo")
    plt.close()

def problem3():
    '''
     用生物信息预测社会调查
    '''
    
    course_case2=[
    ("pkubioinfo-001",4,10),# 2
    ("pkubioinfo-003",0,16)
    ]

    #PART1
    train_x,train_y=case2(4,14,"pkubioinfo-001")
    test_x,test_y=case1(1,11,"methodologysocial-001")
    print np.array(train_x).shape,np.array(train_y).shape,np.array(test_x).shape,np.array(test_y).shape

    clf = SVC(probability=True)

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange'])
    lw = 2

    probas_ = clf.fit(train_x, train_y).predict_proba(test_x)
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(test_y, probas_[:, 1])
    mean_tpr += np.interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    print roc_auc
    plt.plot(fpr, tpr, lw=lw, color="r",
             label='ROC using 10 week(area = %0.2f)' % (roc_auc))

    #PART2
    train_x,train_y=case2(4,9,"pkubioinfo-001")
    test_x,test_y=case1(1,6,"methodologysocial-001")
    
    print np.array(train_x).shape,np.array(train_y).shape,np.array(test_x).shape,np.array(test_y).shape


    clf = SVC(probability=True)

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    lw = 2

    probas_ = clf.fit(train_x, train_y).predict_proba(test_x)
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(test_y, probas_[:, 1])
    mean_tpr += np.interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    print roc_auc
    plt.plot(fpr, tpr, lw=lw, color="g",
             label='ROC using 5 week(area = %0.2f)' % (roc_auc))



    #PART3
    train_x,train_y=case2(4,7,"pkubioinfo-001")
    test_x,test_y=case1(1,4,"methodologysocial-001")
    
    print np.array(train_x).shape,np.array(train_y).shape,np.array(test_x).shape,np.array(test_y).shape


    clf = SVC(probability=True)

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    lw = 2

    probas_ = clf.fit(train_x, train_y).predict_proba(test_x)
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(test_y, probas_[:, 1])
    mean_tpr += np.interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    print roc_auc
    plt.plot(fpr, tpr, lw=lw, color="b",
             label='ROC using 3 week(area = %0.2f)' % (roc_auc))

    # 基本的
    plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='black',
         label='Luck')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('pkubioinfo 2 methodologysocial')
    plt.legend(loc="lower right")
    plt.savefig("pkubioinfo to methodologysocial")
    plt.close()



def main():
    problem1()
    problem2()
    problem3()
        
  
if __name__ == '__main__':
    main()

