#coding=utf8


import os
from collections import defaultdict


import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
course_list=[
"20cnwm-001",
"algorithms-001",
"aoo-001",
"aoo-002",
"arthistory-001",
"bdsalgo-001",
"biologicalevolution-001",
"bjmuepiabc-001",
"catmooc-002",
"chemistry-001",
"chemistry-002",
"chemistry-003",
"criminallaw-001",
"dsalgo-001",
"electromagnetism-002",
"englishspeech-001",
"epiapps-001",
# "medstat-001",
"methodologysocial-001",
"methodologysocial2-001",
"orgchem-001",
"os-001",
"osvirtsecurity-002",
"peopleandnetworks-001",
"pkuacc-001",
"pkubioinfo-001",
"pkubioinfo-002",
"pkubioinfo-003",
"pkuco-002",
"pkuic-001",
"pkuic-002",
"pkupop-001",
"undpcso-001",
]

FEA_DIR='fea2'
CLICK_DIR='click_data'

def fea_list():
    course_list1=[
    "methodologysocial-001",
    "methodologysocial2-001",
    "pkubioinfo-001",
    "pkubioinfo-002",
    "pkubioinfo-003",
    ]
    for coursename in course_list1:
        course_dir=os.path.join(CLICK_DIR,coursename)
        rs_line=''
        rs_line2=''
        rs_line=defaultdict(list)
        rs_line2=defaultdict(list)
        for week in range(1,40):
            week_cur='week'+str(week)+'.txt'
            week_filename=os.path.join(course_dir,week_cur)
            if not os.path.exists(week_filename):
                break
            week_f=file(week_filename)
            for line in week_f.readlines():
                line=line.strip()
                if len(line)<1:
                    break
                line_list=line.split(',')

                name=line_list[0]
                ## 之前并没有出现该向量。
                # if name=="7000479":
                #     print week,len(rs_line[name]),19*(week-1),(week-1)-len(rs_line[name])
                #     print rs_line2[name]
                if name not in rs_line:
                    rs_line[name]=[0. for i in range(19*(week-1))]
                    rs_line2[name]=[0 for i in range(week-1)]

                if len(rs_line[name])<19*(week-1):
                    rs_line[name].extend([0. for i in range((week-1)*19-len(rs_line[name]))])
                    rs_line2[name].extend([0 for i in range((week-1)-len(rs_line2[name]))])
                # if name=="7000479":
                #     print week,len(rs_line[name]),19*(week-1),
                #     print rs_line2[name]
                rs_line2[name].append(1)
                rs_line[name].extend([float(t) for t in line_list[1:]])

        print coursename,week

        if not os.path.exists(FEA_DIR):
            os.mkdir(FEA_DIR)

        rs_filename=os.path.join(FEA_DIR,coursename+'.fea')
        rs_filename2=os.path.join(FEA_DIR,coursename+'.fea2')
        
        rs_f=file(rs_filename,'w')
        for line in rs_line:
            print>>rs_f,line+' '+','.join([str(t) for t in rs_line[line]])+','+\
                ','.join([str(0.) for t in range(week*19-len(rs_line[line]))])
        
        rs_f=file(rs_filename2,'w')
        for line in rs_line2:
            print>>rs_f,line+' '+','.join([str(t) for t in rs_line2[line]])+','+\
                ','.join([str(0) for t in range(week-len(rs_line2[line]))])


def read_score(score_filename):
    '''读取分数'''
    f=file(score_filename)
    score_dic=dict()
    for i in f.readlines():
        i=i.strip()
        i_list=i.split(' ')
        score_dic[i_list[0]]=i_list[1]
    return score_dic


def begin_time():
    '''计算最大学习周期，0001序列，直到最后一个1为止，定义一个学习周期'''

    course_list2=[
    "methodologysocial-001",
    "methodologysocial2-001",
    "pkubioinfo-001",
    "pkubioinfo-002",
    "pkubioinfo-003"
]
    rs_f=file('learning_period22.txt','w')
    c_list2=[]
    for coursename in course_list2:
        num=0
        fea_name=os.path.join(FEA_DIR,coursename+'.fea2')
        fea_f=file(fea_name)
        fea_line=[i.strip().split() for i in fea_f.readlines()]
        fea_dic=dict([(i[0],i[1].split(',')) for i in fea_line])
        print>>rs_f,coursename
        course_score_name=os.path.join(CLICK_DIR,coursename+'_score.data')
        score_dic=read_score(course_score_name)
        c_list=[]
        
        for i in score_dic:
            if i in fea_dic and float(score_dic[i])>=60.0:
                def func(line_list):
                    line_str=''.join(line_list)
                    index=line_str.rfind('1')
                    return line_str[:index+1]
                num+=1
                rs=func(fea_dic[i])
                print>>rs_f, i,rs
                c_list.append(len(rs))
        c_list2.append(c_list)
        print coursename,max(c_list),min(c_list),stats.mode(c_list)[0][0],np.mean(c_list)
        print>>rs_f
    plt.hist(c_list2[0],normed=1)
    plt.show()
    rs_f=file('tmp.csv','w')
    for c_name,t in zip(course_list2,c_list2):
        print>>rs_f,c_name+','+','.join([str(i) for i in t])



    

if __name__ == '__main__':
    fea_list()


