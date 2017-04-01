#coding=utf8

import MySQLdb
import os
import json
import numpy as np

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
"medstat-001",
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



CLICK_DIR='click_data'
BEGIN_TIME=1410710400
# HOST='127.0.0.1'
# USER='root'
# PASSWD='admin'
HOST='162.105.14.102'
USER='admin'
PASSWD='admin'


def db_query_time(db_name):
    connection = MySQLdb.connect(HOST, USER, PASSWD, db_name)
    cursor = connection.cursor()
    connection .set_character_set('utf8')
    cursor.execute('SET NAMES utf8;')
    cursor.execute('SET CHARACTER SET utf8;')
    cursor.execute('SET character_set_connection=utf8;')
    sql='select open_time,close_time from announcements'
    cursor.execute(sql)
    _rs=cursor.fetchall()
    rs=_rs[9:]

    ## 规则2 判断第一个open_time不是空的
    tmp_iter=0
    while tmp_iter<5:
        begin=rs[tmp_iter][0]
        tmp_iter+=1
        if begin>0:break
    end=rs[-1][1]
    print db_name,begin,end
        
    # f=file(db_name+'_announcement.csv','w')

    # print db_name,len(rs),rs[9]
    # for line in rs:
    #     func=lambda x:str(x).replace('\n','')
    #     result=','.join([func(i) for i in line])
    #     f.write(result+'\n')


def statistics():
    for course_name in course_list:
        course_f_path=os.path.join('trec',course_name)
        course_f=file(course_name)
    for index,line in enumerate(content):
        l_list=line.split()
        if l_list[0] in course_list:
            tmp.append(index)
    tmp.append(len(content))
    print tmp
    for i in range(len(tmp)-1):
        x=tmp[i]
        next_x=tmp[i+1]
        name=content[x].split()[0]
        rs[name]=[]
        for j in range(x+1,next_x):
            rs[name].append(content[j])
    for item in rs:
        print item,
        tmp2=[]
        for index,i in enumerate(rs[item]):
            if index%2==0:
                tmp2.append(i.split()[1])
        print ' '.join(tmp2)

def register_num():
    '''统计人数，统计id_map情况'''
    for course_name in course_list:
        course_name1=course_name+'_id_map.data'
        filename=os.path.join(CLICK_DIR,course_name1)
        f=file(filename)
        line_list=[i for i in f.readlines()]
        num1=len(line_list)
        
        course_name1=course_name+'_grade_rs.data'
        filename=os.path.join(CLICK_DIR,course_name1)
        f=file(filename)

        line_list2=[i.strip() for i in f.readlines()]

        line_list3=[i.split() for i in line_list2]
        num2=len(line_list3)
        #有分数
        line_list4=[i for i in line_list3 if float(i[0])>0.1]

        num3=len(line_list4)
        # 不是None
        line_list5=[i for i in line_list3 if i[2]!='none']
        num4=len(line_list5)
        print course_name,num1,num2,num3,num4

def db_query_sql(sql,db_name):
    connection = MySQLdb.connect(HOST, USER, PASSWD, db_name)
    cursor = connection.cursor()
    connection .set_character_set('utf8')
    cursor.execute('SET NAMES utf8;')
    cursor.execute('SET CHARACTER SET utf8;')
    cursor.execute('SET character_set_connection=utf8;')
    cursor.execute(sql)
    rs=cursor.fetchall()
    cursor.close()
    connection.close()
    return rs

def read_id_map(course_name):
    '''获取id_map_filename的数据'''
    id_map_filename=course_name+'_id_map.data'
    id_map_filename=os.path.join(CLICK_DIR,id_map_filename)
    f=file(id_map_filename)
    id_map=dict()
    for i in f.readlines():
        i=i.strip()
        i_list=i.split(' ')
        id_map[i_list[0]]=i_list[1]

    return id_map


def grade_makers():
    '''获得最后童鞋们的成绩，记录在$course_name$_grade_rs.data中'''
    for course_name in course_list:
        id_map=read_id_map(course_name)
        sql="select session_user_id,normal_grade,achievement_level from course_grades"
        rs=db_query_sql(sql,course_name)
        rs_fname=course_name+'_grade_rs.data'
        rs_fname=os.path.join(CLICK_DIR,rs_fname)
        rs_f=file(rs_fname,'w')
        num=0
        for i in rs:
            grade=i[1]
            try:
                id=id_map[i[0]]
                normal_level=i[2]
                if int(grade)>=60:num+=1 
                print>>rs_f,grade,id,normal_level
            except KeyError as e:
                print e,' KeyError'
            
        print course_name,num

def click_or_forum():
    for course_name in course_list:
        # 遍历course_name下的文件，得到参与过论坛讨论的用户的集合
        course_week_dir=os.path.join(CLICK_DIR,course_name)
        _id_list=[]
        for filename in os.listdir(course_week_dir):
            filename=os.path.join(course_week_dir,filename)
            f=file(filename)
            def has_forum_data(line_list):
                forum_data=line_list[-8:]
                for j in forum_data:
                    if int(j)>0:return True
                return False
            line_list=[i.split(',') for i in f.readlines()]
            id_list=[i[0] for i in line_list if has_forum_data(i)]
            _id_list.extend(id_list)
        has_forum_id_set=set(_id_list)
        

        #得到取得了成绩的用户的集合
        grade_rs_fname=course_name+'_grade_rs.data'
        grade_rs_fname=os.path.join(CLICK_DIR,grade_rs_fname)
        grade_rs_f=file(grade_rs_fname)
        grade_list=[i.split() for i in grade_rs_f.readlines()]

        user1=set([i[1] for i in grade_list if float(i[0])>0.1])
        user2=set([i[1] for i in grade_list if float(i[0])>60.0])

        union_1=user1 & has_forum_id_set
        union_2=user2 & has_forum_id_set

        print course_name,\
        len(has_forum_id_set),len(user1),len(user2),\
        len(union_1),len(union_2)

def click_or_forum2():
    '''以周围单位的点击和论坛讨论参与情况'''
    for course_name in course_list:
        # 遍历course_name下的文件，得到每周参与过论坛讨论的用户
        course_week_dir=os.path.join(CLICK_DIR,course_name)
        print course_name
        for filename in os.listdir(course_week_dir):
            filename=os.path.join(course_week_dir,filename)
            f=file(filename)
            def has_forum_data(line_list):
                forum_data=line_list[-8:]
                for j in forum_data:
                    if int(j)>0:return True
                return False
            line_list=[i.split(',') for i in f.readlines()]
            id_list=[i[0] for i in line_list if has_forum_data(i)]
            print len(id_list)*1./len(line_list)
        print
        break

def click_times():
    '''便利该所有的点击流数据输出课程名和最早的时间'''
    for course_name in course_list:
        click_f_name=course_name+'.json'
        click_f_name=os.path.join(CLICK_DIR,click_f_name)
        click_f=file(click_f_name)
        json_data=json.load(click_f)
        time_list=[int(i[:10]) for i in json_data]
        if len(time_list)>0:
            print course_name, min(time_list)
        else:
            print course_name


def main():
    click_or_forum2()
   

if __name__ == '__main__':
    main()