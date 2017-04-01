#coding=utf8
import codecs
import json
import random
import os

from collections import defaultdict 

import MySQLdb



CLICK_DIR='click_data'
ERROR_DIR='error'
HOST='127.0.0.1'
USER='root'
PASSWD='admin'

class DataHandle(object):

    def __init__(self,db_name):
        '''该类实现了数据的处理，在CLICK_DIR文件夹中
        1. 得到文件夹 下面是每周的数据
        2. 得到点击流的数据，整理之后重新生成时间在前的数据
        3. 获取id_map数据
        '''
        self.db_name=db_name
        self.db_dir=os.path.join(CLICK_DIR,db_name)
        if not os.path.exists(self.db_dir):
            os.mkdir(self.db_dir)
        
        # 获取数据库的读取游标
        self.cursor=self.__init_cur(db_name)
       
        # 或许点击流数据
        self.click_filename=os.path.join(CLICK_DIR,db_name+'.click')
        self.click_filename2=os.path.join(CLICK_DIR,db_name+'.json')
        if not os.path.exists(self.click_filename2):
            self.__init_click_file()
        
        print 'init click_stream done!'
        # 获取idmap的数据
        self.id_map_filename=os.path.join(CLICK_DIR,db_name+'_id_map.data')
        if not os.path.exists(self.id_map_filename):
            self.__init_id_map_file()

        self.id_map=self.read_id_map()
        print 'init id_map done!'


        #获取id分数的数据
        self.score_filename=os.path.join(CLICK_DIR,db_name+'_score.data')
        if not os.path.exists(self.score_filename):
            self.__init_score_file()

        self.score_dic=self.read_score()
        print 'init score done!'


    def __init_cur(self,db_name):

        self.connection = MySQLdb.connect(HOST, USER, PASSWD, db_name)
        self.cursor = self.connection.cursor()
        self.connection .set_character_set('utf8')
        self.cursor.execute('SET NAMES utf8;')
        self.cursor.execute('SET CHARACTER SET utf8;')
        self.cursor.execute('SET character_set_connection=utf8;')

        return self.cursor

    def db_query(self,sql):
        self.cursor.execute(sql)
        rs=self.cursor.fetchall()
        return rs


    def __init_score_file(self):
        '''写出score文件'''
        sql=("select distinct hash_mapping.user_id,course_grades.normal_grade "
            "from course_grades,hash_mapping where hash_mapping.session_user_id="
            "course_grades.session_user_id")
        rs=self.db_query(sql)
        with open(self.score_filename,'w') as f:
            for i in rs:
                f.write(str(i[0])+' '+str(i[1])+'\n')
        return
        

    def __init_id_map_file(self):
        '''写出id_map文件'''
        sql='select session_user_id,user_id from hash_mapping'
        rs=self.db_query(sql)
        with open(self.id_map_filename,'w') as f:
            for i in rs:
                f.write(str(i[0])+' '+str(i[1])+'\n')
        return  



    def __init_click_file(self):
        '''该函数提取了新的点击数据，并且用时间作为字典的类型，如果一个时间多个事件发生，列表'''
        f=codecs.open(self.click_filename)
        
        output_file=file(self.click_filename2,'w')

        output_data=defaultdict(list)

        for  i in f.readlines():
            i=i.replace("{\\\"height\\\":768,\\\"width\\\":1366}",'0')
            i=i.replace('MIT\\\'',' ')
            
            try:
                json_data=json.loads(i.strip())
            except ValueError as e:
                if not os.path.exists(ERROR_DIR):
                    os.mkdir(ERROR_DIR)
                err_f_path=os.path.join(ERROR_DIR,self.db_name+'.error')
                err_f=file(err_f_path,'a')
                err_f.write("Value Error\n")
                err_f.write(i)
                continue

            if json_data['key']=="pageview":

                data=dict()
                data['key']=json_data['key']
                data['username']=json_data['username']
                data['page_url']=json_data['page_url']
                output_data[json_data['timestamp']].append(data)

            elif json_data['key']=="user.video.lecture.action":
                data=dict()
                data['username']=json_data['username']
                data['key']=json_data['key']
                data['value']=json_data['value']
                data['page_url']=json_data['page_url']
                output_data[json_data['timestamp']].append(data)

        output_file.write(json.dumps(output_data))

        return


    def read_score(self):
        '''读取分数'''
        f=file(self.score_filename)
        score_dic=dict()
        for i in f.readlines():
            i=i.strip()
            i_list=i.split(' ')
            score_dic[i_list[0]]=i_list[1]

        return score_dic

    def read_id_map(self):
        f=file(self.id_map_filename)
        id_map=dict()
        for i in f.readlines():
            i=i.strip()
            i_list=i.split(' ')
            id_map[i_list[0]]=i_list[1]

        return id_map

    def get_begin_end_time(self):
        '''使用announcement判断课程的开始和结束'''
        sql="SELECT open_time,close_time FROM announcements"
        _rs=self.db_query(sql)

        ## rule1 选择10之后
        rs=_rs[9:]

        ## 规则2 判断第一个open_time不是空的
        begin=rs[0][0]

        tmp_iter=0
        while tmp_iter<5: # trick 其实可以报个错误，如果为5了。
            begin=rs[tmp_iter][0]
            tmp_iter+=1
            if begin>0:break

        end=rs[-1][1]

        return (begin,end) 



    def get_datas(self,begin_time):

        click_data=json.load(file(self.click_filename2))
        if not os.path.exists('trec'):
            os.mkdir('trec')
        trec_name=os.path.join('trec',self.db_name+'.trec')
        
        f2=file(trec_name,'a')

        week_num=1
        flag=1 #不退出

        for week in range(50):

            vec=dict()
            beg=begin_time+week*3600*24*7
            end=begin_time+(week+1)*3600*24*7


            # if beg>end_time:break 
            # print beg,'~',end
            # 查看form次数
            sql_view_form=('select user_id,count(*)from activity_log where '
                'action="view.forum"and timestamp>%s and timestamp<%s '
                'group by user_id'%(str(beg),str(end)))
            rs=self.db_query(sql_view_form)
            for line in rs:
                if str(line[0]) not in vec:
                    vec.setdefault(str(line[0]),dict())
                    vec[str(line[0])]['view_forum']=line[1]
                else:
                    vec[str(line[0])]['view_forum']=line[1]

            # 查看帖子次数
            sql_view_thread=('select user_id,count(*) from activity_log where'
                ' action="view.thread" and timestamp>%s and timestamp<%s '
                'group by user_id'%(str(beg),str(end)))
            rs=self.db_query(sql_view_thread)
            for line in rs:
                if str(line[0]) not in vec:
                    vec.setdefault(str(line[0]),dict())
                    vec[str(line[0])]['thread_forum']=line[1]
                else:
                    vec[str(line[0])]['thread_forum']=line[1]

            # sql提交次数
            sql_post_thread=('select user_id,count(*)'
                'from forum_posts where '
                'post_time>%s and post_time<%s '
                'group by user_id'%(str(beg),str(end)))
            rs=self.db_query(sql_post_thread)
            for line in rs:
                if str(line[0]) not in vec:
                    vec.setdefault(str(line[0]),dict())
                    vec[str(line[0])]['post_thread']=line[1]
                else:
                    vec[str(line[0])]['post_thread']=line[1]

            # sql 评论提交次数
            sql_post_comments=('select user_id, count(*) '
                'from forum_comments where '
                'post_time>%s and post_time<%s group '
                'by user_id'%(str(beg),str(end)))
            rs=self.db_query(sql_post_comments)
            for line in rs:
                if str(line[0]) not in vec:
                    vec.setdefault(str(line[0]),dict())
                    vec[str(line[0])]['post_comments']=line[1]
                else:
                    vec[str(line[0])]['post_comments']=line[1]

            # sql 点赞次数
            sql_upvote=('select user_id,count(*) from activity_log where'
                ' action="upvote" and timestamp>%s and timestamp<%s '
                'group by user_id'%(str(beg),str(end)))
            rs=self.db_query(sql_upvote)
            for line in rs:
                if str(line[0]) not in vec:
                    vec.setdefault(str(line[0]),dict())
                    vec[str(line[0])]['upvote']=line[1]
                else:
                    vec[str(line[0])]['upvote']=line[1]

            # 点down次数
            sql_downvote=('select user_id,count(*) from activity_log where'
                ' action="downvote" and timestamp>%s and timestamp<%s '
                'group by user_id'%(str(beg),str(end)))
            rs=self.db_query(sql_downvote)
            for line in rs:
                if str(line[0]) not in vec:
                    vec.setdefault(str(line[0]),dict())
                    vec[str(line[0])]['downvote']=line[1]
                else:
                    vec[str(line[0])]['downvote']=line[1]

            # 增加标签次数
            sql_add_tags=('select user_id,count(*) from activity_log where'
                ' action="tag.add" and timestamp>%s and timestamp<%s '
                'group by user_id'%(str(beg),str(end)))
            rs=self.db_query(sql_add_tags)
            for line in rs:
                if str(line[0]) not in vec:
                    vec.setdefault(str(line[0]),dict())
                    vec[str(line[0])]['add_tag']=line[1]
                else:
                    vec[str(line[0])]['add_tag']=line[1]

            # 删除标签次数
            sql_del_tags=('select user_id,count(*) from activity_log where'
                ' action="tag.delete" and timestamp>%s and timestamp<%s '
                'group by user_id'%(str(beg),str(end)))
            rs=self.db_query(sql_del_tags)
            for line in rs:
                if str(line[0]) not in vec:
                    vec.setdefault(str(line[0]),dict())
                    vec[str(line[0])]['del_tag']=line[1]
                else:
                    vec[str(line[0])]['del_tag']=line[1]

            # 尝试作业次数
            sql_try_hw=('select hash_mapping.user_id,count(assignment_submission_metadata.submission_number) '
                'from hash_mapping,assignment_submission_metadata '
                'where assignment_submission_metadata.session_user_id=hash_mapping.session_user_id '
                'and assignment_submission_metadata.submission_time>%s '
                'and assignment_submission_metadata.submission_time<%s group by hash_mapping.user_id'%(str(beg),str(end)))
            rs=self.db_query(sql_try_hw)
            for line in rs:
                if str(line[0]) not in vec:
                    vec.setdefault(str(line[0]),dict())
                    vec[str(line[0])]['try_hw']=line[1]
                else:
                    vec[str(line[0])]['try_hw']=line[1]

            #尝试quzi次数
            sql_try_quiz=('select hash_mapping.user_id,count(quiz_submission_metadata.submission_number) '
                'from hash_mapping,quiz_submission_metadata '
                'where quiz_submission_metadata.session_user_id=hash_mapping.session_user_id '
                'and quiz_submission_metadata.submission_time>%s '
                'and quiz_submission_metadata.submission_time<%s group by hash_mapping.user_id'%(str(beg),str(end)))
            rs=self.db_query(sql_try_quiz)
            for line in rs:
                if str(line[0]) not in vec:
                    vec.setdefault(str(line[0]),dict())
                    vec[str(line[0])]['try_quiz']=line[1]
                else:
                    vec[str(line[0])]['try_quiz']=line[1]

            #尝试lec次数
            sql_lec_sub=('select hash_mapping.user_id,count(lecture_submission_metadata.submission_number) '
                'from hash_mapping,lecture_submission_metadata '
                'where lecture_submission_metadata.session_user_id=hash_mapping.session_user_id '
                'and lecture_submission_metadata.submission_time>%s '
                'and lecture_submission_metadata.submission_time<%s group by hash_mapping.user_id'%(str(beg),str(end)))
            rs=self.db_query(sql_lec_sub)
            for line in rs:
                if str(line[0]) not in vec:
                    vec.setdefault(str(line[0]),dict())
                    vec[str(line[0])]['try_lec']=line[1]
                else:
                    vec[str(line[0])]['try_lec']=line[1]

            # click stream的数据
            for i in click_data:
                time=int(i[:10])
                if time>beg and time<end:
                    for sth in click_data[i]:
                        username=sth['username']
                        username1=str(self.id_map[username])
                        key=sth['key']
                        if username1 not in vec:
                            vec.setdefault(username1,dict())
                            vec[username1]['page_view']=0
                            vec[username1]['page_view_quiz']=0
                            vec[username1]['page_view_wiki']=0
                            vec[username1]['page_view_forum']=0
                            vec[username1]['page_view_lecture']=0

                            vec[username1]['video_view_times']=0
                            vec[username1]['video_pause_times']=0
                            vec[username1]['video_pause_ratio']=[0,0]
                           

                        if key=='pageview':
                            #print time,username1,vec[username1]
                            if 'page_view' not in vec[username1]:
                                vec[username1]['page_view']=0

                            if 'page_view_quiz' not in vec[username1]:
                                vec[username1]['page_view_quiz']=0

                            if 'page_view_wiki' not in vec[username1]:    
                                vec[username1]['page_view_wiki']=0

                            if 'page_view_forum' not in vec[username1]:    
                                vec[username1]['page_view_forum']=0
                            
                            if 'page_view_lecture' not in vec[username1]: 
                                vec[username1]['page_view_lecture']=0

                            vec[username1]['page_view']+=1

                            url=sth['page_url']
                            if url.find('quiz')!=-1:vec[username1]['page_view_quiz']+=1
                            if url.find('wiki')!=-1:vec[username1]['page_view_wiki']+=1
                            if url.find('lecture')!=-1:vec[username1]['page_view_lecture']+=1
                            if url.find('forum')!=-1:vec[username1]['page_view_forum']+=1

                        else:
                            if 'video_view_times' not in vec[username1]:
                                 vec[username1]['video_view_times']=0
                            if 'video_pause_times' not in vec[username1]:
                                 vec[username1]['video_pause_times']=0
                            if 'video_pause_ratio' not in vec[username1]:
                                 vec[username1]['video_pause_ratio']=[0,0]
                            
                            vec[username1]['video_view_times']+=1
                            value=json.loads(sth['value'])

                            if value['paused']:
                                vec[username1]['video_pause_times']+=1
                            
                            if "playbackRate" in value and value["playbackRate"]:
                                vec[username1]['video_pause_ratio'][0]+=int(value["playbackRate"])
                                vec[username1]['video_pause_ratio'][1]+=1


            print>>f2, week,len(vec)
            
            if len(vec)==0 and flag==0:break # flag退出
            
            if flag==1 and len(vec)!=0:
                flag=0

            # 证明vec>0
            if len(vec)>0:
                fname=os.path.join(self.db_dir,'week%s.txt'%str(week_num))
                week_num+=1
                print>>f2, week_num-1,beg,end

                with open(fname,'w') as f:
                    for j in vec:
                        value=[]
                        value.append('0' if 'page_view' not in vec[j] else vec[j]['page_view'])
                        value.append('0' if 'page_view_quiz' not in vec[j] else vec[j]['page_view_quiz'])
                        value.append('0' if 'page_view_forum' not in vec[j] else vec[j]['page_view_forum'])
                        value.append('0' if 'page_view_lecture' not in vec[j] else vec[j]['page_view_lecture'])
                        value.append('0' if 'page_view_wiki' not in vec[j] else vec[j]['page_view_wiki'])
                        
                        

                        value.append('0' if 'video_view_times' not in vec[j] else vec[j]['video_view_times'])
                        value.append('0' if 'video_pause_times' not in vec[j] else vec[j]['video_pause_times'])
                        ratio='1.0' if 'video_pause_ratio' not in vec[j] or int(vec[j]['video_pause_ratio'][1])<1  else str(float(vec[j]['video_pause_ratio'][0])/vec[j]['video_pause_ratio'][1])
                        value.append(ratio)


                        value.append('0' if 'try_hw' not in vec[j] else vec[j]['try_hw'])
                        value.append('0' if 'try_quiz' not in vec[j] else vec[j]['try_quiz'])
                        value.append('0' if 'try_lec' not in vec[j] else vec[j]['try_lec'])

                        value.append('0' if 'view_forum' not in vec[j] else vec[j]['view_forum'])
                        value.append('0' if 'thread_forum' not in vec[j] else vec[j]['thread_forum'])
                        value.append('0' if 'post_thread' not in vec[j] else vec[j]['post_thread'])
                        value.append('0' if 'post_comments' not in vec[j] else vec[j]['post_comments'])
                        value.append('0' if 'upvote' not in vec[j] else vec[j]['upvote'])
                        value.append('0' if 'downvote' not in vec[j] else vec[j]['downvote'])
                        value.append('0' if 'add_tag' not in vec[j] else vec[j]['add_tag'])
                        value.append('0' if 'del_tag' not in vec[j] else vec[j]['del_tag'])

                        f.write(str(j)+','+','.join([str(x) for x in value])+'\n')





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
course_list1=[
("20cnwm-001","1409557497"),
("algorithms-001","1425205899"),
("aoo-001","1391793933"),
("aoo-002","1425172936"),
("arthistory-001","1380694310"),
("bdsalgo-001","1392018490"),
("biologicalevolution-001","1409812812"),
("bjmuepiabc-001","1409696724"),
("catmooc-002","1425208040"),
("chemistry-001","1378110740"),
("chemistry-002","1392068099"),
("chemistry-003","1422807435"),
("criminallaw-001","1392614163"),
("dsalgo-001","1380590894"),
("electromagnetism-002","1425184176"),
("englishspeech-001","1425180680"),
("epiapps-001","1409696564"),
("methodologysocial-001","1409696799"),
("methodologysocial2-001","1415625135"),
("orgchem-001","1392309607"),
("os-001","1425176009"),
("osvirtsecurity-002","1425193860"),
("peopleandnetworks-001","1380619798"),
("pkuacc-001","1409696599"),
("pkubioinfo-001","1378275819"),
("pkubioinfo-002","1393866931"),
("pkubioinfo-003","1412122444"),
("pkuco-002","1425169375"),
("pkuic-001","1378715462"),
("pkuic-002","1409694283"),
("pkupop-001","1392133790"),
("undpcso-001","1425183460")
]



def main():
    # course_list2=["aoo-002",]
    # for i in course_list1:
    #     print i[0],i[1]
    #     tmp=DataHandle(i[0])
    #     print i,' init finished'
    #     tmp.get_datas(int(i[1]))
    pass

if __name__ == '__main__':
    main()