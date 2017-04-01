#coding=utf8
import os
import os.path
import gzip,MySQLdb 
import sys




class Data_input:

    def __init__(self,dirname):
        '''dirname为coursera1等
        该类主要实现的是如何进行一个数据的整理的过程，主要包括
        1.    解压数据到对应的data文件目录下
        2.    产生SQL文件到coursera1等目录下
        # reload(sys) 
        # sys.setdefaultencoding('utf-8') 
        # data1=Data_input('coursera3')
        # # data1.tar_dir() #解压各种文件
        # data1.generate_sql_dir() # 产生sql文件夹

        '''
        self.dirname=dirname
        self.dir_dict=self._get_dir_dict()


    def _get_dir_dict(self):
        dir_list=os.listdir(self.dirname)
        dir_dict=dict()
        for course_name in dir_list:
            x=course_name
            course_name=os.path.join(self.dirname,course_name)
            if os.path.isdir(course_name):
                dir_f_list=os.listdir(course_name)
                for filename in dir_f_list:
                    _idx=filename.find('_click')
                    if _idx!=-1:
                        dir_dict[x.decode('gbk')]=filename[:_idx]
        rs_filename=os.path.join(self.dirname,'course_name.txt')
        f=file(rs_filename,'w')
        for i in dir_dict:
            f.write(i.encode('utf8')+','+dir_dict[i]+'\n')
        return dir_dict

    def tar_dir(self):
        for course_name in os.listdir(self.dirname):
            x=course_name
            course_name=os.path.join(self.dirname,course_name)
            if os.path.isdir(course_name):
                self.tar_file(course_name)


                

    def tar_file(self,x):
        '''给一个文件夹名字，遍历文件夹下的gz文件，并且对其进行解压处理,返回'''
        list_file= os.listdir(x)
        data_dir=os.path.join(x,'./data')
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        for i in list_file:
            filepath=os.path.join(x,i)
            if os.path.isfile(filepath):
                filename='./data/'+i[:i.find('.')]+'.sql'
                if filepath.endswith('.sql.gz'):
                    g = gzip.open(filepath,'rb')
                    f=file(os.path.join(x,filename),'wb')
                    f.write(g.read())
                elif filepath.endswith('.sql'):
                    g=file(filepath,'rb')
                    f=file(os.path.join(x,filename),'wb')
                    f.write(g.read())


    def generate_sql_dir(self):
        for course_name in os.listdir(self.dirname):
            x=course_name
            course_name=os.path.join(self.dirname,x)
            if os.path.isdir(course_name):
                self.generate_sql(x)

    def generate_sql(self,x):
    	'''生成sql文件,使用source 文件即可'''
        y=x
        x=os.path.join(self.dirname,x)
        x=x+'/data'
        f2=file('./'+self.dirname+'/'+self.dir_dict[y.decode('gbk')]+'.sql','w')
        f2.write('create database `%s` default charset=utf8;\n'%self.dir_dict[y.decode('gbk')])
        f2.write('use `%s`;\n'%self.dir_dict[y.decode('gbk')])
        f2.write('SET NAMES utf8mb4;\n')
        f2.write('SET CHARACTER SET utf8;\n')
        f2.write('SET character_set_connection=utf8;\n')
        f2.write('tee %s.logs\n'%self.dir_dict[y.decode('gbk')])
        list_file=os.listdir(x)
        for i in list_file:
            filepath=os.path.join(x,i)
            if os.path.isfile(filepath):
                if filepath.endswith('.sql') and filepath.find('all.sql')==-1:
                    f2.write('source '+os.path.abspath(filepath)+'\n')





if __name__=='__main__':
    pass
    # reload(sys) 
    # sys.setdefaultencoding('utf-8') 
    # data1=Data_input('coursera3')
    # # data1.tar_dir() #解压各种文件
    # data1.generate_sql_dir() # 产生sql文件夹

