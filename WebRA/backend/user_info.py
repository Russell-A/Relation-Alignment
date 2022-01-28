import path_copy_copy as path
import joblib
from random import shuffle
import util
import numpy as np

file_all_people = set()


inquiry_dict = {
    '0': '用户登录界面',
    '10': '单条路径语义判断',
    '100': '判断哪一条关系存在无关语义',
    '11': '单条路径逻辑结构判断',
    '20': '多条路径语义判断',
    '21': '多条路径逻辑结构判断'
}

options_dict = {
    '0': None,
    '10': ['语义一致', '路径中存在无关语义', '路径语义存在不足','切换图结构','无法判断，跳过'],
    '100': [],  # 判断哪一条路径存在无关语义，长度由rdf长度给出
    '11': ['路径逻辑结构一致', '路径逻辑结构不一致','路径逻辑结构存在不足','切换图结构','无法判断，跳过'],
    '20': ['语义一致', '语义不足/冗余','切换图结构','无法判断，跳过'],
    '21': ['逻辑结构一致', '逻辑结构不一致','切换图结构','无法判断，跳过']
}

next_inquiry_dict = {
    '0': ['10', '20'],
    '10': {
        '语义完全一致': '11',
        '路径中存在无关语义': '100',
        '路径语义存在不足': ['10', '20']
    },
    '100': ['10', '20'],
    '11': {
        '路径逻辑结构一致': '20',
        '路径逻辑结构不一致': ['10', '20']
    },
    '20': {
        '语义完全一致': '21',
        '语义不足/冗余': ['10', '20']
    },
    '21': {
        '路径逻辑结构一致': ['10', '20'],
        '路径逻辑结构不一致': ['10', '20'],
        '路径逻辑结构存在不足': ['10','20'],
    }
}

title_dict = {
    '0': '',
    '10': '图中关系与给出文字关系是否在语义上一致？',
    '100': '图中哪一个关系存在冗余？',
    '11': '图中关系与给出文字关系是否在逻辑结构上一致？',
    '20': '图中关系与给出文字关系是否在语义上一致？',
    '21': '图中关系与给出文字关系是否在逻辑结构上一致？',
}

uid_dict = {}


class user_info():
    def __init__(self, uid):
        self.files_all = joblib.load('./cluster_result/cluster_local_result_70')
        shuffle(self.files_all)
        self.user_id = uid
        self.files_done = []  # 已经完成的文件
        self.inquiry_received = '0'  # 现在最新接收到的query
        self.inquiry_next = None
        self.files_doing = self.files_all[0][0]  # 现在正在处理的文件
        self.result_received = None
        self.structures = path.structure_list(self.files_doing)
        self.structures.shuffle()  # 为了避免我想不到的bug先shuffle一下

        # 执行完send去前端之后用next替换这个
        self.candidate = None
        self.candidate_type = None
        self.candidate_rdf_lists = None
        self.rdfs_list = []
        self.rdf = None
        self.fb_line = None
        self.db_line = None
        self.candidate_degree = 0


        self.next_candidate = None
        self.next_candidate_type = None
        self.next_candidate_rdf_lists = None
        self.next_rdfs_list = []
        self.next_rdf = None
        self.next_fb_line = None
        self.next_db_line = None
        self.next_candidate_degree = 0
        
        self.no_seen = 0
        self.path_seen = [0,0,0]
        self.path_right = [0,0,0]
        self.pass_lingustic = 0
        self.no_right = 0
        self.accquracy = 0
        self.path_len_right = []
        self.path_len = []
        self.degree_right = []
        self.degree = []
        self.time_start = None
        self.times = []
        self.candidate_is_right = None
        self.times_right = []
        self.times_wrong = []


        self.next_inquiry()

    def receive_result(self, result):
        self.result_received = result

    def receive_inquiry(self, inquiry_num):
        self.inquiry_received = inquiry_num

    def add_files_done(self, file):
        self.files_done.append(file)

    def files_to_do_next(self):
        if len(self.files_done) == 1252:
            raise ('所有文件都做完了')
        file_done_last = self.files_done[-1]  # 最后一个做好的文件
        file_all_people.add(file_done_last)
        file_group1 = None  # 记录这一个文件和下一个文件是不是一个文件group里的
        file_group2 = None
        return_file = None
        next1 = False
        if len(file_all_people) < 1252:
            for i in range(len(self.files_all)):
                for j in range(len(self.files_all[i])):
                    file = self.files_all[i][j]
                    if not (file in file_all_people):
                        return_file = file
                        break

                else:
                    continue
                break
        else:
            for i in range(len(self.files_all)):
                for j in range(len(self.files_all[i])):
                    file = self.files_all[i][j]
                    if not (file in self.files_done):
                        return_file = file
                        break
                else:
                    continue
                break
        assert return_file is not None

        if len(file_all_people) == 1252:
            print('all file done once', file = codecs.open('./log/administration.txt','a','utf-8'))
        return return_file, file_group1 == file_group2  # 返回下一个文件还有文件所属的组是不是一致

    def check_next_file(self):
        return self.structures.check_next_file()

    def get_next_candidate(self):
        self.structures.shuffle()
        # print(self.structures.candidate)
        candidate_path, candidate_path_type, rdfs_list, degree = self.structures.return_candidate_rdf()
        self.next_candidate = candidate_path
        self.next_candidate_type = candidate_path_type
        self.next_rdfs_list = rdfs_list
        self.next_candidate_degree = degree
        self.degree.append(degree)

    def get_next_rdf(self):
        if self.next_rdf is None:
            # print('self.next_rdfs_list', self.next_rdfs_list)
            self.next_rdf = self.next_rdfs_list[0][0]
        else:
            raise ('不应该使用这个函数(get_rdf)')
        if self.next_db_line is None:
            self.next_db_line = self.next_rdfs_list[0][1]
        else:
            raise ('不应该使用这个函数(get_rdf)')
        if self.next_fb_line is None:
            self.next_fb_line = self.next_rdfs_list[0][2]
        else:
            raise ('不应该使用这个函数(get_rdf)')

    def change_rdf(self):
        tuple1 = (self.rdf, self.db_line, self.fb_line)
        self.done_to_next()
        if tuple1 in self.rdfs_list:
            # print(tuple1)
            index = self.rdfs_list.index(tuple1)
            # print(index)
            index_next = (index + 1) % len(self.rdfs_list)
            self.next_rdf, self.next_db_line, self.next_fb_line = self.rdfs_list[index_next]
            # print(self.next_rdf)
        else:
            raise ('不该使用这个函数(change_rdf)')

    def return_graph(self, head, tail):
        dict_neighbour = {}
        nodes = list()
        edges = list()
        curve = list()
        dict_hyperlink = dict()
        dict_relation_full = self.structures.dict_relation_full
        num = 0
        if self.next_candidate_type == 'path':
            next_rdf = self.next_rdf
        else:
            next_rdf = set(self.next_rdf)

        for lines in next_rdf:
            num += 1
            # print(lines)
            s, r, o = lines.strip().split('\t')
            if s.startswith('dr:'):
                s = s[3:]
                dict_hyperlink[' '.join(s.split('_'))] = 'https://dbpedia.org/page/'+s
            else:
                if s.startswith('<') and s.endswith('>'):
                    dict_hyperlink[' '.join(s.split('_'))] = s[1:-1]
                else:
                    dict_hyperlink[' '.join(s.split('_'))] = s
            if o.startswith('dr:'):
                o = o[3:]
                dict_hyperlink[' '.join(o.split('_'))] = 'https://dbpedia.org/page/'+o
            else:
                if o.startswith('<') and o.endswith('>'):
                    dict_hyperlink[' '.join(o.split('_'))] = o[1:-1]
                else:
                    dict_hyperlink[' '.join(o.split('_'))] = o
            s = ' '.join(s.split('_'))
            o = ' '.join(o.split('_'))


            edges.append([s, o, str(num) + ':' + r])
            if s in dict_neighbour:
                dict_neighbour[s].append(o)
            else:
                dict_neighbour[s] = [o]
            if o in dict_neighbour:
                dict_neighbour[o].append(s)
            else:
                dict_neighbour[o] = [s]
        start, _, end = self.next_db_line.strip().split(' ')
        start = ' '.join(start.split('_'))
        end = ' '.join(end.split('_'))
        if start.startswith('dr:'):
            start = start[3:]
        if end.startswith('dr:'):
            end = o[3:]
        nodes.append(start)
        node_now = start
        while len(nodes) != len(dict_neighbour.keys()):
            for node in dict_neighbour[nodes[-1]]:
                if node not in nodes:
                    nodes.append(node)
                    node_old = node_now
                    node_now = node
                    break
            else:
                for node1 in dict_neighbour.keys():
                    if node1 not in nodes:
                        nodes.append(node1)
        curve = [0] * len(edges)
        for i in range(len(edges)):
            if curve[i] == 0:
                for j in range(i+1, len(edges)):
                    if edges[i][0] == edges[j][0] and edges[i][1] == edges[j][1]:
                        
                        curve[i] = 0.15
                        curve[j] = -0.15
        hyperlinks_edges = []
        for i in range(len(edges)):
            r = edges[i][2].split(':')[-1]
            rfull = dict_relation_full[r][0]
            if rfull.startswith('<') and rfull.endswith('>'):
                rfull = rfull[1:-1]
            hyperlinks_edges.append(rfull)

        # print('curve',curve)
        # print('edges',edges)
        hyperlinks = []
        for node in nodes:
            hyperlinks.append(dict_hyperlink[node])
        color = []
        for node in nodes:
            if node == head or node == tail:
                color.append('0')
            else:
                color.append('1')
        
        return nodes, edges, curve, hyperlinks, color, hyperlinks_edges

    def add_seen(self, candidate_path, type):
        self.structures.add_seen(candidate_path, type)

    def add_written(self, candidate_path, type):
        self.structures.add_written(candidate_path, type)

    def determine_next_candidate_type_and_get_candidate_and_rdf(self):
        time_start = time.time()
        if not (self.time_start is None):
            self.times.append(time_start-self.time_start)
            if self.candidate_is_right is None:
                pass
            elif self.candidate_is_right == True:
                self.times_right.append(time_start-self.time_start)
            elif self.candidate == False:
                self.times_wrong.append(time_start-self.time_start)
        self.time_start = time.time()
        self.accquracy = self.no_right/self.no_seen if self.no_seen != 0 else 0
        util.write_log(self.user_id,f'已检查的关系对：{self.no_seen}')
        util.write_log(self.user_id,f'通过的关系对：{self.no_right}')
        util.write_log(self.user_id,f'通过语义的关系对：{self.pass_lingustic}')
        util.write_log(self.user_id,f'正确率:{self.accquracy}')
        util.write_log(self.user_id,f'见过的路径平均长度:{np.average(self.path_len)}')
        util.write_log(self.user_id,f'正确的路径平均长度:{np.average(self.path_len_right)}')
        util.write_log(self.user_id,f'见过的路径平均度:{np.average(self.degree)}')
        util.write_log(self.user_id,f'正确的路径平均度:{np.average(self.degree_right)}')
        util.write_log(self.user_id,f'完成的文件数:{len(self.files_done)}')
        util.write_log(self.user_id,f'见过的路径数目:{self.path_seen}')
        util.write_log(self.user_id,f'正确的路径数目:{self.path_right}')
        util.write_log(self.user_id,f'所花时间:{self.times}')

        self.structures.shuffle()
        if self.check_next_file():
            self.add_files_done(self.files_doing)
            file_next, same_group = self.files_to_do_next()
            self.files_doing = file_next
            print('next file:',self.files_doing)
            self.structures = path.structure_list(self.files_doing)

        self.get_next_candidate()
        self.get_next_rdf()
        self.path_len.append(len(self.next_rdf))
        if self.next_candidate_type == 'path':
            self.inquiry_next = '10'
            self.path_seen[len(self.next_candidate)-1] += 1
        elif self.next_candidate_type == 'multipath':
            self.inquiry_next = '20'
        else:
            raise ('next candidate type 出错')
        self.no_seen += 1
        

    def next_inquiry(self):  # 调用了get_next_candidate函数
        assert self.inquiry_received in next_inquiry_dict
        if self.inquiry_received == '0':
            self.determine_next_candidate_type_and_get_candidate_and_rdf()
        elif self.inquiry_received == '10':
            if self.result_received == '语义一致':
                self.structures.pass_linguistic += 1
                self.pass_lingustic +=1
                self.done_to_next()
                self.inquiry_next = '11'
            elif self.result_received == '路径中存在无关语义':
                self.done_to_next()
                self.inquiry_next = '100'
            elif self.result_received == '路径语义存在不足':
                self.add_seen(self.candidate, self.candidate_type)
                self.add_multipath(self.candidate)
                self.candidate_is_right = False
                self.determine_next_candidate_type_and_get_candidate_and_rdf()
            elif self.result_received == '切换图结构':
                self.change_rdf()
                self.inquiry_next = '10'
            elif self.result_received == '无法判断，跳过':
                self.add_seen(self.candidate, self.candidate_type)
                self.candidate_is_right = None
                self.determine_next_candidate_type_and_get_candidate_and_rdf()
        elif self.inquiry_received == '20':
            if self.result_received == '语义一致':
                self.structures.pass_linguistic += 1
                self.pass_lingustic += 1
                self.done_to_next()
                self.inquiry_next = '21'
            elif self.result_received == '语义不足/冗余':
                self.add_seen(self.candidate, self.candidate_type)
                self.candidate_is_right = False
                self.determine_next_candidate_type_and_get_candidate_and_rdf()
            elif self.result_received == '切换图结构':
                self.change_rdf()
                self.inquiry_next = '20'
            elif self.result_received == '无法判断，跳过':
                self.add_seen(self.candidate, self.candidate_type)
                self.candidate_is_right = None
                self.determine_next_candidate_type_and_get_candidate_and_rdf()
        elif self.inquiry_received == '11':
            if self.result_received == '路径逻辑结构一致':
                self.add_written(self.candidate, self.candidate_type)
                self.add_seen(self.candidate, self.candidate_type)
                self.candidate_is_right = True
                util.write_aligned_result(out_file=self.files_doing, fb_line=self.fb_line, rdf_new=self.rdf, uid= self.user_id, dict_relation_full=self.structures.dict_relation_full)
                self.path_len_right.append(len(self.candidate))
                self.degree_right.append(self.candidate_degree)
                self.no_right += 1
                self.path_right[len(self.candidate)-1] += 1
                self.determine_next_candidate_type_and_get_candidate_and_rdf()
            elif self.result_received == '路径逻辑结构不一致':
                self.add_seen(self.candidate, self.candidate_type)
                self.candidate_is_right = False
                self.determine_next_candidate_type_and_get_candidate_and_rdf()
            elif self.result_received == '路径逻辑结构存在不足':
                self.add_seen(self.candidate, self.candidate_type)
                self.add_multipath(self.candidate)
                self.candidate_is_right = False
                self.determine_next_candidate_type_and_get_candidate_and_rdf()
            elif self.result_received == '切换图结构':
                self.change_rdf()
                self.inquiry_next = '11'
            elif self.result_received == '无法判断，跳过':
                self.add_seen(self.candidate, self.candidate_type)
                self.candidate_is_right = None
                self.determine_next_candidate_type_and_get_candidate_and_rdf()
        elif self.inquiry_received == '21':
            if self.result_received == '逻辑结构一致':
                self.add_seen(self.candidate, self.candidate_type)
                self.candidate_is_right = True
                self.add_written(self.candidate, self.candidate_type)
                util.write_aligned_result(out_file=self.files_doing, fb_line=self.fb_line, rdf_new=self.rdf, uid = self.user_id,dict_relation_full=self.structures.dict_relation_full)
                self.no_right += 1
                self.path_len_right.append(len(self.candidate))
                self.degree_right.append(self.candidate_degree)
                self.determine_next_candidate_type_and_get_candidate_and_rdf()
            elif self.result_received == '逻辑结构不一致':
                self.add_seen(self.candidate, self.candidate_type)
                self.candidate_is_right = False
                self.determine_next_candidate_type_and_get_candidate_and_rdf()
            elif self.result_received == '切换图结构':
                self.change_rdf()
                self.inquiry_next = '21'
            elif self.result_received == '无法判断，跳过':
                self.add_seen(self.candidate, self.candidate_type)
                self.candidate_is_right = None
                self.determine_next_candidate_type_and_get_candidate_and_rdf()
        elif self.inquiry_received == '100':
            self.add_seen(self.candidate, self.candidate_type)
            self.structures.delete_unrelated_relations(self.candidate, int(self.result_received))
            self.candidate_is_right = False
            self.determine_next_candidate_type_and_get_candidate_and_rdf()

    def return_options(self):
        assert self.inquiry_next in options_dict
        if self.inquiry_next == '100':
            length = len(self.next_rdf)
            options = []
            for i in range(length):
                options.append(str(i + 1))
            return options
        else:
            options = options_dict[self.inquiry_next]
            return options

    def done_to_next(self):
        self.next_candidate = self.candidate
        self.next_candidate_type = self.candidate_type
        self.next_candidate_rdf_lists = self.candidate_rdf_lists
        self.next_rdfs_list = self.rdfs_list
        self.next_rdf = self.rdf
        self.next_fb_line = self.fb_line
        self.next_db_line = self.db_line
        self.next_candidate_degree = self.candidate_degree

    def next_to_done(self):
        self.candidate = self.next_candidate
        self.candidate_type = self.next_candidate_type
        self.candidate_rdf_lists = self.next_candidate_rdf_lists
        self.rdfs_list = self.next_rdfs_list
        self.rdf = self.next_rdf
        self.fb_line = self.next_fb_line
        self.db_line = self.next_db_line
        self.candidate_degree = self.next_candidate_degree

        self.next_candidate = None
        self.next_candidate_type = None
        self.next_candidate_rdf_lists = None
        self.next_rdfs_list = []
        self.next_rdf = None
        self.next_fb_line = None
        self.next_db_line = None
        self.next_candidate_degree = 0
    
    def add_multipath(self,relation):
        self.structures.add_multipath(relation)


def get_result(inquiry_num, uid, result):
    assert uid in uid_dict
    user_information = uid_dict[uid]
    ''':type user_information: user_info'''
    user_information.receive_result(result)
    user_information.receive_inquiry(inquiry_num)
    user_information.next_to_done()
    user_information.next_inquiry()


def send_information(uid):
    assert uid is not None
    if not (uid in uid_dict):
        user_information = user_info(uid)
        uid_dict[uid] = user_information
    else:
        user_information = uid_dict[uid]
    ''':type user_information: user_info'''
    
    option = user_information.return_options()
    inquiry_id = user_information.inquiry_next
    uid = uid
    title = title_dict[inquiry_id]
    s, r, o = user_information.next_db_line.split(' ')
    s = ' '.join(s.split('_')); o = ' '.join(o.split('_'))
    if s.startswith('dr:'): s = '<'+s[3:]+'>'
    if o.startswith('dr:'): o = '<'+o[3:]+'>'
    if r.startswith('<http://rdf.freebase.com/ns/') and r.endswith('>'): r = '<'+r[len('<http://rdf.freebase.com/ns/'):-1]+'>'
    text = ' , '.join((s,r,o)) 
    graph = user_information.return_graph(s[1:-1], o[1:-1])
    passdetermine = str(user_information.no_right) + '/' + str(user_information.no_seen)
    return graph, option, inquiry_id, uid, title, text, s, o, passdetermine




import time, os, codecs

if __name__ == '__main__':
    u = user_info(100)
    u1 = user_info(101)
    f = u.files_doing
    f1 = u.files_doing
    no = 0
    set1 = set()
    while True:
        print(f, no)
        no+=1
        print(f1,no)
        set1.add(f)
        set1.add(f1)
        no+=1
        if no == 1252:
            break
        u.add_files_done(f)
        f, _ = u.files_to_do_next()
        u1.add_files_done(f1)
        f1, _ = u1.files_to_do_next()
    print(len(set1))
    exit()
        
    

