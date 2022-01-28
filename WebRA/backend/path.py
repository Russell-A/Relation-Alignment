import copy
from collections import Counter
import util as ut
import word2vec
import re
import numpy as np
import codecs
from math import log, exp, tanh
import multiprocessing as mp
from util import uid
import joblib

dict_degree = joblib.load('./degree/degree_dict')

class node():
    def __init__(self, name):
        self.name = name
        self.in_neighbour = set()  # 指向该节点的节点
        self.out_neighbour = set()
        self.in_edges = []
        self.out_edges = []
        self.degree_node = dict_degree[self.name]
        # self.degree = dict_degree[self.name]

    def add_in_edges(self, edge):
        self.in_edges.append((edge, edge.out_node))

    def add_out_edges(self, edge):
        self.out_edges.append((edge, edge.in_node))


    def get_in_edges(self):
        return self.in_edges

    def get_out_edges(self):
        return self.out_edges

    def degree(self):
        return self.degree_node


# class edge():
#     def __init__(self, name, node_in, node_out):
#         self.name = name
#         self.in_node = node_in
#         self.out_node = node_out

#
# class path():
#     def __init__(self, rdf_list, dict_node, dict_relation_sturcture, head, tail):
#         self.rdf = rdf_list
#         self.relations = []
#         self.nodes = []
#         self.type = 'path'
#         num_of_relation = 0
#         for item in self.rdf:
#             s, r, o = item.strip().split('\t')
#             num_of_relation += 1
#             self.relations.append((r, num_of_relation))
#             self.nodes.append(dict_node[s])
#             self.nodes.append(dict_node[o])
#         self.nodes = list(set(self.nodes))
#         self.relations = tuple(self.relations)
#         self.path_len = len(self.relations)
#         self.path_len = 1
#         if self.relations in dict_relation_sturcture:
#             dict_relation_sturcture[self.relations].append((rdf_list, (head, tail)))
#         else:
#             dict_relation_sturcture[self.relations] = [(rdf_list, (head, tail))]
#
class path():
    def __init__(self, rdf_list, dict_relation_sturcture, head, tail, dict_relation_node_degree, dict_node):
        self.rdf = rdf_list  # path 的 rdf
        self.relations = []  # rdf里所有的关系
        self.nodes = []
        self.type = 'path'
        num_of_relation = 0
        for item in self.rdf:
            s, r, o = item.strip().split('\t')
            num_of_relation += 1
            self.relations.append((r, num_of_relation))
            self.nodes.append(dict_node[s])
            self.nodes.append(dict_node[o])
        self.nodes = list(set(self.nodes))
        self.relations = tuple(self.relations)
        self.path_len = len(self.relations)  # 总的跳数
        self.path_num = 1  # 路径数
        if self.relations in dict_relation_sturcture:
            if not ((rdf_list, (head, tail)) in dict_relation_sturcture[self.relations]):
                dict_relation_sturcture[self.relations].append((rdf_list, (head, tail)))  # 存放关系对应的rdf
        else:
            dict_relation_sturcture[self.relations] = [(rdf_list, (head, tail))]
        for node in self.nodes:
            if self.relations in dict_relation_node_degree:
                dict_relation_node_degree[self.relations].append(node.degree())
            else:
                dict_relation_node_degree[self.relations] = [node.degree()]



# class multipath():
#     # 这里rdf需要已经去除重复项
#     def __init__(self, rdf1, rdf2, dict_node, dict_relation_structure, head, tail, dict_multipath_pathnum):
#         self.rdf = ut.merge_rdf(rdf1, rdf2)
#         self.path_len = len(self.rdf)
#         self.nodes = []
#         self.relations = []
#         self.type = 'multipath'
#         num = 0
#         for item in rdf1:
#             num += 1
#             s, r, o = item.strip().split('\t')
#             self.relations.append((r, num))
#             self.nodes.append(dict_node[s])
#             self.nodes.append(dict_node[o])
#         num = 0
#         for item in rdf2:
#             num += 1
#             if item in rdf1:
#                 continue
#             s, r, o = item.strip().split('\t')
#             self.relations.append((r, num))
#             self.nodes.append(dict_node[s])
#             self.nodes.append(dict_node[o])
#         self.nodes = list(set(self.nodes))
#         self.relations = tuple(self.relations)
#         self.path_len = len(self.relations)
#         if self.path_len == len(rdf1) + len(rdf2):
#             self.path_num = 2
#         else:
#             self.path_num = 1
#         dict_multipath_pathnum[(self.relations, self.type)] = self.path_num
#         if self.relations in dict_relation_structure:
#             dict_relation_structure[self.relations].append((self.rdf, (head, tail)))
#         else:
#             dict_relation_structure[self.relations] = [(self.rdf, (head, tail))]

class multipath():
    # 这里rdf需要已经去除重复项
    def __init__(self, rdf1, rdf2, dict_relation_structure, head, tail, dict_multipath_pathnum, dict_relation_node_degree, dict_node):
        self.rdf = ut.merge_rdf(rdf1, rdf2)
        self.path_len = len(self.rdf)
        self.nodes = []
        self.relations = []
        self.type = 'multipath'
        num = 0
        for item in rdf1:
            num += 1
            s, r, o = item.strip().split('\t')
            self.relations.append((r, num))
            self.nodes.append(dict_node(s))
            self.nodes.append(dict_node(o))
        num = 0
        for item in rdf2:
            num += 1
            if item in rdf1:
                continue
            s, r, o = item.strip().split('\t')
            self.relations.append((r, num))
            self.nodes.append(dict_node(s))
            self.nodes.append(dict_node(o))
        self.nodes = list(set(self.nodes))
        self.relations = tuple(self.relations)
        self.path_len = len(self.relations)
        if self.path_len == len(rdf1) + len(rdf2):
            self.path_num = 2
        else:
            self.path_num = 1
        dict_multipath_pathnum[(self.relations, self.type)] = self.path_num
        if self.relations in dict_relation_structure:
            if (self.rdf, (head, tail)) not in dict_relation_structure:
                dict_relation_structure[self.relations].append((self.rdf, (head, tail)))
        else:
            dict_relation_structure[self.relations] = [(self.rdf, (head, tail))]
        for node in self.nodes:
            if self.relations in dict_relation_node_degree:
                dict_relation_node_degree[self.relations].append(node.degree())
            else:
                dict_relation_node_degree[self.relations] = [node.degree()]

class structure():
    def __init__(self, rdf_list, start, end, fbline, dict_relation_sturcture, dict_relation_node_degree):
        self.fb_line = fbline  # freebase 三元组信息
        self.fb_relation = fbline.strip().split('\t')[1]  # freebase里的relation
        self.head = start  # string type
        self.tail = end
        self.node_name_set = []  # 结构里所有的节点名
        self.dict_node = dict()  # 键是name, 值是对应的节点
        self.dict_edge = dict()
        self.rdf_list = copy.deepcopy(rdf_list)  # 结构的整个rdf

        for lines in set(rdf_list):
            s, r, o = lines.strip().split('\t')
            # 新建两个节点和边
            if s not in self.node_name_set:
                node_s = node(s)
                # node_s = s
                self.node_name_set.append(s)
                self.dict_node[s] = node_s
            else:
                node_s = self.dict_node[s]
                pass
            if o not in self.node_name_set:
                node_o = node(o)
                # node_o = o
                self.node_name_set.append(o)
                self.dict_node[o] = node_o
            else:
                node_o = self.dict_node[o]
                pass
            node_s.out_neighbour.add(node_o)
            node_o.in_neighbour.add(node_s)
            # edge_r = edge(r, node_s, node_o)
            # self.dict_edge[r] = edge_r
            # node_s.add_out_edges(edge_r)
            # node_o.add_in_edges(edge_r)
        '''
        找到所有的path
        '''
        rdf_list = list(set(rdf_list))
        paths = []
        paths_incomplete = []
        for lines in rdf_list:
            if self.head in lines:
                s, r, o = lines.strip().split('\t')
                if s == self.head and o != self.tail:
                    paths_incomplete.append(([lines], o))
                elif s == self.head and o == self.tail:
                    paths.append((lines,))
                elif o == self.head and s != self.tail:
                    paths_incomplete.append(([lines], s))
                elif o == self.head and s == self.tail:
                    paths.append((lines,))
        paths_incomplete_new = []
        for lines in rdf_list:
            for incomplete_path, miss in paths_incomplete:
                if miss in lines:
                    s, r, o = lines.strip().split('\t')
                    if s == miss and o != self.tail:
                        paths_incomplete_new.append((incomplete_path + [lines], o))
                    elif s == miss and o == self.tail:
                        paths.append(tuple(incomplete_path + [lines]))
                    elif o == miss and s != self.tail:
                        paths_incomplete_new.append((incomplete_path + [lines], s))
                    elif o == miss and s == self.tail:
                        paths.append(tuple(incomplete_path + [lines]))
        paths_incomplete = paths_incomplete_new
        for lines in rdf_list:
            for incomplete_path, miss in paths_incomplete:
                if miss in lines:
                    s, r, o = lines.strip().split('\t')
                    if s == miss and o == self.tail:
                        paths.append(tuple(incomplete_path + [lines]))
                    elif o == miss and s == self.tail:
                        paths.append(tuple(incomplete_path + [lines]))

        self.paths = []  # 所有的path的集合
        for p in paths:
            path1 = path(p, dict_relation_sturcture, self.head, self.tail, dict_relation_node_degree, self.dict_node)
            self.paths.append(path1)
        list_relation = []
        for p1 in self.paths:
            list_relation.append(p1.relations)
        self.counter_relation = Counter(list_relation)  # structure里所有path的relation种类数目统计
        self.multipaths = []


class structure_list():  # 由文件构成的structure的集合
    def __init__(self, file):
        self.thereshold_point = None  # 最低分数
        num = 0
        rdf_list = []
        self.structures = []  # 存放所有structure
        self.dict_relation_structure = {}  # 通过relation找rdf的字典
        self.wait_for_complete = []  # 存放语义不足的relation
        self.dict_relation_full = {}
        self.dict_relation_node_degree = {}
        for lines in codecs.open(file, 'r', 'utf-8'):
            num += 1
            if lines == '\n':
                num = 0
                import time
                time1 = time.time()
                s = structure(rdf_list, start, end, fb_lines, self.dict_relation_structure, self.dict_relation_node_degree)
                print('构建structure时间',time.time()-time1)
                self.structures.append(s)
                rdf_list = []
            else:
                if num == 1:
                    fb_lines = lines
                elif num == 2:
                    start, end = lines.strip().split('\t')
                    start, end = ut.dr(start), ut.dr(end)
                else:
                    s, r, o = lines.strip().split('\t')
                    r_full = r
                    r = r.strip().split('/')[-1][:-1]
                    lines = '\t'.join((s, r, o))
                    rdf_list.append(lines)
                    if r in self.dict_relation_full and r_full not in self.dict_relation_full[r]:
                        self.dict_relation_full[r].append(r_full)
                    else:
                        self.dict_relation_full[r] = [r_full]
        '''
        构建structure的        
        '''
        self.get_path_relation_set()
        self.get_fb_relation()
        self.get_dict_linguistic_similarity()
        self.candidate = []  # 所有的候选 存放路径的relation
        self.already_seen = []  # 存放判断过的候选
        self.multipaths = []
        self.dict_multipath_path_num = {}
        self.written = []  # 存放写入的候选
        self.structure_point = {}

    def get_path_relation_set(self):  # 找到structure里的所有的relation组合
        relation = []
        for s1 in self.structures:
            for rdf_path in s1.paths:
                relation.append((rdf_path.relations, rdf_path.type))
        self.Counter_relation = Counter(relation)
        self.path_realtion_set = set(relation)

    def get_fb_relation(self):
        fbline = self.structures[0].fb_line
        self.fb_relation = fbline.strip().split('\t')[1]

    def get_dict_linguistic_similarity(self):  # 把relation对应的语言分存到字典里
        self.dict_linguistic_similarity = {}
        fb_relation = self.fb_relation.split('/')[-1][:-1]
        fb_relation = re.split('[\.|\_]', fb_relation)
        for item in self.path_realtion_set:
            db_relation = []
            for i in item[0]:
                i = i[0]
                print(i)
                db_word = re.split('/', i)[-1][:]
                print(db_word)
                db_word = re.findall('[a-zA-Z][^A-Z]*', db_word)
                print(db_word)
                db_relation.extend(db_word)
                print(fb_relation, db_relation)
            self.dict_linguistic_similarity[item] = word2vec.get_cos(fb_relation, db_relation)

    def shuffle(self):  # 加权平均语言分和结构分，统计每一个候选candidate_relation的分数并排序
        # if ut.exist('linguistic_weight', uid):
        #     linguistic_weight = ut.load('linguistic_weight', uid)
        # else:
        #     linguistic_weight = 0.5
        #     ut.dump(linguistic_weight, 'linguistic_weight', uid)
        # structure_weight = 1 - linguistic_weight
        positive_num = len(self.written)
        all_num = len(self.already_seen)
        negative_num = all_num - positive_num
        if all_num != 0:
            linguistic_weight = (tanh(2 * (positive_num - negative_num) / all_num) + 1) / 2  # 权数
        else:
            linguistic_weight = 0.5
        structure_weight = 1 - linguistic_weight

        # print('self.path_relation_set', self.path_realtion_set)
        for path_relation in self.path_realtion_set:
            point = structure_weight * self.get_stucture_points(path_relation) + \
                    linguistic_weight * self.get_linguistic_points(path_relation)
            self.candidate.append((path_relation, point))
        self.candidate.sort(key=lambda x: x[1], reverse=True)
        top125 = len(self.candidate) // 8
        ''' 确定thereshold的分数'''
        if self.thereshold_point is None:
            self.thereshold_point = self.candidate[top125][1]
        # self.thereshold_point = self.candidate[0][1]

    def get_stucture_points(self, path_relation):  # 标准化structures的分数
        for key in self.Counter_relation.keys():
            # print(key)
            # print(len(key[0]))
            # print(np.mean(self.dict_relation_node_degree[key[0]]))
            # print(self.dict_relation_node_degree)
            # print((len(key[0])*(log(len(key[0]))+1)))
            self.structure_point[key] = self.Counter_relation[key] /(len(key[0])*(log(len(key[0]))+1)) / (log(np.mean(self.dict_relation_node_degree[key[0]]))+1)
        mean_of_path_structure = np.mean(list(self.structure_point.values()))
        var_of_path_structure = np.var(list(self.structure_point.values()))
        if var_of_path_structure == 0:
            var_of_path_structure = 1
        structure_point = (self.structure_point[path_relation] - mean_of_path_structure) / var_of_path_structure
        return structure_point

    def get_linguistic_points(self, path_relation):  # 标准化语言分
        mean_of_path_linguistic = np.mean(list(self.dict_linguistic_similarity.values()))
        var_of_path_linguistic = np.var(list(self.dict_linguistic_similarity.values()))
        if var_of_path_linguistic == 0:
            var_of_path_linguistic = 1
        linguistic_point = (self.dict_linguistic_similarity[
                                path_relation] - mean_of_path_linguistic) / var_of_path_linguistic
        return linguistic_point

    # def show_picture_and_fb_line(self):
    #     next_file = False
    #     for i in range(len(self.candidate)):
    #         if self.candidate[i][0] in self.already_seen:
    #             print('seen')
    #             continue
    #         if self.candidate[i][1] >= self.thereshold_point:
    #             candiate_path = self.candidate[i][0][0]
    #             candiate_path_type = self.candidate[i][0][1]
    #             break
    #     else:
    #         next_file = True
    #         return (None, None, None, None, next_file)
    #     if candiate_path_type == 'path':
    #         for s in self.structures:
    #             for path in s.paths:
    #                 if path.relations == candiate_path:
    #                     ut.nx_construct_path_from_rdf(path.rdf)
    #                     rdf_new = path.rdf
    #                     fb_line = s.fb_line
    #                     print('\t'.join((s.head, s.fb_relation, s.tail)))
    #                     i = input('需要更多的图吗？(Y|N)')
    #                     # i = 'n'
    #                     if i == 'N' or i == 'n':
    #                         break
    #             else:
    #                 continue
    #             break
    #         else:
    #             print('没有图了')
    #     elif candiate_path_type == 'multipath':
    #         for s in self.structures:
    #             for path in s.multipaths:
    #                 if path.relations == candiate_path:
    #                     ut.nx_construct_path_from_rdf(path.rdf)
    #                     rdf_new = path.rdf
    #                     fb_line = s.fb_line
    #                     print('\t'.join((s.head, s.fb_relation, s.tail)))
    #                     # i = input('需要更多的图吗？(Y|N)')
    #                     i = 'n'
    #                     if i == 'N' or i == 'n':
    #                         break
    #             else:
    #                 continue
    #             break
    #         else:
    #             print('没有图了')
    #     return (candiate_path, candiate_path_type, rdf_new, fb_line, next_file)

    def check_next_file(self):  # 判断是否进入下一个文件
        next_file = False
        for i in range(len(self.candidate)):
            if self.candidate[i][0] in self.already_seen:
                continue
            if self.candidate[i][1] >= self.thereshold_point:
                # candiate_path = self.candidate[i][0][0]
                # candiate_path_type = self.candidate[i][0][1]
                break
        else:
            next_file = True
        return next_file

    def return_candidate_rdf(self):  # 返回第一个未见的候选的relation，candidte_type， rdf
        """

        :rtype: object
        """
        if self.check_next_file() == True:
            raise ('应该下一个文件了')
        for i in range(len(self.candidate)):
            if self.candidate[i][0] in self.already_seen:
                continue
            if self.candidate[i][1] >= self.thereshold_point:
                candiate_path_relations = self.candidate[i][0][0]
                candiate_path_type = self.candidate[i][0][1]
                break
        rdfs_list = []
        num = 0
        if candiate_path_type == 'path':
            for s in self.structures:
                for path in s.paths:
                    if path.relations == candiate_path_relations:
                        db_line = ' '.join((s.head, s.fb_relation, s.tail))
                        rdfs_list.append((path.rdf, db_line, s.fb_line))
                        num += 1
                        if num == 8:
                            break
                else:
                    continue
                break
        elif candiate_path_type == 'multipath':
            for s in self.structures:
                for path in s.multipaths:
                    if path.relations == candiate_path_relations:
                        db_line = ' '.join((s.head, s.fb_relation, s.tail))
                        rdfs_list.append((path.next_rdf, db_line, s.fb_line))
                        num += 1
                        if num == 8:
                            break
                else:
                    continue
                break
        return candiate_path_relations, candiate_path_type, rdfs_list

    # def delete_unrelated_relations(self, candidate_path, num): # 删除无关关系 Todo：bug
    #     delete_relation = candidate_path[num - 1]
    #     structures = self.structures
    #     candidate = []
    #     for item in self.candidate:
    #         if item[0][-1] == 'multipath':
    #             candidate.append(item)
    #     self.candidate = candidate
    #     self.structures = []
    #     for s in structures:
    #         rdf_list_new = []
    #         for lines in s.rdf_list:
    #             if delete_relation not in lines:
    #                 rdf_list_new.append(lines)
    #         self.structures.append(structure(rdf_list_new, s.head, s.tail, s.fb_line, self.dict_relation_structure))
    #     self.get_path_relation_set()
    #     self.shuffle()

    def delete_unrelated_relations(self, candidate_path, num):  # 处理好path_relation set 和 counter dict 就好 todo:没测试过
        delete_relation = candidate_path[num - 1][0]
        print(delete_relation)
        path_realtion_set_new = set()
        for item in self.path_realtion_set:
            for relation, no in item[0]:
                if relation == delete_relation:
                    # print(item[0])
                    self.dict_relation_structure.pop(item[0])
                    break
            else:
                path_realtion_set_new.add(item)
        self.path_realtion_set = path_realtion_set_new

    def add_seen(self, candidate_path, type):  # 加入用户见过
        self.already_seen.append((candidate_path, type))

    def add_multipath(self, relations):  # relations存在不足
        self.wait_for_complete.append(relations)
        list_temp = []
        for i in range(len(self.wait_for_complete)):
            for rdf1, tuple_head_tail1 in self.dict_relation_structure[self.wait_for_complete[i]]:
                for rdf2, tuple_head_tail2 in self.dict_relation_structure[relations]:
                    if tuple_head_tail1 == tuple_head_tail2 and rdf2 != rdf1:
                        for s in self.structures:
                            if s.head == tuple_head_tail1[0] and s.tail == tuple_head_tail1[1]:
                                m = multipath(rdf1, rdf2, self.dict_relation_structure, s.head, s.tail,
                                              self.dict_multipath_path_num, dict_relation_node_degree=self.dict_relation_node_degree, dict_node=s.dict_node)
                                s.multipaths.append(m)
                                list_temp.append((m.relations, m.type))
                                break
        c = Counter(list_temp)
        # for key in c.keys():
        #     if key[1] == 'multipath':
        #         path_num = self.dict_multipath_path_num[key]
        #         c[key] = c[key] ** (1 / path_num)
                # c[key] = c[key]**10
        self.Counter_relation.update(c)
        self.path_realtion_set.update(set(list_temp))
        # linguistic_similarity
        fb_relation = self.fb_relation.split('/')[-1][:-1]
        fb_relation = re.split('[\.|\_]', fb_relation)
        for item in set(list_temp):
            if item not in self.dict_linguistic_similarity:
                db_relation = []
                for i in item[0]:
                    i = i[0]
                    db_word = re.split('/', i)[-1][:-1]
                    db_word = re.findall('[a-zA-Z][^A-Z]*', db_word)
                    db_relation.extend(db_word)
                self.dict_linguistic_similarity[item] = word2vec.get_cos(fb_relation, db_relation)
        print('add multipath done')

    # def raise_linguistic_weight(self):
    #     linguistic_weight = ut.load('linguistic_weight', uid)
    #     t = -log(1 / linguistic_weight - 1)
    #     t += 0.5
    #     linguistic_weight = 1 / (1 + exp(-t))
    #     ut.dump(linguistic_weight, 'linguistic_weight', uid)
    #     return 0
    #
    # def decrease_linguistic_weight(self):
    #     linguistic_weight = ut.load('linguistic_weight', uid)
    #     t = -log(1 / linguistic_weight - 1)
    #     t -= 0.5
    #     linguistic_weight = 1 / (1 + exp(-t))
    #     ut.dump(linguistic_weight, 'linguistic_weight', uid)
    #     return 0

    def add_written(self, candidate_path, type):  # 加到写入的文件
        self.written.append((candidate_path, type))

import time
if __name__ == '__main__':
    time1 = time.time()
    s = structure_list('candidate_for_experiment/candidate_full_pathpeople.person.profession.txt')
    time2 = time.time()
    print(time2-time1)




