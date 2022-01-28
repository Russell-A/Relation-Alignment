import copy
from collections import Counter

import util as ut
import word2vec
import re
import numpy as np
import codecs
from math import log, exp, tanh, sqrt
from util import uid
import joblib

print('load digree dict')
degree_dict = joblib.load('./degree/degree_dict')
print('load digree dict done')


class node():
    def __init__(self, name):
        self.name = name
        self.in_neighbour = set()  # 指向该节点的节点
        self.out_neighbour = set()
        self.edges = dict()
        self.degree_node = degree_dict[self.name]
        self.edge_dict = dict() # key = edge(r) value = endpoint


    def add_edges(self, nodein, edge):
        if nodein in self.edges:
            self.edges[nodein].append(edge)
        else:
            self.edges[nodein] = [edge]
        s, r, o = edge
        if r in self.edge_dict:
            self.edge_dict[r].append(nodein)
        else:
            self.edge_dict[r] = [nodein]


    def all_neighbour(self):
        return self.in_neighbour | self.out_neighbour

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
# class path():
#     def __init__(self, rdf_list, dict_relation_sturcture, head, tail, dict_relation_node_degree, dict_node):
#         self.rdf = rdf_list  # path 的 rdf
#         self.relations = []  # rdf里所有的关系
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
#         self.path_len = len(self.relations)  # 总的跳数
#         self.path_num = 1  # 路径数
#         if self.relations in dict_relation_sturcture:
#             if not ((rdf_list, (head, tail)) in dict_relation_sturcture[self.relations]):
#                 dict_relation_sturcture[self.relations].append((rdf_list, (head, tail)))  # 存放关系对应的rdf
#         else:
#             dict_relation_sturcture[self.relations] = [(rdf_list, (head, tail))]
#         for node in self.nodes:
#             if self.relations in dict_relation_node_degree:
#                 dict_relation_node_degree[self.relations].append(node.degree())
#             else:
#                 dict_relation_node_degree[self.relations] = [node.degree()]



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

# class multipath():
#     # 这里rdf需要已经去除重复项
#     def __init__(self, rdf1, rdf2, dict_relation_structure, head, tail, dict_multipath_pathnum, dict_relation_node_degree, dict_node):
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
#             self.nodes.append(dict_node(s))
#             self.nodes.append(dict_node(o))
#         num = 0
#         for item in rdf2:
#             num += 1
#             if item in rdf1:
#                 continue
#             s, r, o = item.strip().split('\t')
#             self.relations.append((r, num))
#             self.nodes.append(dict_node(s))
#             self.nodes.append(dict_node(o))
#         self.nodes = list(set(self.nodes))
#         self.relations = tuple(self.relations)
#         self.path_len = len(self.relations)
#         if self.path_len == len(rdf1) + len(rdf2):
#             self.path_num = 2
#         else:
#             self.path_num = 1
#         dict_multipath_pathnum[(self.relations, self.type)] = self.path_num
#         if self.relations in dict_relation_structure:
#             if (self.rdf, (head, tail)) not in dict_relation_structure:
#                 dict_relation_structure[self.relations].append((self.rdf, (head, tail)))
#         else:
#             dict_relation_structure[self.relations] = [(self.rdf, (head, tail))]
#         for node in self.nodes:
#             if self.relations in dict_relation_node_degree:
#                 dict_relation_node_degree[self.relations].append(node.degree())
#             else:
#                 dict_relation_node_degree[self.relations] = [node.degree()]

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
        self.multipaths = []
        self.dict_relation_structure = dict()
        self.dict_relation_degree = dict()

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

            node_s.add_edges(node_o, (s,r,o))
            node_o.add_edges(node_s, (s,r,o))
        '''
        找到所有的path
        '''
        f = 'forward'
        b = 'backward'
        self.paths = []
        node_start = self.dict_node[self.head]
        node_start : node
        for node1 in node_start.all_neighbour():
            node1: node
            if node1.name == self.tail:
                for s,edge,o in node1.edges[node_start]:
                    if s == node_start.name:
                        relation = ((edge, f ,1, 'n1'),)
                    elif s == node1.name:
                        relation = ((edge, b ,1, 'n1'),)
                    self.paths.append(relation)
                    line = '\t'.join((s, edge, o))
                    if relation in dict_relation_sturcture:
                        dict_relation_sturcture[relation].append(((line,), (self.head, self.tail)))
                    else:
                        dict_relation_sturcture[relation] = [((line,), (self.head, self.tail))]
                    if relation in self.dict_relation_structure:
                        self.dict_relation_structure[relation].append((line,))
                    else:
                        self.dict_relation_structure[relation] = [(line,)]                    
                    if relation in dict_relation_node_degree:
                        dict_relation_node_degree[relation].append(node_start.degree())
                        dict_relation_node_degree[relation].append(node1.degree())
                    else:
                        dict_relation_node_degree[relation] = [node_start.degree(), node1.degree()]
                    if relation in self.dict_relation_degree:
                        self.dict_relation_degree[relation].extend([node_start.degree(), node1.degree()])
                    else:
                        self.dict_relation_degree[relation] = [node_start.degree(), node1.degree()]
            else:
                for node2 in node1.all_neighbour():
                    node2 : node
                    if node2.name == self.tail:
                        for s1,edge1,o1 in node1.edges[node_start]:
                            for s2,edge2,o2 in node2.edges[node1]:
                                if (s1, o1) == (s2, o2) or (s1,o1) == (o1,s1):
                                    break
                                if s1 == node_start.name:
                                    relation1 = ((edge1, f ,1,'n1'),)
                                elif s1 == node1.name:
                                    relation1 = ((edge1, b ,1,'n1'),)
                                if s2 == node1.name:
                                    relation2 = ((edge2, f ,2,'n2'),)
                                elif s2 == node2.name:
                                    relation2 = ((edge2, b ,2,'n2'),)
                                relation = relation1 + relation2
                                self.paths.append(relation)
                                line1 = '\t'.join((s1,edge1,o1))
                                line2 = '\t'.join((s2,edge2,o2))
                                if relation in dict_relation_sturcture:
                                    dict_relation_sturcture[relation].append(((line1, line2), (self.head, self.tail)))
                                else:
                                    dict_relation_sturcture[relation] = [((line1, line2), (self.head, self.tail))]
                                if relation in self.dict_relation_structure:
                                    self.dict_relation_structure[relation].append((line1, line2))
                                else:
                                    self.dict_relation_structure[relation] = [(line1, line2)]
                                if relation in dict_relation_node_degree:
                                    dict_relation_node_degree[relation].append(node_start.degree())
                                    dict_relation_node_degree[relation].append(node1.degree())
                                    dict_relation_node_degree[relation].append(node2.degree())
                                else:
                                    dict_relation_node_degree[relation] = [node_start.degree(), node1.degree(), node2.degree()]
                                if relation in self.dict_relation_degree:
                                    self.dict_relation_degree[relation].extend([node_start.degree(), node1.degree(), node2.degree()])
                                else:
                                    self.dict_relation_degree[relation] = [node_start.degree(), node1.degree(), node2.degree()]
                    else:
                        for node3 in node2.all_neighbour():
                            node3:node
                            if node3.name == self.tail:
                                for s1,edge1,o1 in node1.edges[node_start]:
                                    for s2,edge2,o2 in node2.edges[node1]:
                                        for s3,edge3,o3 in node3.edges[node2]:
                                            if (s1,o1) == (s2,o2) or (s1,o1) == (s3,o3) or (s2,o2) == (s3,o3) \
                                            or (s1,o1) == (o2,s2) or (s1,o1) == (o3,s3) or (s2,o2) == (o3,s3):
                                                break
                                            if s1 == node_start.name:
                                                relation1 = ((edge1, f ,1, 'n1'),)
                                            elif s1 == node1.name:
                                                relation1 = ((edge1, b ,1, 'n1'),)
                                            if s2 == node1.name:
                                                relation2 = ((edge2, f ,2, 'n2'),)
                                            elif s2 == node2.name:
                                                relation2 = ((edge2, b ,2, 'n2'),)
                                            if s3 == node2.name:
                                                relation3 = ((edge3, f ,3, 'n3'),)
                                            elif s3 == node3.name:
                                                relation3 = ((edge3, b, 3, 'n3'),)
                                            relation = relation1+relation2+relation3
                                            self.paths.append(relation)
                                            line1 = '\t'.join((s1,edge1,o1))
                                            line2 = '\t'.join((s2,edge2,o2))
                                            line3 = '\t'.join((s3,edge3,o3))
                                            if relation in dict_relation_sturcture:
                                                dict_relation_sturcture[relation].append(((line1, line2,line3), (self.head, self.tail)))
                                            else:
                                                dict_relation_sturcture[relation] = [((line1, line2, line3), (self.head, self.tail))]
                                            if relation in self.dict_relation_structure:
                                                self.dict_relation_structure[relation].append((line1, line2,line3))
                                            else:
                                                self.dict_relation_structure[relation] = [(line1, line2, line3)]
                                            if relation in dict_relation_node_degree:
                                                dict_relation_node_degree[relation].append(node_start.degree())
                                                dict_relation_node_degree[relation].append(node1.degree())
                                                dict_relation_node_degree[relation].append(node2.degree())
                                                dict_relation_node_degree[relation].append(node3.degree())
                                            else:
                                                dict_relation_node_degree[relation] = [node_start.degree(), node1.degree(), node2.degree(),node3.degree()]
                                            if relation in self.dict_relation_degree:
                                                self.dict_relation_degree[relation].extend([node_start.degree(), node1.degree(), node2.degree(), node3.degree()])
                                            else:
                                                self.dict_relation_degree[relation] = [node_start.degree(), node1.degree(), node2.degree(), node3.degree()]

        self.counter_relation = Counter(self.paths)
        self.relation_path_set = set(self.paths)
        

        # paths = []
        # paths_incomplete = []
        # rdf_list_new = []
        # for lines in rdf_list:
        #     if self.head in lines:
        #         s, r, o = lines.strip().split('\t')
        #         if s == self.head and o != self.tail:
        #             paths_incomplete.append(([lines], o))
        #         elif s == self.head and o == self.tail:
        #             paths.append((lines,))
        #         elif o == self.head and s != self.tail:
        #             paths_incomplete.append(([lines], s))
        #         elif o == self.head and s == self.tail:
        #             paths.append((lines,))
        #         else:
        #             rdf_list_new.append(lines)
        #     else:
        #         rdf_list_new.append(lines)
        # rdf_list = rdf_list_new
        # rdf_list_new = []
        # paths_incomplete_new = []
        # for lines in rdf_list:
        #     add = True
        #     for incomplete_path, miss in paths_incomplete:
        #         if miss in lines:
        #             s, r, o = lines.strip().split('\t')
        #             if s == miss and o != self.tail:
        #                 add = False
        #                 paths_incomplete_new.append((incomplete_path + [lines], o))
        #             elif s == miss and o == self.tail:
        #                 add = False
        #                 paths.append(tuple(incomplete_path + [lines]))
        #             elif o == miss and s != self.tail:
        #                 add = False
        #                 paths_incomplete_new.append((incomplete_path + [lines], s))
        #             elif o == miss and s == self.tail:
        #                 add = False
        #                 paths.append(tuple(incomplete_path + [lines]))
        #     if add == True:
        #         rdf_list_new.append(lines)
        # rdf_list = rdf_list_new
        # paths_incomplete = paths_incomplete_new
        # for lines in rdf_list:
        #     for incomplete_path, miss in paths_incomplete:
        #         if miss in lines:
        #             s, r, o = lines.strip().split('\t')
        #             if s == miss and o == self.tail:
        #                 paths.append(tuple(incomplete_path + [lines]))
        #             elif o == miss and s == self.tail:
        #                 paths.append(tuple(incomplete_path + [lines]))

        # self.paths = []  # 所有的path的集合
        # for p in paths:
        #     path1 = path(p, dict_relation_sturcture, self.head, self.tail, dict_relation_node_degree, self.dict_node)
        #     self.paths.append(path1)
        # list_relation = []
        # for p1 in self.paths:
        #     list_relation.append(p1.relations)
        # self.counter_relation = Counter(list_relation)  # structure里所有path的relation种类数目统计
        # self.multipaths = []


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
        self.pass_linguistic = 0
        for lines in codecs.open(file, 'r', 'utf-8'):
            num += 1
            if lines == '\n':
                num = 0
                import time
                time5 = time.time()
                s = structure(rdf_list, start, end, fb_lines, self.dict_relation_structure, self.dict_relation_node_degree)
                # print('构建structure时间',time.time()-time5)
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
                    r = r.strip().split('/')[-1]
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
        paths = []
        for s in self.structures:
            for relation in s.paths:
                paths.append((relation,'path'))
        self.Counter_relation = Counter(paths)
        self.path_realtion_set = set(paths)

    def get_fb_relation(self):
        fbline = self.structures[0].fb_line
        self.fb_relation = fbline.strip().split('\t')[1]

    def get_dict_linguistic_similarity(self):  # 把relation对应的语言分存到字典里
        self.dict_linguistic_similarity = {}
        # print(self.fb_relation)
        fb_relation = self.fb_relation.split('/')[-1]
        fb_relation = re.split('[\.|\_]', fb_relation)
        for item in self.path_realtion_set:
            print(item)
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
        positive_num = self.pass_linguistic
        all_num = len(self.already_seen)
        negative_num = all_num - positive_num
        if all_num != 0:
            linguistic_weight = (tanh(2 * (positive_num - negative_num) / all_num) + 1) / 2  # 权数
        else:
            linguistic_weight = 0.5
        structure_weight = 1 - linguistic_weight

        for key in self.Counter_relation.keys():
            # print(key)
            # print(len(key[0]))
            # print(np.mean(self.dict_relation_node_degree[key[0]]))
            # print(self.dict_relation_node_degree)
            # print((len(key[0])*(log(len(key[0]))+1)))
            # print(np.mean(self.dict_relation_node_degree[key[0]]))
            self.structure_point[key] = self.Counter_relation[key] /(len(key[0])*(log(len(key[0]))+1)) / (log(np.mean(self.dict_relation_node_degree[key[0]]))+1)

        # print('self.path_relation_set', self.path_realtion_set)
        self.candidate = []
        for path_relation in self.path_realtion_set:
            point = structure_weight * self.get_stucture_points(path_relation) + \
                    linguistic_weight * self.get_linguistic_points(path_relation)
            # print(self.get_stucture_points(path_relation), self.get_linguistic_points(path_relation))
            self.candidate.append((path_relation, point))
        self.candidate.sort(key=lambda x: x[1], reverse=True)
        top125 = len(self.candidate) // 5
        ''' 确定thereshold的分数'''
        if self.thereshold_point is None:
            self.thereshold_point = self.candidate[top125][1]
        # self.thereshold_point = self.candidate[0][1]

    def get_stucture_points(self, path_relation):  # 标准化structures的分数
        
        mean_of_path_structure = np.mean(list(self.structure_point.values()))
        var_of_path_structure = np.var(list(self.structure_point.values()))
        if var_of_path_structure == 0:
            var_of_path_structure = 1
        structure_point = (self.structure_point[path_relation] - mean_of_path_structure) / sqrt(var_of_path_structure)
        return structure_point

    def get_linguistic_points(self, path_relation):  # 标准化语言分
        mean_of_path_linguistic = np.mean(list(self.dict_linguistic_similarity.values()))
        var_of_path_linguistic = np.var(list(self.dict_linguistic_similarity.values()))
        # max_of_path_linguistic = max(list(self.dict_linguistic_similarity.values()))
        # min_of_path_linguistic = min(list(self.dict_linguistic_similarity.values()))
        if var_of_path_linguistic == 0:
            var_of_path_linguistic = 1
        linguistic_point = (self.dict_linguistic_similarity[
                                path_relation] - mean_of_path_linguistic) / sqrt(var_of_path_linguistic)
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
        # if candiate_path_type == 'path':
        for s in self.structures:
            s:structure
            if candiate_path_relations in s.dict_relation_structure:
                for rdf in s.dict_relation_structure[candiate_path_relations]:
                    # print(rdf)
                    db_line = ' '.join((s.head, s.fb_relation, s.tail))
                    rdfs_list.append((rdf, db_line, s.fb_line))
                    num += 1
                    if num == 8:
                        break
                else:
                    continue
                break
        # elif candiate_path_type == 'multipath':
        #     for s in self.structures:
        #         for path in s.multipaths:
        #             if path.relations == candiate_path_relations:
        #                 db_line = ' '.join((s.head, s.fb_relation, s.tail))
        #                 rdfs_list.append((path.next_rdf, db_line, s.fb_line))
        #                 num += 1
        #                 if num == 8:
        #                     break
        #         else:
        #             continue
        #         break
        if rdfs_list == []:
            raise('rdfs_list is empty')
        # print(len(rdfs_list))
        degree = np.mean(self.dict_relation_node_degree[candiate_path_relations])
        return candiate_path_relations, candiate_path_type, rdfs_list, degree

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
        # print(delete_relation)
        path_realtion_set_new = set()
        for item in self.path_realtion_set:
            for relation, forback ,no, nodenno in item[0]:
                if relation == delete_relation:
                    # print(relation)
                    if item[0] in self.dict_relation_structure:
                    # print(item[0])
                        self.dict_relation_structure.pop(item[0])
                    break
            else:
                path_realtion_set_new.add(item)
        self.path_realtion_set = path_realtion_set_new
        self.shuffle()
        print('delete_relation', delete_relation)
        # print(self.path_realtion_set)
        # print(self.candidate)

    def add_seen(self, candidate_path, type):  # 加入用户见过
        self.already_seen.append((candidate_path, type))
    
    def add_multipath(self, relations):
        self.wait_for_complete.append(relations)
        for s in self.structures:
            if not (relations in s.relation_path_set):
                continue
            else:
                s: structure
                for i in range(len(self.wait_for_complete)):
                    relation1 = relations
                    relation2 = self.wait_for_complete[i]
                    if not (relation2 in s.relation_path_set):
                        continue
                    if relation1 != relation2:
                        for rdf1 in s.dict_relation_structure[relation1]:
                            for rdf2 in s.dict_relation_structure[relation2]:
                                node_now = s.head
                                no = 0
                                dict_nodename_no = {}
                                degree_list = [degree_dict[s.head]]
                                for line in rdf1:
                                    s1, r, o1 = line.strip().split('\t')
                                    if s1 == node_now:
                                        node_now = o1
                                    else: 
                                        node_now = s1
                                    no += 1
                                    dict_nodename_no[node_now] = 'n'+str(no)
                                    degree_list.append(degree_dict[node_now])
                                node_now = s.head
                                relation2_new = []
                                for i in range(len(rdf2)):
                                    line = rdf2[i]
                                    s1, r, o1 = line.strip().split('\t')
                                    if s1 == node_now:
                                        node_now = o1 
                                    else:
                                        node_now = s1
                                    if node_now in dict_nodename_no:
                                        relation2_new.append(relation2[i][0:3]+(dict_nodename_no[node_now],))
                                    else:
                                        relation2_new.append(relation2[i][0:3]+('n'+str(no),))
                                        degree_list.append(degree_dict[node_now])
                                relation2_new = tuple(relation2_new)
                                new_relation = tuple()
                                for item in relation1 +relation2_new:
                                    if not (item in new_relation):
                                        new_relation = new_relation + (item,)
                                print(new_relation)
                                self.path_realtion_set.add((new_relation,'multipath'))
                                s.relation_path_set.add(new_relation)
                                if new_relation in s.counter_relation:
                                    s.counter_relation[new_relation] += 1
                                else:
                                    s.counter_relation[new_relation] = 1
                                if new_relation in s.dict_relation_structure:
                                    s.dict_relation_structure[new_relation].append(rdf1+rdf2)
                                else:
                                    s.dict_relation_structure[new_relation] = [rdf1+rdf2]

                                if new_relation in self.dict_relation_structure:
                                    self.dict_relation_structure[new_relation].append(rdf1+rdf2)
                                else:
                                    self.dict_relation_structure = [rdf1+rdf2]
                                if not (new_relation in self.dict_relation_node_degree):
                                     self.dict_relation_node_degree[new_relation] = \
                                        degree_list
                                else:
                                    self.dict_relation_node_degree[new_relation] += \
                                        degree_list
                                if not ( (new_relation,'multipath') in self.Counter_relation):
                                    self.Counter_relation[(new_relation,'multipath')] = 1
                                else:
                                    self.Counter_relation[(new_relation,'multipath')] += 1
                                # linguistic_similarity
                                fb_relation = self.fb_relation.split('/')[-1]
                                fb_relation = re.split('[\.|\_]', fb_relation)
                
                                if new_relation not in self.dict_linguistic_similarity:
                                    db_relation = []
                                    for i in new_relation:
                                        i = i[0]
                                        db_word = re.split('/', i)[-1][:-1]
                                        db_word = re.findall('[a-zA-Z][^A-Z]*', db_word)
                                        db_relation.extend(db_word)
                                    self.dict_linguistic_similarity[(new_relation,'multipath')] = word2vec.get_cos(fb_relation, db_relation)
                                # print(new_relation)
                            else:
                                continue
        print('add multipath done')







    # def add_multipath(self, relations):  # relations存在不足
    #     self.wait_for_complete.append(relations)
    #     list_temp = []
    #     for i in range(len(self.wait_for_complete)):
    #         for rdf1, tuple_head_tail1 in self.dict_relation_structure[self.wait_for_complete[i]]:
    #             for rdf2, tuple_head_tail2 in self.dict_relation_structure[relations]:
    #                 if tuple_head_tail1 == tuple_head_tail2 and rdf2 != rdf1:
    #                     for s in self.structures:
    #                         if s.head == tuple_head_tail1[0] and s.tail == tuple_head_tail1[1]:
    #                             m = multipath(rdf1, rdf2, self.dict_relation_structure, s.head, s.tail,
    #                                           self.dict_multipath_path_num, dict_relation_node_degree=self.dict_relation_node_degree, dict_node=s.dict_node)
    #                             s.multipaths.append(m)
    #                             list_temp.append((m.relations, m.type))
    #                             break
    #     c = Counter(list_temp)
    #     # for key in c.keys():
    #     #     if key[1] == 'multipath':
    #     #         path_num = self.dict_multipath_path_num[key]
    #     #         c[key] = c[key] ** (1 / path_num)
    #             # c[key] = c[key]**10
    #     self.Counter_relation.update(c)
    #     self.path_realtion_set.update(set(list_temp))
    #     # linguistic_similarity
    #     fb_relation = self.fb_relation.split('/')[-1][:-1]
    #     fb_relation = re.split('[\.|\_]', fb_relation)
    #     for item in set(list_temp):
    #         if item not in self.dict_linguistic_similarity:
    #             db_relation = []
    #             for i in item[0]:
    #                 i = i[0]
    #                 db_word = re.split('/', i)[-1][:-1]
    #                 db_word = re.findall('[a-zA-Z][^A-Z]*', db_word)
    #                 db_relation.extend(db_word)
    #             self.dict_linguistic_similarity[item] = word2vec.get_cos(fb_relation, db_relation)
    #     print('add multipath done')

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


import os, time
if __name__ == '__main__':
    time1 = time.time()
    s = structure_list('candidate_for_experiment/candidate_full_pathpeople.person.profession.txt')
    time2 = time.time()
    print(time2-time1)

        
    # print(s.candidate)
    # print(s.Counter_relation)

 



    # print(time.time()-time1)
   

