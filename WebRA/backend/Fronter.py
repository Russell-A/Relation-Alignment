import time
import os

import math
import path
import joblib
import codecs
from random import shuffle
import util as ut
import multiprocessing
from util import uid

'''这个好像暂时用不到
'''

# get_uid是给不同的用户记录不同的用户名
def get_uid():
    global uid
    uid = input('请输入你的用户名的小写拼音\n').strip()
    return uid

def dr(string):
    if string.startswith("<http://dbpedia.org/resource/") and string.endswith('>'):
        return "dr:" + string[29:-1]
    else:
        return string


# 用来在用户登录时返回做哪一个文件的函数
def file_to_do():
    # files 是所有的要做的文件的double——list
    if ut.exist('files', uid):
        files = ut.load('files',uid)
    else:
        files = joblib.load('/root/dxh/ccw_workplace/cluster_result')
        shuffle(files)
        ut.dump(files,'files',uid)
    # files done 是已经做完的文件的list
    if ut.exist('files_done', uid):
        files_done = ut.load('files_done',uid)
    else:
        files_done = []
        ut.dump(files_done,'files_done',uid)
    for lists in files:
        for item in lists:
            if item not in files_done:
                return item
    else:
        print('所有的文件都已经做完了！')
        exit()


# 这是在一个文件做完之后切换到下一个文件的函数，其中输入是当前文件名，返回是下一个文件的文件名
def file_next(file):
    print('下一个文件')
    files = ut.load('files',uid)
    files_done = ut.load('files_done',uid)
    files_done.append(file)
    ut.dump(files_done,'files_done',uid)
    next = False
    group_for_file = []
    group_for_file1 = []
    for lists in files:
        for item in lists:
            if item == file:
                next = True
                group_for_file = lists
                continue
            if next:
                group_for_file1 = lists
                if group_for_file != group_for_file1:
                    # 只保留files 和 files_done 两个参数
                    ut.clear_folder(uid)
                    ut.dump(files,'files','uid')
                    ut.dump(files_done,'files_done','uid')
                return item
    else:
        print('所有的文件都已经做完了！')
        exit()

# Todo： 中途退出保存程序和重新加载的两个程序，逻辑并不是很清晰，可能需要重新设计
def stop_program(structures, file):
    ut.dump(structures,f'{file}_structures',uid)
    exit()

def get_sturctures(file):
    if ut.exist(f'{file}_structures',uid):
        return ut.load(f'{file}_structures',uid)
    else:
        structures = path.structure_list(file)
        return structures



def main():
    get_uid()
    file = file_to_do()
    structures = get_sturctures(file)
    print(file)
    while True:
        # shuffle 生成所有的candidate并按分数排序
        structures.shuffle()
        # show_picture_and_fb_line 会选择当前评分最高的候选结构并展示，返回值有candidate_path: 展示的rdf的relations元组
        # candidate_path_type 为 'path' 或者 'multipath'； rdf_new 返回展示的rdf； fbline 是展示的freebase里的元组
        # next_file是一个bool，如果为T则说明要进行下一个文件了
        candidate_path, candidate_path_type, rdf_new, fb_line, next_file = structures.show_picture_and_fb_line()
        if next_file:
            file_next(file)
            # todo: 这里逻辑没写清楚
            pass
        else:
            if candidate_path_type == 'path':
                answer_linguitistic = input('这两个图结构在语义上是否相同？（1：相同或几乎相同； 2：图像图结构语义存在不足； 3：图像图结构存在无关语义）')
                if answer_linguitistic == '1':
                    # 语义相同
                    structures.raise_linguistic_weight()
                    answer_structure = input('这两个图结构在结构上是否相同？（1：相同 2：不相同）')
                    # 结构相同
                    if answer_structure == '1':
                        write_aligned_result(file, fb_line, rdf_new)
                    # add_seen 是为了记录已经看过的结构
                    structures.add_seen(candidate_path, candidate_path_type)
                    # TODO: 结构不同的时候怎么更新
                elif answer_linguitistic == '2':
                    # 语义不足的时候将这个relation加入到'不足'这个list中，然后组合成multipath
                    structures.decrease_linguistic_weight()
                    structures.add_multipath(candidate_path)
                    structures.add_seen(candidate_path, candidate_path_type)
                elif answer_linguitistic == '3':
                    # 语义无关的时候删除这个关系
                    structures.decrease_linguistic_weight()
                    wuguan = input('哪个关系的语义无关？')
                    # Todo：重构structures好像需要的时间太多了，想想办法
                    structures.delete_unrelated_relations(candidate_path,int(wuguan))
                elif answer_linguitistic == 'exit':
                    # 暂时退出程序 todo：没写完
                    stop_program(structures, file)
            elif candidate_path_type == 'multipath':
                answer_linguitistic = input('这两个图结构在语义上是否相同？（1：相同或几乎相同； 2: 不相同）')
                if answer_linguitistic == '1':
                    structures.raise_linguistic_weight()
                    answer_structure = input('这两个图结构在结构上是否相同？（1：相同 2：不相同）')
                    if answer_structure == '1':
                        write_aligned_result(file, fb_line, rdf_new)
                    structures.add_seen(candidate_path, candidate_path_type)
                elif answer_linguitistic == '2':
                    structures.decrease_linguistic_weight()
                    pass
                elif answer_linguitistic == 'exit':
                    stop_program(structures, file)

def fuc1(file):
    print(file)
    s = path.structure_list(file)
    joblib.dump(s,
                f"/root/dxh/Fronter/structure_lists/{file[len('/root/dxh/candidate_full_path_not_empty/candidate_full_path'):]}")
def test():
    file= '/root/dxh/candidate_full_path_not_empty1/american_football.football_conference.teams.txt'

    print(file)
    structures = path.structure_list(file)

    # shuffle 生成所有的candidate并按分数排序
    structures.shuffle()
    # show_picture_and_fb_line 会选择当前评分最高的候选结构并展示，返回值有candidate_path: 展示的rdf的relations元组
    # candidate_path_type 为 'path' 或者 'multipath'； rdf_new 返回展示的rdf； fbline 是展示的freebase里的元组
    # next_file是一个bool，如果为T则说明要进行下一个文件了
    candiate_path, candiate_path_type, rdfs_list, next_file = structures.return_candidate_rdf()
    for i in range(len(rdfs_list)):
        rdf, head, tail = rdfs_list[i]
        nodes = set()
        edges = list()
        for lines in rdf:
            s, r, o = lines.strip().split('\t')
            nodes.add(s)
            nodes.add(o)
            edges.append([s, o, r])
        nodes = list(nodes)
        print(nodes)
        print(edges)
        break





if __name__ == '__main__':
    test()
