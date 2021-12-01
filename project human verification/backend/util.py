import joblib
import os
import shutil
import networkx as nx
import matplotlib.pyplot as plt
import re
import math
import codecs
import itertools

uid = None


def write_log(uid,text):
    with codecs.open(f'./log/{uid}_log.txt','a','utf-8') as i:
        print(text, file=i)

def write_aligned_result(out_file, fb_line, rdf_new, uid, dict_relation_full):
    out_file = out_file.strip('./candidate5m/').strip('.txt')
    file = f'./result/aligned_{out_file}_{uid}.txt'
    list1 = list()
    for lines in rdf_new:
        s, r, o = lines.strip().split('\t')
        if s.startswith('dr:'):
            s = '<http://dbpedia.org/resource/'+s[3:]+'>'
        if o.startswith('dr:'):
            o = '<http://dbpedia.org/resource/'+o[3:]+'>'
        l = []
        for r_full in dict_relation_full[r]:
            l.append('\t'.join((s,r_full,o)))
        list1.append(l)
    if len(list1) == 1:
        pass
    elif len(list1) == 2:
        list1 = itertools.product(list1[0],list1[1])
    elif len(list1) == 3:
        list1 = itertools.product(list1[0],list1[1],list1[2])
    elif len(list1) == 4:
        list1 = itertools.product(list1[0],list1[1],list1[2],list1[3])
    elif len(list1) == 5:
        list1 = itertools.product(list1[0],list1[1],list1[2],list1[3], list1[4])
    elif len(list1) == 6:
        list1 = itertools.product(list1[0],list1[1],list1[2],list1[3], list1[4], list1[5])
    list1 = list(list1)

    with codecs.open(file, 'a', 'utf-8') as out:
        for item in list1:
            out.write(fb_line)
            for lines in item:
            # s, r, o = item.strip().split('\t')
                out.write(lines+'\n')
            out.write('\n')
        out.write('\n')
    print('入库完毕')


def clear_folder(uid):
    shutil.rmtree(f'../test/{uid}/')
    os.mkdir(f'../test/{uid}/')


def dr(string):
    if string.startswith("<http://dbpedia.org/resource/") and string.endswith('>'):
        return "dr:" + string[29:-1]
    else:
        return string


def merge_rdf(rdf1, rdf2):
    list = []
    for lines in rdf1:
        list.append(lines)
    for lines in rdf2:
        if lines not in rdf1:
            list.append(lines)
    return list


def mkdir(path):
    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")

    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)
    # 判断结果
    if not isExists:
        os.makedirs(path)


def dump(item, name, uid):
    mkdir(f'../test/{uid}')
    joblib.dump(item, f'../test/{uid}/{name}')
    return 0


def exist(name, uid):
    path = f'../test/{uid}/{name}'
    if os.path.exists(path):
        return True
    else:
        return False


def load(name, uid):
    path = f'../test/{uid}/{name}'
    if os.path.exists(path):
        return joblib.load(f'../test/{uid}/{name}')
    else:
        return None


# 用来画图的函数， 不需要管， 参数输入rdf_list就好了
def nx_construct_path_from_rdf(rdf_list, plot=True):
    G = nx.DiGraph()
    plt.figure(figsize=(10, 10))
    node_list = []
    num = 1
    for lines in rdf_list:
        s, r, o = lines.strip().split('\t')
        s = '\n'.join(s.strip().split('_'))
        o = '\n'.join(o.strip().split('_'))
        r = r.strip().split('/')[-1][:-1]
        r = '\n'.join(re.findall('[a-zA-Z][^A-Z]*', r))
        if not s in node_list:
            G.add_node(s)
            node_list.append(s)
        if not o in node_list:
            G.add_node(o)
            node_list.append(o)
        G.add_edge(s, o, name=str(num) + ' : ' + r, weight=1 / math.sqrt(len(r)))
        num += 1
    pos = nx.spring_layout(G, weight="weight")
    # pos = nx.spring_layout(G,scale=0.2)
    # pos = nx.bipartite_layout(G,G.nodes)
    # pos = nx.circular_layout(G)
    option = {'arrowsize': 20,  # default 10
              'with_labels': True,
              'font_size': 15,
              'node_color': '#fbfdfe',
              'node_size': 10000}
    nx.draw_networkx(G, pos, **option)
    edge_labels = nx.get_edge_attributes(G, 'name')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=15)
    plt.show()




