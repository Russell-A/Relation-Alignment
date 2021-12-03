import codecs,os,time

class node():
    def __init__(self, name):
        self.name = name
        self.inneighbour_dict = dict()
        self.outneighbour_dict = dict()
        self.inedge_dict = dict() # 指向该节点的边
        self.outedge_dict = dict()
    
    def add_inneighbour_dict(self, node, edge):
        if node in self.inedge_dict:
            self.inneighbour_dict[node].append(edge)
        else:
            self.inneighbour_dict[node] = [edge]

    def add_outneighbour_dict(self, node, edge):
        if node in self.inedge_dict:
            self.outneighbour_dict[node].append(edge)
        else:
            self.outneighbour_dict[node] = [edge]

    def add_inedge_dict(self, node, edge):
        if edge in self.inedge_dict:
            self.inedge_dict[edge].append(node)
        else:
            self.inedge_dict[edge] = [node]
    
    def add_outedge_dict(self, node, edge):
        if edge in self.outedge_dict:
            self.outedge_dict[edge].append(node)
        else:
            self.outedge_dict[edge] = [node]





def DFS(structure, head, tail):
    node_name_set = set()
    dict_node = dict()
    for lines in set(structure):
        s, r, o = lines.strip().split('\t')
        # 新建两个节点和边
        if s not in node_name_set:
            node_s = node(s)
            # node_s = s
            node_name_set.add(s)
            dict_node[s] = node_s
        else:
            node_s = dict_node[s]
        if o not in node_name_set:
            node_o = node(o)
            # node_o = o
            node_name_set.add(o)
            dict_node[o] = node_o
        else:
            node_o = dict_node[o]
            pass

        node_s.add_outneighbour_dict(node_o,r)
        node_s.add_inneighbour_dict(node_s,r)
        node_s.add_outedge_dict(node_o,r)
        node_o.add_inedge_dict(node_s,r)


    '''
    找到所有的path
    '''
    f = 'forward'
    b = 'backward'
    paths = []
    node_start = dict_node[head]
    node_start : node
    for edge1 in node_start.outedge_dict.keys():
        for node1 in node_start.outedge_dict[edge1]:
            if node1.name == tail:
                relation = ((edge1, f ,1, 'n1'),)
                paths.append(relation)
            else:
                for edge2 in node1.outedge_dict.keys():
                    for node2 in node1.outedge_dict[edge2]:
                        if node2.name == tail:
                            relation = ((edge1, f ,1, 'n1'),(edge2, f , 2 , 'n2'))
                            paths.append(relation)
                        else:
                            for edge3 in node2.outedge_dict.keys():
                                for node3 in node2.outedge_dict[edge3]:
                                    if node3.name == tail:
                                        relation = ((edge1, f ,1, 'n1'),(edge2, f , 2 , 'n2'),(edge3,f,3,'n3'))
                                        paths.append(relation)
                            for edge3 in node2.inedge_dict.keys():
                                for node3 in node2.inedge_dict[edge3]:
                                    if node3.name == tail:
                                        relation = ((edge1, f ,1, 'n1'),(edge2, f , 2 , 'n2'),(edge3,b,3,'n3'))
                                        paths.append(relation)
                for edge2 in node1.inedge_dict.keys():
                    for node2 in node1.inedge_dict[edge2]:
                        if node2.name == tail:
                            relation = ((edge1, f ,1, 'n1'),(edge2, b , 2 , 'n2'))
                            paths.append(relation)
                        else:
                            for edge3 in node2.outedge_dict.keys():
                                for node3 in node2.outedge_dict[edge3]:
                                    if node3.name == tail:
                                        relation = ((edge1, f ,1, 'n1'),(edge2, b , 2 , 'n2'),(edge3,f,3,'n3'))
                                        paths.append(relation)
                            for edge3 in node2.inedge_dict.keys():
                                for node3 in node2.inedge_dict[edge3]:
                                    if node3.name == tail:
                                        relation = ((edge1, f ,1, 'n1'),(edge2, b , 2 , 'n2'),(edge3,b,3,'n3'))
                                        paths.append(relation)
    for edge1 in node_start.inedge_dict.keys():
        for node1 in node_start.inedge_dict[edge1]:
            if node1.name == tail:
                relation = ((edge1, b ,1, 'n1'),)
                paths.append(relation)
            else:
                for edge2 in node1.outedge_dict.keys():
                    for node2 in node1.outedge_dict[edge2]:
                        if node2.name == tail:
                            relation = ((edge1, b ,1, 'n1'),(edge2, f , 2 , 'n2'))
                            paths.append(relation)
                        else:
                            for edge3 in node2.outedge_dict.keys():
                                for node3 in node2.outedge_dict[edge3]:
                                    if node3.name == tail:
                                        relation = ((edge1, b ,1, 'n1'),(edge2, f , 2 , 'n2'),(edge3,f,3,'n3'))
                                        paths.append(relation)
                            for edge3 in node2.inedge_dict.keys():
                                for node3 in node2.inedge_dict[edge3]:
                                    if node3.name == tail:
                                        relation = ((edge1, b ,1, 'n1'),(edge2, f , 2 , 'n2'),(edge3,b,3,'n3'))
                                        paths.append(relation)
                for edge2 in node1.inedge_dict.keys():
                    for node2 in node1.inedge_dict[edge2]:
                        if node2.name == tail:
                            relation = ((edge1, b ,1, 'n1'),(edge2, b , 2 , 'n2'))
                            paths.append(relation)
                        else:
                            for edge3 in node2.outedge_dict.keys():
                                for node3 in node2.outedge_dict[edge3]:
                                    if node3.name == tail:
                                        relation = ((edge1, b ,1, 'n1'),(edge2, b , 2 , 'n2'),(edge3,f,3,'n3'))
                                        paths.append(relation)
                            for edge3 in node2.inedge_dict.keys():
                                for node3 in node2.inedge_dict[edge3]:
                                    if node3.name == tail:
                                        relation = ((edge1, b ,1, 'n1'),(edge2, b , 2 , 'n2'),(edge3,b,3,'n3'))
                                        paths.append(relation)

                        
                        









    

def FCGA(structure, head, tail):
    node_name_set = set()
    dict_node = dict()
    for lines in set(structure):
        s, r, o = lines.strip().split('\t')
        # 新建两个节点和边
        if s not in node_name_set:
            node_s = node(s)
            # node_s = s
            node_name_set.add(s)
            dict_node[s] = node_s
        else:
            node_s = dict_node[s]
        if o not in node_name_set:
            node_o = node(o)
            # node_o = o
            node_name_set.add(o)
            dict_node[o] = node_o
        else:
            node_o = dict_node[o]
            pass

        node_s.add_outneighbour_dict(node_o,r)
        node_s.add_inneighbour_dict(node_s,r)
        node_s.add_outedge_dict(node_o,r)
        node_o.add_inedge_dict(node_s,r)

    '''
    找到所有的path
    '''
    f = 'forward'
    b = 'backward'
    paths = []
    node_start = dict_node[head]
    node_start:node
    for node1 in node_start.outneighbour_dict.keys():
        if node1.name == tail:
            for edge1 in node_start.outneighbour_dict[node1]:
                relation = ((edge1, f , 1, 'n1'),)
                paths.append(relation)
        else:
            for node2 in node1.outneighbour_dict.keys():
                if node2.name == tail:
                    for edge1 in node_start.outneighbour_dict[node1]:
                        for edge2 in node1.outneighbour_dict[node2]:
                            relation = ((edge1, f , 1, 'n1'),(edge2, f , 2, 'n2'),)
                            paths.append(relation)
                else:
                    for node3 in node2.outneighbour_dict.keys():
                        if node3.name == tail:
                            for edge1 in node_start.outneighbour_dict[node1]:
                                for edge2 in node1.outneighbour_dict[node2]:
                                    for edge3 in node2.outneighbour_dict[node3]:
                                        relation = ((edge1, f , 1, 'n1'),(edge2, f , 2, 'n2'),(edge3, f, 3, 'n3'))
                                        paths.append(relation)
                    for node3 in node2.inneighbour_dict.keys():
                        if node3.name == tail:
                            for edge1 in node_start.outneighbour_dict[node1]:
                                for edge2 in node1.outneighbour_dict[node2]:
                                    for edge3 in node2.inneighbour_dict[node3]:
                                        relation = ((edge1, f , 1, 'n1'),(edge2, f , 2, 'n2'),(edge3, b, 3, 'n3'))
                                        paths.append(relation)
            for node2 in node1.inneighbour_dict.keys():
                if node2.name == tail:
                    for edge1 in node_start.outneighbour_dict[node1]:
                        for edge2 in node1.inneighbour_dict[node2]:
                            relation = ((edge1, f , 1, 'n1'),(edge2, b , 2, 'n2'),)
                            paths.append(relation)
                else:
                    for node3 in node2.outneighbour_dict.keys():
                        if node3.name == tail:
                            for edge1 in node_start.outneighbour_dict[node1]:
                                for edge2 in node1.inneighbour_dict[node2]:
                                    for edge3 in node2.outneighbour_dict[node3]:
                                        relation = ((edge1, f , 1, 'n1'),(edge2, b , 2, 'n2'),(edge3, f, 3, 'n3'))
                                        paths.append(relation)
                    for node3 in node2.inneighbour_dict.keys():
                        if node3.name == tail:
                            for edge1 in node_start.outneighbour_dict[node1]:
                                for edge2 in node1.inneighbour_dict[node2]:
                                    for edge3 in node2.intneighbour_dict[node3]:
                                        relation = ((edge1, f , 1, 'n1'),(edge2, b , 2, 'n2'),(edge3, b, 3, 'n3'))
                                        paths.append(relation)
    for node1 in node_start.inneighbour_dict.keys():
        if node1.name == tail:
            for edge1 in node_start.inneighbour_dict[node1]:
                relation = ((edge1, b , 1, 'n1'),)
                paths.append(relation)
        else:
            for node2 in node1.outneighbour_dict.keys():
                if node2.name == tail:
                    for edge1 in node_start.inneighbour_dict[node1]:
                        for edge2 in node1.outneighbour_dict[node2]:
                            relation = ((edge1, b , 1, 'n1'),(edge2, f , 2, 'n2'),)
                            paths.append(relation)
                else:
                    for node3 in node2.outneighbour_dict.keys():
                        if node3.name == tail:
                            for edge1 in node_start.inneighbour_dict[node1]:
                                for edge2 in node1.outneighbour_dict[node2]:
                                    for edge3 in node2.outneighbour_dict[node3]:
                                        relation = ((edge1, b , 1, 'n1'),(edge2, f , 2, 'n2'),(edge3, f, 3, 'n3'))
                                        paths.append(relation)
                    for node3 in node2.inneighbour_dict.keys():
                        if node3.name == tail:
                            for edge1 in node_start.inneighbour_dict[node1]:
                                for edge2 in node1.outneighbour_dict[node2]:
                                    for edge3 in node2.inneighbour_dict[node3]:
                                        relation = ((edge1, b , 1, 'n1'),(edge2, f , 2, 'n2'),(edge3, b, 3, 'n3'))
                                        paths.append(relation)
            for node2 in node1.inneighbour_dict.keys():
                if node2.name == tail:
                    for edge1 in node_start.inneighbour_dict[node1]:
                        for edge2 in node1.inneighbour_dict[node2]:
                            relation = ((edge1, b , 1, 'n1'),(edge2, b , 2, 'n2'),)
                            paths.append(relation)
                else:
                    for node3 in node2.outneighbour_dict.keys():
                        if node3.name == tail:
                            for edge1 in node_start.inneighbour_dict[node1]:
                                for edge2 in node1.inneighbour_dict[node2]:
                                    for edge3 in node2.outneighbour_dict[node3]:
                                        relation = ((edge1, b , 1, 'n1'),(edge2, b , 2, 'n2'),(edge3, f, 3, 'n3'))
                                        paths.append(relation)
                    for node3 in node2.inneighbour_dict.keys():
                        if node3.name == tail:
                            for edge1 in node_start.inneighbour_dict[node1]:
                                for edge2 in node1.inneighbour_dict[node2]:
                                    for edge3 in node2.intneighbour_dict[node3]:
                                        relation = ((edge1, b , 1, 'n1'),(edge2, b , 2, 'n2'),(edge3, b, 3, 'n3'))
                                        paths.append(relation)
                    




if __name__ == '__main__':
    for item in os.walk('./datasets/'):
        files = item[2]
    candidate_files = list(map(lambda x: './datasets/'+x, files))

    f = codecs.open('./FCGA.txt','w','utf-8')
    result = []
    total = len(candidate_files)
    finish_num = 0
    for file in candidate_files:
        if not file.endswith('.txt'):
            continue
        time_for_DFS = 0
        time_for_FCGA = 0
        relations = 0
        entities = 0
        raw_relation_structure = 0
        filesize = os.path.getsize(file)
        with codecs.open(file, 'r','utf-8') as input:
            line_num = 0
            structure = []
            entity_set = set()
            relation_set = set()
            for line in input:
                line_num += 1

                if line == '\n':
                    line_num = 0
                    if structure == []:
                        continue
                    time1 = time.time()
                    DFS(structure,db_h,db_o)
                    time2 = time.time()
                    FCGA(structure,db_h, db_o)
                    time3 = time.time()
                    time_for_DFS += time2-time1
                    time_for_FCGA += time3-time2
                    structure = []
                    entities += len(entity_set)
                    relations += len(relation_set)
                    entity_set = set()
                    relation_set = set()
                    raw_relation_structure += 1
                    continue
                if line_num == 1:
                    fbline = line
                    continue
                elif line_num == 2:
                    dbline = line
                    db_h, db_o = dbline.strip().split('\t')
                    if db_h.startswith('<http://dbpedia.org/resource/'):
                        db_h = 'dr:'+db_h[len('<http://dbpedia.org/resource/'):-1]
                    if db_o.startswith('<http://dbpedia.org/resource/'):
                        db_o = 'dr'+db_o[len('<http://dbpedia.org/resource/'):-1]
                    continue
                else:
                    s, r, o = line.strip().split('\t')
                    r_full = r
                    r = r.strip().split('/')[-1][:-1]
                    line = '\t'.join((s, r, o))
                    structure.append(line.strip())
                    entity_set.add(s)
                    entity_set.add(o)
                    relation_set.add((s,r,o))

        print('file',file,'time_for_DFS',time_for_DFS,'time_for_FCGA',time_for_FCGA,
              'raw_relation_structure',raw_relation_structure, 'filesize',filesize, 'entities',entities,
              'relations',relations, 'timeration', time_for_DFS/time_for_FCGA ,file= f)
        finish_num+=1
        
        result.append([time_for_DFS/time_for_FCGA, file, time_for_DFS, time_for_FCGA, raw_relation_structure, filesize, entities, relations])
    f.close()
    print('done')


