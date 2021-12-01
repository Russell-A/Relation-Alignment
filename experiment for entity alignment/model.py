import codecs
import numpy as np
import copy
import time
import random
import json
import multiprocessing
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import os
import joblib

entities2id = {}
relations2id = {}
relation_tph = {}
relation_hpt = {}


def dataloader(file1, file2, file3, file4):
    print("load file...")

    entity = []
    relation = []
    with open(file2, 'r') as f1, open(file3, 'r') as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()
        for line in lines1:
            line = line.strip().split('\t')
            if len(line) != 2:
                continue
            entities2id[line[0]] = line[1]
            entity.append(int(line[1]))

        for line in lines2:
            line = line.strip().split('\t')
            if len(line) != 2:
                continue
            relations2id[line[0]] = line[1]
            relation.append(int(line[1]))

    triple_list = []
    relation_head = {}
    relation_tail = {}

    with codecs.open(file1, 'r') as f:
        content = f.readlines()
        for line in content:
            triple = line.strip().split("\t")
            if len(triple) != 3:
                continue

            h_ = int(entities2id[triple[0]])
            r_ = int(relations2id[triple[1]])
            t_ = int(entities2id[triple[2]])

            triple_list.append([h_, r_, t_])
            if r_ in relation_head:
                if h_ in relation_head[r_]:
                    relation_head[r_][h_] += 1
                else:
                    relation_head[r_][h_] = 1
            else:
                relation_head[r_] = {}
                relation_head[r_][h_] = 1

            if r_ in relation_tail:
                if t_ in relation_tail[r_]:
                    relation_tail[r_][t_] += 1
                else:
                    relation_tail[r_][t_] = 1
            else:
                relation_tail[r_] = {}
                relation_tail[r_][t_] = 1

    for r_ in relation_head:
        sum1, sum2 = 0, 0
        for head in relation_head[r_]:
            sum1 += 1
            sum2 += relation_head[r_][head]
        tph = sum2 / sum1
        relation_tph[r_] = tph

    for r_ in relation_tail:
        sum1, sum2 = 0, 0
        for tail in relation_tail[r_]:
            sum1 += 1
            sum2 += relation_tail[r_][tail]
        hpt = sum2 / sum1
        relation_hpt[r_] = hpt

    valid_triple_list = []
    with codecs.open(file4, 'r') as f:
        content = f.readlines()
        for line in content:
            triple = line.strip().split("\t")
            if len(triple) != 3:
                continue

            h_ = int(entities2id[triple[0]])
            r_ = int(relations2id[triple[1]])
            t_ = int(entities2id[triple[2]])

            valid_triple_list.append([h_, r_, t_])

    print("Complete load. entity : %d , relation : %d , train triple : %d, , valid triple : %d" % (
        len(entity), len(relation), len(triple_list), len(valid_triple_list)))

    return entity, relation, triple_list, valid_triple_list


def norm_l1(h, r, t):
    return np.sum(np.fabs(h + r - t))


def norm_l2(h, r, t):
    return np.sum(np.square(h + r - t))


class Entity_Measure:
    def __init__(self, entity2id_1, entity2id_2, entity_emb_1, entity_emb_2):
        self.entity2_id_1 = entity2id_1
        self.entity2_id_2 = entity2id_2
        self.entity_emb_1 = entity_emb_1
        self.entity_emb_2 = entity_emb_2
        self.entitydic_1 = {}
        self.entitydic_2 = {}
        self.entity_vec_1 = {}
        self.entity_vec_2 = {}

        self.test_sample_count = 0
        self.hit_1_count = 0
        self.hit_10_count = 0
        self.mean_rank = []
        self.hit_1 = 0
        self.hit_10 = 0

    def load_dic(self):
        print('load dic ...')
        with codecs.open(self.entity2_id_1, 'r', encoding='UTF-8') as f1, codecs.open(self.entity2_id_2, 'r',
                                                                                      encoding='UTF-8') as f2:
            lines1 = f1.readlines()
            lines2 = f2.readlines()
            for line in lines1:
                line = line.strip().split('\t')
                if len(line) != 2:
                    continue
                self.entitydic_1[line[0]] = line[1]

            for line in lines2:
                line = line.strip().split('\t')
                if len(line) != 2:
                    continue
                self.entitydic_2[line[0]] = line[1]
        print('load dic done!')

    def load_vec(self):
        print('load vec ...')
        with codecs.open(self.entity_emb_1, 'r') as f1, codecs.open(self.entity_emb_2, 'r') as f2:
            lines1 = f1.readlines()
            lines2 = f2.readlines()
            for line in lines1:
                line = line.strip().split('\t')
                if len(line) != 2:
                    continue
                self.entity_vec_1[int(line[0])] = json.loads(line[1])

            for line in lines2:
                line = line.strip().split('\t')
                if len(line) != 2:
                    continue
                self.entity_vec_2[int(line[0])] = json.loads(line[1])
        print('load vec done!')

    def calculate_single_pair(self, shiti1, shiti2):
        query = self.entity_vec_1[eval(self.entitydic_1[shiti1])]
        answer = eval(self.entitydic_2[shiti2])
        temporary_dic = copy.deepcopy(self.entity_vec_2)

        for index, value in enumerate(temporary_dic.values()):
            temporary_dic[index] = np.linalg.norm(np.array(value) - np.array(query))
        temporary_list = sorted(temporary_dic.items(), key=lambda x: x[1], reverse=False)

        hit_10_list = [temporary_list[x][0] for x in range(10)]

        for index, all_answer in enumerate(hit_10_list):
            if answer == hit_10_list[index]:
                self.hit_10_count += 1
                if index == 0:
                    self.hit_1_count += 1

        # 计算mean_rank
        for index, value in enumerate(temporary_list):
            if value[0] == answer:
                self.mean_rank.append(index + 1)

    def calculate_all(self, entity_test_way, outputfile):
        start = time.time()

        print('start calculate ...')
        with codecs.open(entity_test_way, 'r', encoding='UTF-8') as f:
            lines = f.readlines()
            self.test_sample_count = len(lines)
            print(self.test_sample_count, 'samples')
            for line in lines:
                line = line.strip().split()
                self.calculate_single_pair(line[0], line[1])

        self.hit_1 = self.hit_1_count / self.test_sample_count
        self.hit_10 = self.hit_10_count / self.test_sample_count
        self.mean_rank = np.array(self.mean_rank).mean()
        end = time.time()
        with codecs.open(f'{outputfile}test_result.txt', 'w', encoding='UTF-8') as f:
            f.write(f'consuming {end - start} s')
            f.write('\n')
            f.write(f'hit_1 is {self.hit_1} hit_10 is {self.hit_10} mean_rank is {self.mean_rank}')
        print('calculate done! consuming', end - start, 's')
        print('hit_1 is', self.hit_1, 'hit_10 is', self.hit_10, 'mean_rank is', self.mean_rank)

    def calculate_single_pair_multi(self, line, i):
        hit1 = 0
        hit10 = 0
        mean_rank = 0

        line = line.strip().split()
        shiti1, shiti2 = line[1], line[0]
        query = self.entity_vec_1[eval(self.entitydic_1[shiti1])]
        answer = eval(self.entitydic_2[shiti2])
        temporary_dic = copy.deepcopy(self.entity_vec_2)

        for index, value in enumerate(temporary_dic.values()):
            temporary_dic[index] = np.linalg.norm(np.array(value) - np.array(query))
        temporary_list = sorted(temporary_dic.items(), key=lambda x: x[1], reverse=False)


        hit_10_list = [temporary_list[x][0] for x in range(10)]

        for index, all_answer in enumerate(hit_10_list):
            if answer == hit_10_list[index]:
                hit10 = 1
                if index == 0:
                    hit1 = 1

        for index, value in enumerate(temporary_list):
            if value[0] == answer:
                mean_rank = index

        print(i, 'done!', hit1, hit10, mean_rank)

    def calculate_all_multi(self, entity_test_way, outputfile):
        p = multiprocessing.Pool(10)
        print('start calculate ...')
        with codecs.open(entity_test_way, 'r', encoding='UTF-8') as f:
            lines = f.readlines()
            self.test_sample_count = len(lines)
            for index, line in enumerate(lines):
                p.apply_async(self.calculate_single_pair_multi, (line, index))
            p.close()
            p.join()



class Multi_Model(nn.Module):
    def __init__(self, entity_num_1, relation_num_1, entity_num_2, relation_num_2, dim, margin, norm, C):
        super(Multi_Model, self).__init__()
        self.entity_num_1 = entity_num_1
        self.entities_1 = [x for x in range(self.entity_num_1)]
        self.relation_num_1 = relation_num_1
        self.entity_num_2 = entity_num_2
        self.entities_2 = [x for x in range(self.entity_num_2)]
        self.relation_num_2 = relation_num_2
        self.dim = dim
        self.margin = margin
        self.norm = norm
        self.C = C
        self.entity_dic_1 = {}
        self.entity_dic_2 = {}
        self.relation_dic_1 = {}
        self.relation_dic_2 = {}

        self.ent_embedding_1 = torch.nn.Embedding(num_embeddings=self.entity_num_1,
                                                  embedding_dim=self.dim).cuda()
        self.rel_embedding_1 = torch.nn.Embedding(num_embeddings=self.relation_num_1,
                                                  embedding_dim=self.dim).cuda()
        self.ent_embedding_2 = torch.nn.Embedding(num_embeddings=self.entity_num_2,
                                                  embedding_dim=self.dim).cuda()
        self.rel_embedding_2 = torch.nn.Embedding(num_embeddings=self.relation_num_2,
                                                  embedding_dim=self.dim).cuda()

        self.loss_F = nn.MarginRankingLoss(self.margin, reduction="mean")
        self.loss_mse = nn.MSELoss()

        self.__data_init()

    def __data_init(self):
        nn.init.xavier_uniform_(self.ent_embedding_1.weight)
        nn.init.xavier_uniform_(self.ent_embedding_2.weight)
        nn.init.xavier_uniform_(self.rel_embedding_1.weight)
        nn.init.xavier_uniform_(self.rel_embedding_2.weight)
        self.normalization_rel_embedding()
        self.normalization_ent_embedding()

    def normalization_ent_embedding(self):
        norm1 = self.ent_embedding_1.weight.detach().cpu().numpy()
        norm1 = norm1 / np.sqrt(np.sum(np.square(norm1), axis=1, keepdims=True))
        self.ent_embedding_1.weight.data.copy_(torch.from_numpy(norm1))

        norm2 = self.ent_embedding_2.weight.detach().cpu().numpy()
        norm2 = norm2 / np.sqrt(np.sum(np.square(norm2), axis=1, keepdims=True))
        self.ent_embedding_2.weight.data.copy_(torch.from_numpy(norm2))

    def normalization_rel_embedding(self):
        norm1 = self.rel_embedding_1.weight.detach().cpu().numpy()
        norm1 = norm1 / np.sqrt(np.sum(np.square(norm1), axis=1, keepdims=True))
        self.rel_embedding_1.weight.data.copy_(torch.from_numpy(norm1))

        norm2 = self.rel_embedding_2.weight.detach().cpu().numpy()
        norm2 = norm2 / np.sqrt(np.sum(np.square(norm2), axis=1, keepdims=True))
        self.rel_embedding_2.weight.data.copy_(torch.from_numpy(norm2))

    def prepare_data(self, entity1_way, entity2_way, relation1_way, relation2_way):
        print('Prepare data...')
        with codecs.open(entity1_way, 'r', encoding='UTF-8') as f1, codecs.open(entity2_way, 'r',
                                                                                encoding='UTF-8') as f2:
            lines1 = f1.readlines()
            lines2 = f2.readlines()
            for line in lines1:
                line = line.strip().split('\t')
                if len(line) != 2:
                    continue
                self.entity_dic_1[line[0]] = line[1]

            for line in lines2:
                line = line.strip().split('\t')
                if len(line) != 2:
                    continue
                self.entity_dic_2[line[0]] = line[1]

        f1.close()
        f2.close()

        with codecs.open(relation1_way, 'r', encoding='UTF-8') as f1, codecs.open(relation2_way, 'r',
                                                                                  encoding='UTF-8') as f2:
            lines1 = f1.readlines()
            lines2 = f2.readlines()
            for line in lines1:
                line = line.strip().split('\t')
                if len(line) != 2:
                    continue
                self.relation_dic_1[line[0]] = line[1]

            for line in lines2:
                line = line.strip().split('\t')
                if len(line) != 2:
                    continue
                self.relation_dic_2[line[0]] = line[1]

        f1.close()
        f2.close()

        print('Prepare data done!')

    def entity_train_data(self, entity_train_way):
        entity_train_data = []
        with codecs.open(entity_train_way, 'r', encoding='UTF-8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split()
                entity_train_data.append([int(self.entity_dic_2[line[0]]), int(self.entity_dic_1[line[1]])])

        f.close()
        return entity_train_data

    def get_triples(self, file1, file2):

        triple_list_1 = []
        triple_list_2 = []
        with codecs.open(file1, 'r', encoding='UTF-8') as f:
            content = f.readlines()
            for line in content:
                triple = line.strip().split("\t")
                if len(triple) != 3:
                    continue

                h_ = int(self.entity_dic_1[triple[0]])
                r_ = int(self.relation_dic_1[triple[1]])
                t_ = int(self.entity_dic_1[triple[2]])

                triple_list_1.append([h_, r_, t_])

        f.close()

        with codecs.open(file2, 'r', encoding='UTF-8') as f:
            content = f.readlines()
            for line in content:
                triple = line.strip().split("\t")
                if len(triple) != 3:
                    continue

                h_ = int(self.entity_dic_2[triple[0]])
                r_ = int(self.relation_dic_2[triple[1]])
                t_ = int(self.entity_dic_2[triple[2]])

                triple_list_2.append([h_, r_, t_])

        return triple_list_1, triple_list_2

    def distance(self, h, r, t, entity):
        if entity == 'entity_1':
            head = self.ent_embedding_1(h)
            rel = self.rel_embedding_1(r)
            tail = self.ent_embedding_1(t)

            distance = head + rel - tail

            score = torch.norm(distance, p=self.norm, dim=1)
        else:
            head = self.ent_embedding_2(h)
            rel = self.rel_embedding_2(r)
            tail = self.ent_embedding_2(t)

            distance = head + rel - tail

            score = torch.norm(distance, p=self.norm, dim=1)
        return score

    def train_relation(self, ent_rel_1, ent_rel_2):
        ent_tel_1_list = []
        ent_tel_2_list = []

        for value in ent_rel_1[0]:
            ent_tel_1_list.append(
                self.ent_embedding_1(torch.tensor(eval(self.entity_dic_1[value])).long().cuda()).unsqueeze(0))
        for value in ent_rel_1[1]:
            ent_tel_1_list.append(
                self.ent_embedding_1(torch.tensor(eval(self.relation_dic_1[value])).long().cuda()).unsqueeze(0))

        for value in ent_rel_2[0]:
            ent_tel_2_list.append(
                self.ent_embedding_2(torch.tensor(eval(self.entity_dic_2[value])).long().cuda()).unsqueeze(0))
        for value in ent_rel_2[1]:
            ent_tel_2_list.append(
                self.ent_embedding_2(torch.tensor(eval(self.relation_dic_2[value])).long().cuda()).unsqueeze(0))

        ent_tel_1_list = torch.cat(ent_tel_1_list, dim=0)
        ent_tel_2_list = torch.cat(ent_tel_2_list, dim=0)

        ent_tel_1_mean = torch.mean(ent_tel_1_list, dim=0)
        ent_tel_2_mean = torch.mean(ent_tel_2_list, dim=0)
        loss = self.loss_mse(ent_tel_1_mean, ent_tel_2_mean)
        return loss

    def train_entity_align(self, entity_1, entity_2):
        ent_1_vec = self.ent_embedding_2(torch.tensor(eval(self.entity_dic_2[entity_1])).long().cuda())
        ent_2_vec = self.ent_embedding_1(torch.tensor(eval(self.entity_dic_1[entity_2])).long().cuda())
        loss = self.loss_mse(ent_1_vec, ent_2_vec)
        return loss

    def scale_loss(self, embedding):
        return torch.sum(
            torch.max(
                torch.sum(
                    embedding ** 2, dim=1, keepdim=True
                ) - torch.autograd.Variable(torch.FloatTensor([1.0]).cuda()),
                torch.autograd.Variable(torch.FloatTensor([0.0]).cuda())
            ))

    def forward(self, current_triples, corrupted_triples, train_type):
        # current_triples和corrupted_triples输进来的是tensor维度为(batch_size,3)

        if train_type == 'entity_1':  # 这里是对实体进行训练
            h, r, t = torch.chunk(current_triples, 3, dim=1)
            h_c, r_c, t_c = torch.chunk(corrupted_triples, 3, dim=1)

            h = torch.squeeze(h, dim=1).cuda()
            r = torch.squeeze(r, dim=1).cuda()
            t = torch.squeeze(t, dim=1).cuda()
            h_c = torch.squeeze(h_c, dim=1).cuda()
            r_c = torch.squeeze(r_c, dim=1).cuda()
            t_c = torch.squeeze(t_c, dim=1).cuda()

            # torch.nn.embedding类的forward只接受longTensor类型的张量
            pos = self.distance(h, r, t, 'entity_1')
            neg = self.distance(h_c, r_c, t_c, 'entity_1')

            entity_embedding = self.ent_embedding_1(torch.cat([h, t, h_c, t_c]).cuda())
            relation_embedding = self.rel_embedding_1(torch.cat([r, r_c]).cuda())

            # loss_F = max(0, -y*(x1-x2) + margin)
            # loss1 = torch.sum(torch.relu(pos - neg + self.margin))
            y = Variable(torch.Tensor([-1])).cuda()
            loss = self.loss_F(pos, neg, y)

            ent_scale_loss = self.scale_loss(entity_embedding)
            rel_scale_loss = self.scale_loss(relation_embedding)

            return loss # + self.C * (ent_scale_loss / len(entity_embedding) + rel_scale_loss / len(relation_embedding))
        elif train_type == 'entity_2':  # 这里是对实体进行训练
            h, r, t = torch.chunk(current_triples, 3, dim=1)

            h_c, r_c, t_c = torch.chunk(corrupted_triples, 3, dim=1)

            h = torch.squeeze(h, dim=1).cuda()
            r = torch.squeeze(r, dim=1).cuda()
            t = torch.squeeze(t, dim=1).cuda()
            h_c = torch.squeeze(h_c, dim=1).cuda()
            r_c = torch.squeeze(r_c, dim=1).cuda()
            t_c = torch.squeeze(t_c, dim=1).cuda()

            # torch.nn.embedding类的forward只接受longTensor类型的张量

            pos = self.distance(h, r, t, 'entity_2')
            neg = self.distance(h_c, r_c, t_c, 'entity_2')

            entity_embedding = self.ent_embedding_2(torch.cat([h, t, h_c, t_c]).cuda())
            relation_embedding = self.rel_embedding_2(torch.cat([r, r_c]).cuda())

            # loss_F = max(0, -y*(x1-x2) + margin)
            # loss1 = torch.sum(torch.relu(pos - neg + self.margin))
            y = Variable(torch.Tensor([-1])).cuda()
            loss = self.loss_F(pos, neg, y)

            ent_scale_loss = self.scale_loss(entity_embedding)
            rel_scale_loss = self.scale_loss(relation_embedding)
            return loss # + self.C * (ent_scale_loss / len(entity_embedding) + rel_scale_loss / len(relation_embedding))
        elif train_type == 'relation':  # 这里是对关系进行训练，即type == 'relation',此时输入的是(batch_size*)
            if corrupted_triples == 'p':
                h1, r1, t1, h2, r2, t2 = torch.chunk(current_triples, 6, dim=1)
                h1 = torch.squeeze(h1, dim=1).cuda()
                r1 = torch.squeeze(r1, dim=1).cuda()
                t1 = torch.squeeze(t1, dim=1).cuda()
                h2 = torch.squeeze(h2, dim=1).cuda()
                r2 = torch.squeeze(r2, dim=1).cuda()
                t2 = torch.squeeze(t2, dim=1).cuda()
                dbp1 = self.ent_embedding_1(h1)
                dwy1 = self.ent_embedding_2(h2)
                dbp2 = self.rel_embedding_1(r1)
                dwy2 = self.rel_embedding_2(r2)
                dbp3 = self.ent_embedding_1(t1)
                dwy3 = self.ent_embedding_2(t2)
                loss = 5 * (self.loss_mse(dbp1, dwy1) + self.loss_mse(dbp2, dwy2) + self.loss_mse(dbp3, dwy3))
                return loss
            if corrupted_triples == 'n':
                h1, r1, t1, h2, r2, t2 = torch.chunk(current_triples, 6, dim=1)
                h1 = torch.squeeze(h1, dim=1).cuda()
                r1 = torch.squeeze(r1, dim=1).cuda()
                t1 = torch.squeeze(t1, dim=1).cuda()
                h2 = torch.squeeze(h2, dim=1).cuda()
                r2 = torch.squeeze(r2, dim=1).cuda()
                t2 = torch.squeeze(t2, dim=1).cuda()
                dbp1 = self.ent_embedding_1(h1)
                dwy1 = self.ent_embedding_2(h2)
                dbp2 = self.rel_embedding_1(r1)
                dwy2 = - self.rel_embedding_2(r2)
                dbp3 = self.ent_embedding_1(t1)
                dwy3 = self.ent_embedding_2(t2)
                loss = 5 * (self.loss_mse(dbp1, dwy1) + self.loss_mse(dbp2, dwy2) + self.loss_mse(dbp3, dwy3))
                return loss
            if corrupted_triples == 'pp':
                h1, r1, t1, h2, r2, r22, t2 = torch.chunk(current_triples, 7, dim=1)
                h1 = torch.squeeze(h1, dim=1).cuda()
                r1 = torch.squeeze(r1, dim=1).cuda()
                t1 = torch.squeeze(t1, dim=1).cuda()
                h2 = torch.squeeze(h2, dim=1).cuda()
                r2 = torch.squeeze(r2, dim=1).cuda()
                r22 = torch.squeeze(r22, dim=1).cuda()
                t2 = torch.squeeze(t2, dim=1).cuda()
                dbp1 = self.ent_embedding_1(h1)
                dwy1 = self.ent_embedding_2(h2)
                dbp2 = self.rel_embedding_1(r1)
                dwy2 = self.rel_embedding_2(r2)
                dwy22 = self.rel_embedding_2(r22)
                dbp3 = self.ent_embedding_1(t1)
                dwy3 = self.ent_embedding_2(t2)
                loss = 5 * (self.loss_mse(dbp1, dwy1) + self.loss_mse(dbp2, dwy2+dwy22) + self.loss_mse(dbp3, dwy3))
                return loss
            if corrupted_triples == 'pn':
                h1, r1, t1, h2, r2, r22, t2 = torch.chunk(current_triples, 7, dim=1)
                h1 = torch.squeeze(h1, dim=1).cuda()
                r1 = torch.squeeze(r1, dim=1).cuda()
                t1 = torch.squeeze(t1, dim=1).cuda()
                h2 = torch.squeeze(h2, dim=1).cuda()
                r2 = torch.squeeze(r2, dim=1).cuda()
                r22 = torch.squeeze(r22, dim=1).cuda()
                t2 = torch.squeeze(t2, dim=1).cuda()
                dbp1 = self.ent_embedding_1(h1)
                dwy1 = self.ent_embedding_2(h2)
                dbp2 = self.rel_embedding_1(r1)
                dwy2 = self.rel_embedding_2(r2)
                dwy22 = self.rel_embedding_2(r22)
                dbp3 = self.ent_embedding_1(t1)
                dwy3 = self.ent_embedding_2(t2)
                loss = 5 * (self.loss_mse(dbp1, dwy1) + self.loss_mse(dbp2, dwy2-dwy22) + self.loss_mse(dbp3, dwy3))
                return loss
            if corrupted_triples == 'np':
                h1, r1, t1, h2, r2, r22, t2 = torch.chunk(current_triples, 7, dim=1)
                h1 = torch.squeeze(h1, dim=1).cuda()
                r1 = torch.squeeze(r1, dim=1).cuda()
                t1 = torch.squeeze(t1, dim=1).cuda()
                h2 = torch.squeeze(h2, dim=1).cuda()
                r2 = torch.squeeze(r2, dim=1).cuda()
                r22 = torch.squeeze(r22, dim=1).cuda()
                t2 = torch.squeeze(t2, dim=1).cuda()
                dbp1 = self.ent_embedding_1(h1)
                dwy1 = self.ent_embedding_2(h2)
                dbp2 = self.rel_embedding_1(r1)
                dwy2 = self.rel_embedding_2(r2)
                dwy22 = self.rel_embedding_2(r22)
                dbp3 = self.ent_embedding_1(t1)
                dwy3 = self.ent_embedding_2(t2)
                loss = 5 * (self.loss_mse(dbp1, dwy1) + self.loss_mse(dbp2, -dwy2+dwy22) + self.loss_mse(dbp3, dwy3))
                return loss
            if corrupted_triples == 'nn':
                h1, r1, t1, h2, r2, r22, t2 = torch.chunk(current_triples, 7, dim=1)
                h1 = torch.squeeze(h1, dim=1).cuda()
                r1 = torch.squeeze(r1, dim=1).cuda()
                t1 = torch.squeeze(t1, dim=1).cuda()
                h2 = torch.squeeze(h2, dim=1).cuda()
                r2 = torch.squeeze(r2, dim=1).cuda()
                r22 = torch.squeeze(r22, dim=1).cuda()
                t2 = torch.squeeze(t2, dim=1).cuda()
                dbp1 = self.ent_embedding_1(h1)
                dwy1 = self.ent_embedding_2(h2)
                dbp2 = self.rel_embedding_1(r1)
                dwy2 = self.rel_embedding_2(r2)
                dwy22 = self.rel_embedding_2(r22)
                dbp3 = self.ent_embedding_1(t1)
                dwy3 = self.ent_embedding_2(t2)
                loss = 5 * (self.loss_mse(dbp1, dwy1) + self.loss_mse(dbp2, -dwy2-dwy22) + self.loss_mse(dbp3, dwy3))
                return loss
            if corrupted_triples == 'nnn':
                h1, r1, t1, h2, r2, r22, r222, t2 = torch.chunk(current_triples, 8, dim=1)
                h1 = torch.squeeze(h1, dim=1).cuda()
                r1 = torch.squeeze(r1, dim=1).cuda()
                t1 = torch.squeeze(t1, dim=1).cuda()
                h2 = torch.squeeze(h2, dim=1).cuda()
                r2 = torch.squeeze(r2, dim=1).cuda()
                r22 = torch.squeeze(r22, dim=1).cuda()
                r222 = torch.squeeze(r222, dim=1).cuda()
                t2 = torch.squeeze(t2, dim=1).cuda()
                dbp1 = self.ent_embedding_1(h1)
                dwy1 = self.ent_embedding_2(h2)
                dbp2 = self.rel_embedding_1(r1)
                dwy2 = self.rel_embedding_2(r2)
                dwy22 = self.rel_embedding_2(r22)
                dwy222 = self.rel_embedding_2(r222)
                dbp3 = self.ent_embedding_1(t1)
                dwy3 = self.ent_embedding_2(t2)
                loss = 5 * (self.loss_mse(dbp1, dwy1) + self.loss_mse(dbp2, -dwy2-dwy22-dwy222) + self.loss_mse(dbp3, dwy3))
                return loss
            if corrupted_triples == 'npn':
                h1, r1, t1, h2, r2, r22, r222, t2 = torch.chunk(current_triples, 8, dim=1)
                h1 = torch.squeeze(h1, dim=1).cuda()
                r1 = torch.squeeze(r1, dim=1).cuda()
                t1 = torch.squeeze(t1, dim=1).cuda()
                h2 = torch.squeeze(h2, dim=1).cuda()
                r2 = torch.squeeze(r2, dim=1).cuda()
                r22 = torch.squeeze(r22, dim=1).cuda()
                r222 = torch.squeeze(r222, dim=1).cuda()
                t2 = torch.squeeze(t2, dim=1).cuda()
                dbp1 = self.ent_embedding_1(h1)
                dwy1 = self.ent_embedding_2(h2)
                dbp2 = self.rel_embedding_1(r1)
                dwy2 = self.rel_embedding_2(r2)
                dwy22 = self.rel_embedding_2(r22)
                dwy222 = self.rel_embedding_2(r222)
                dbp3 = self.ent_embedding_1(t1)
                dwy3 = self.ent_embedding_2(t2)
                loss = 5 * (self.loss_mse(dbp1, dwy1) + self.loss_mse(dbp2, -dwy2+dwy22-dwy222) + self.loss_mse(dbp3, dwy3))
                return loss
            if corrupted_triples == 'npp':
                h1, r1, t1, h2, r2, r22, r222, t2 = torch.chunk(current_triples, 8, dim=1)
                h1 = torch.squeeze(h1, dim=1).cuda()
                r1 = torch.squeeze(r1, dim=1).cuda()
                t1 = torch.squeeze(t1, dim=1).cuda()
                h2 = torch.squeeze(h2, dim=1).cuda()
                r2 = torch.squeeze(r2, dim=1).cuda()
                r22 = torch.squeeze(r22, dim=1).cuda()
                r222 = torch.squeeze(r222, dim=1).cuda()
                t2 = torch.squeeze(t2, dim=1).cuda()
                dbp1 = self.ent_embedding_1(h1)
                dwy1 = self.ent_embedding_2(h2)
                dbp2 = self.rel_embedding_1(r1)
                dwy2 = self.rel_embedding_2(r2)
                dwy22 = self.rel_embedding_2(r22)
                dwy222 = self.rel_embedding_2(r222)
                dbp3 = self.ent_embedding_1(t1)
                dwy3 = self.ent_embedding_2(t2)
                loss = 5 * (self.loss_mse(dbp1, dwy1) + self.loss_mse(dbp2, -dwy2+dwy22+dwy222) + self.loss_mse(dbp3, dwy3))
                return loss
            if corrupted_triples == 'pnn':
                h1, r1, t1, h2, r2, r22, r222, t2 = torch.chunk(current_triples, 8, dim=1)
                h1 = torch.squeeze(h1, dim=1).cuda()
                r1 = torch.squeeze(r1, dim=1).cuda()
                t1 = torch.squeeze(t1, dim=1).cuda()
                h2 = torch.squeeze(h2, dim=1).cuda()
                r2 = torch.squeeze(r2, dim=1).cuda()
                r22 = torch.squeeze(r22, dim=1).cuda()
                r222 = torch.squeeze(r222, dim=1).cuda()
                t2 = torch.squeeze(t2, dim=1).cuda()
                dbp1 = self.ent_embedding_1(h1)
                dwy1 = self.ent_embedding_2(h2)
                dbp2 = self.rel_embedding_1(r1)
                dwy2 = self.rel_embedding_2(r2)
                dwy22 = self.rel_embedding_2(r22)
                dwy222 = self.rel_embedding_2(r222)
                dbp3 = self.ent_embedding_1(t1)
                dwy3 = self.ent_embedding_2(t2)
                loss = 5 * (self.loss_mse(dbp1, dwy1) + self.loss_mse(dbp2, dwy2-dwy22-dwy222) + self.loss_mse(dbp3, dwy3))
                return loss
            if corrupted_triples == 'ppn':
                h1, r1, t1, h2, r2, r22, r222, t2 = torch.chunk(current_triples, 8, dim=1)
                h1 = torch.squeeze(h1, dim=1).cuda()
                r1 = torch.squeeze(r1, dim=1).cuda()
                t1 = torch.squeeze(t1, dim=1).cuda()
                h2 = torch.squeeze(h2, dim=1).cuda()
                r2 = torch.squeeze(r2, dim=1).cuda()
                r22 = torch.squeeze(r22, dim=1).cuda()
                r222 = torch.squeeze(r222, dim=1).cuda()
                t2 = torch.squeeze(t2, dim=1).cuda()
                dbp1 = self.ent_embedding_1(h1)
                dwy1 = self.ent_embedding_2(h2)
                dbp2 = self.rel_embedding_1(r1)
                dwy2 = self.rel_embedding_2(r2)
                dwy22 = self.rel_embedding_2(r22)
                dwy222 = self.rel_embedding_2(r222)
                dbp3 = self.ent_embedding_1(t1)
                dwy3 = self.ent_embedding_2(t2)
                loss = 5 * (self.loss_mse(dbp1, dwy1) + self.loss_mse(dbp2, dwy2+dwy22-dwy222) + self.loss_mse(dbp3, dwy3))
                return loss
            if corrupted_triples == 'ppp':
                h1, r1, t1, h2, r2, r22, r222, t2 = torch.chunk(current_triples, 8, dim=1)
                h1 = torch.squeeze(h1, dim=1).cuda()
                r1 = torch.squeeze(r1, dim=1).cuda()
                t1 = torch.squeeze(t1, dim=1).cuda()
                h2 = torch.squeeze(h2, dim=1).cuda()
                r2 = torch.squeeze(r2, dim=1).cuda()
                r22 = torch.squeeze(r22, dim=1).cuda()
                r222 = torch.squeeze(r222, dim=1).cuda()
                t2 = torch.squeeze(t2, dim=1).cuda()
                dbp1 = self.ent_embedding_1(h1)
                dwy1 = self.ent_embedding_2(h2)
                dbp2 = self.rel_embedding_1(r1)
                dwy2 = self.rel_embedding_2(r2)
                dwy22 = self.rel_embedding_2(r22)
                dwy222 = self.rel_embedding_2(r222)
                dbp3 = self.ent_embedding_1(t1)
                dwy3 = self.ent_embedding_2(t2)
                loss = 5 * (self.loss_mse(dbp1, dwy1) + self.loss_mse(dbp2, dwy2+dwy22+dwy222) + self.loss_mse(dbp3, dwy3))
                return loss
        elif train_type == 'entity_align':
            '''
            这里是硬着来的实体对齐
            '''
            dwy, dbpedia = torch.chunk(current_triples, 2, dim=1)
            dwy = torch.squeeze(dwy, dim=1).cuda()
            dbpedia = torch.squeeze(dbpedia, dim=1).cuda()
            dwy_emb = self.ent_embedding_2(dwy)
            dbpedia_emb = self.ent_embedding_1(dbpedia)
            loss = 5 * self.loss_mse(dwy_emb, dbpedia_emb)
            return loss


class Train:
    def __init__(self):
        self.loss = 0

    def init_model(self):
        print('if data vary, remember to rewrite here!')
        self.model = Multi_Model(100000, 302, 100000, 31, 100, 2.0, 1, 0.25)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.005)

    def get_triples(self, file1, file2):
        print('get triples ...')
        self.triples_1, self.triples_2 = self.model.get_triples(file1, file2)
        print('get triples done!')

    def train_relation(self, ent_rel_1, ent_rel_2):
        # 传入[['a', 'b'], 'c'] [['a', 'b'], 'd']这样的即可
        self.optimizer.zero_grad()
        loss = self.model(ent_rel_1, ent_rel_2, 'relation')
        self.loss += loss.item()
        loss.backward()
        self.optimizer.step()

    def train_entity(self, correct_sample, corrupted_sample, entity_1_or_2):
        self.optimizer.zero_grad()
        if entity_1_or_2 == 'entity_1':
            loss = self.model(correct_sample, corrupted_sample, 'entity_1')
        else:
            loss = self.model(correct_sample, corrupted_sample, 'entity_2')
        self.loss += loss.item()
        loss.backward()
        self.optimizer.step()

    def train_entity_align(self, entity_1, entity_2):
        self.optimizer.zero_grad()
        loss = self.model(entity_1, entity_2, 'entity_align')
        self.loss += loss.item()
        loss.backward()
        self.optimizer.step()

    def get_relation_train(self, relation_train_way):
        #  /root/dxh/ccw_workplace/relation_align
        print('loading relation_train data...')
        list_dir = os.listdir(relation_train_way)
        self.relation_train_data = []
        for file in range(len(list_dir)):
            chunk = []
            with codecs.open(f'{relation_train_way}/{list_dir[file]}', 'r', encoding='UTF-8') as f:
                lines = f.readlines()
                for line in lines:
                    if line != '\n':
                        line = line.strip().split()
                        chunk.append(line)
                    else:
                        if chunk:
                            db_ent = [chunk[0][0], chunk[0][2]]
                            db_rel = [chunk[0][1]]
                            dwy_ent = []
                            dwy_rel = []
                            for dwy in chunk[1:]:
                                dwy_ent.append(dwy[0])
                                dwy_ent.append((dwy[2]))
                                dwy_rel.append((dwy[1]))
                            dwy_ent = list(set(dwy_ent))
                            dwy_rel = list(set(dwy_rel))
                            self.relation_train_data.append([[db_ent, db_rel], [dwy_ent, dwy_rel]])
                            chunk = []
        print('loading relation_train data done!')

    def get_entity_train(self, entity_train_way):
        # line[0]是dwy的 line[1]是dbpedia的
        print('get entity train ...')
        # self.entity_train_data = []
        # with codecs.open(entity_train_way, 'r', encoding='UTF-8') as f:
        #     lines = f.readlines()
        #     for line in lines:
        #         line = line.strip().split()
        #         self.entity_train_data.append([line[0], line[1]])
        #
        # f.close()
        self.entity_train_data = self.model.entity_train_data(entity_train_way)
        print('get entity train done!')
        return self.entity_train_data

    def save_weight(self, save_location):
        with codecs.open(save_location + "MTransE_ent_1", "w") as f1:
            for i, e in enumerate(self.model.ent_embedding_1.weight):
                f1.write(str(i) + "\t")
                f1.write(str(e.cpu().detach().numpy().tolist()))
                f1.write("\n")

        with codecs.open(save_location + "MTransE_ent_2", "w") as f1:
            for i, e in enumerate(self.model.ent_embedding_2.weight):
                f1.write(str(i) + "\t")
                f1.write(str(e.cpu().detach().numpy().tolist()))
                f1.write("\n")

        with codecs.open(save_location + "MTransE_rel_1", "w") as f1:
            for i, e in enumerate(self.model.rel_embedding_1.weight):
                f1.write(str(i) + "\t")
                f1.write(str(e.cpu().detach().numpy().tolist()))
                f1.write("\n")

        with codecs.open(save_location + "MTransE_rel_2", "w") as f1:
            for i, e in enumerate(self.model.rel_embedding_2.weight):
                f1.write(str(i) + "\t")
                f1.write(str(e.cpu().detach().numpy().tolist()))
                f1.write("\n")

    def get_aligned_triple1_p(self):
        '''
        这里准备加路径
        '''

        aligned_triple1_p = joblib.load('/root/dxh/ccw_workplace/Archive/aligned_triple1_p.txt')
        self.aligned_triple1_p = []
        for single in aligned_triple1_p:
            tmp = [eval(x) for x in single]
            tmp[0] -= 70000
            tmp[2] -= 70000
            tmp[3] -= 70000
            tmp[4] -= 302
            tmp[-1] -= 70000
            self.aligned_triple1_p.append(tuple(tmp))
        self.aligned_triple1_p = [list(x) for x in list(set(self.aligned_triple1_p))]

        aligned_triple1_p = joblib.load('/root/dxh/ccw_workplace/Archive/aligned_triple1_n.txt')
        self.aligned_triple1_n = []
        for single in aligned_triple1_p:
            tmp = [eval(x) for x in single]
            tmp[0] -= 70000
            tmp[2] -= 70000
            tmp[3] -= 70000
            tmp[4] -= 302
            tmp[-1] -= 70000
            self.aligned_triple1_n.append(tuple(tmp))
        self.aligned_triple1_n = [list(x) for x in list(set(self.aligned_triple1_n))]

        aligned_triple2_pp = joblib.load('/root/dxh/ccw_workplace/Archive/aligned_triple2_pp.txt')
        self.aligned_triple2_pp = []
        for single in aligned_triple2_pp:
            tmp = [eval(x) for x in single]
            tmp[0] -= 70000
            tmp[2] -= 70000
            tmp[3] -= 70000
            tmp[4] -= 302
            tmp[5] -= 302
            tmp[-1] -= 70000
            self.aligned_triple2_pp.append(tuple(tmp))
        self.aligned_triple2_pp = [list(x) for x in list(set(self.aligned_triple2_pp))]

        aligned_triple2_pn = joblib.load('/root/dxh/ccw_workplace/Archive/aligned_triple2_pn.txt')
        self.aligned_triple2_pn = []
        for single in aligned_triple2_pn:
            if single[4] == single[5]:
                continue
            tmp = [eval(x) for x in single]
            tmp[0] -= 70000
            tmp[2] -= 70000
            tmp[3] -= 70000
            tmp[4] -= 302
            tmp[5] -= 302
            tmp[-1] -= 70000
            self.aligned_triple2_pn.append(tuple(tmp))
        self.aligned_triple2_pn = [list(x) for x in list(set(self.aligned_triple2_pn))]

        aligned_triple2_np = joblib.load('/root/dxh/ccw_workplace/Archive/aligned_triple2_np.txt')
        self.aligned_triple2_np = []
        for single in aligned_triple2_np:
            if single[4] == single[5]:
                continue
            tmp = [eval(x) for x in single]
            tmp[0] -= 70000
            tmp[2] -= 70000
            tmp[3] -= 70000
            tmp[4] -= 302
            tmp[5] -= 302
            tmp[-1] -= 70000
            self.aligned_triple2_np.append(tuple(tmp))
        self.aligned_triple2_np = [list(x) for x in list(set(self.aligned_triple2_np))]

        aligned_triple2_nn = joblib.load('/root/dxh/ccw_workplace/Archive/aligned_triple2_nn.txt')
        self.aligned_triple2_nn = []
        for single in aligned_triple2_nn:
            tmp = [eval(x) for x in single]
            tmp[0] -= 70000
            tmp[2] -= 70000
            tmp[3] -= 70000
            tmp[4] -= 302
            tmp[5] -= 302
            tmp[-1] -= 70000
            self.aligned_triple2_nn.append(tuple(tmp))
        self.aligned_triple2_nn = [list(x) for x in list(set(self.aligned_triple2_nn))]

        aligned_triple3_nnn = joblib.load('/root/dxh/ccw_workplace/Archive/aligned_triple3_nnn.txt')
        self.aligned_triple3_nnn = []
        for single in aligned_triple3_nnn:
            tmp = [eval(x) for x in single]
            tmp[0] -= 70000
            tmp[2] -= 70000
            tmp[3] -= 70000
            tmp[4] -= 302
            tmp[5] -= 302
            tmp[6] -= 302
            tmp[-1] -= 70000
            self.aligned_triple3_nnn.append(tuple(tmp))
        self.aligned_triple3_nnn = [list(x) for x in list(set(self.aligned_triple3_nnn))]

        # 这个是0
        aligned_triple3_nnp = joblib.load('/root/dxh/ccw_workplace/Archive/aligned_triple3_nnp.txt')
        self.aligned_triple3_nnp = []
        for single in aligned_triple3_nnp:
            tmp = [eval(x) for x in single]
            tmp[0] -= 70000
            tmp[2] -= 70000
            tmp[3] -= 70000
            tmp[4] -= 302
            tmp[5] -= 302
            tmp[6] -= 302
            tmp[-1] -= 70000
            self.aligned_triple3_nnp.append(tuple(tmp))
        self.aligned_triple3_nnp = [list(x) for x in list(set(self.aligned_triple3_nnp))]

        aligned_triple3_npn = joblib.load('/root/dxh/ccw_workplace/Archive/aligned_triple3_npn.txt')
        self.aligned_triple3_npn = []
        for single in aligned_triple3_npn:
            tmp = [eval(x) for x in single]
            tmp[0] -= 70000
            tmp[2] -= 70000
            tmp[3] -= 70000
            tmp[4] -= 302
            tmp[5] -= 302
            tmp[6] -= 302
            tmp[-1] -= 70000
            self.aligned_triple3_npn.append(tuple(tmp))
        self.aligned_triple3_npn = [list(x) for x in list(set(self.aligned_triple3_npn))]

        aligned_triple3_npp = joblib.load('/root/dxh/ccw_workplace/Archive/aligned_triple3_npp.txt')
        self.aligned_triple3_npp = []
        for single in aligned_triple3_npp:
            tmp = [eval(x) for x in single]
            tmp[0] -= 70000
            tmp[2] -= 70000
            tmp[3] -= 70000
            tmp[4] -= 302
            tmp[5] -= 302
            tmp[6] -= 302
            tmp[-1] -= 70000
            self.aligned_triple3_npp.append(tuple(tmp))
        self.aligned_triple3_npp = [list(x) for x in list(set(self.aligned_triple3_npp))]

        aligned_triple3_pnn = joblib.load('/root/dxh/ccw_workplace/Archive/aligned_triple3_pnn.txt')
        self.aligned_triple3_pnn = []
        for single in aligned_triple3_pnn:
            tmp = [eval(x) for x in single]
            tmp[0] -= 70000
            tmp[2] -= 70000
            tmp[3] -= 70000
            tmp[4] -= 302
            tmp[5] -= 302
            tmp[6] -= 302
            tmp[-1] -= 70000
            self.aligned_triple3_pnn.append(tuple(tmp))
        self.aligned_triple3_pnn = [list(x) for x in list(set(self.aligned_triple3_pnn))]

        # 这个是0
        aligned_triple3_pnp = joblib.load('/root/dxh/ccw_workplace/Archive/aligned_triple3_pnp.txt')
        self.aligned_triple3_pnp = []
        for single in aligned_triple3_pnp:
            tmp = [eval(x) for x in single]
            tmp[0] -= 70000
            tmp[2] -= 70000
            tmp[3] -= 70000
            tmp[4] -= 302
            tmp[5] -= 302
            tmp[6] -= 302
            tmp[-1] -= 70000
            self.aligned_triple3_pnp.append(tuple(tmp))
        self.aligned_triple3_pnp = [list(x) for x in list(set(self.aligned_triple3_pnp))]

        aligned_triple3_ppn = joblib.load('/root/dxh/ccw_workplace/Archive/aligned_triple3_ppn.txt')
        self.aligned_triple3_ppn = []
        for single in aligned_triple3_ppn:
            tmp = [eval(x) for x in single]
            tmp[0] -= 70000
            tmp[2] -= 70000
            tmp[3] -= 70000
            tmp[4] -= 302
            tmp[5] -= 302
            tmp[6] -= 302
            tmp[-1] -= 70000
            self.aligned_triple3_ppn.append(tuple(tmp))
        self.aligned_triple3_ppn = [list(x) for x in list(set(self.aligned_triple3_ppn))]

        aligned_triple3_ppp = joblib.load('/root/dxh/ccw_workplace/Archive/aligned_triple3_ppp.txt')
        self.aligned_triple3_ppp = []
        for single in aligned_triple3_ppp:
            tmp = [eval(x) for x in single]
            tmp[0] -= 70000
            tmp[2] -= 70000
            tmp[3] -= 70000
            tmp[4] -= 302
            tmp[5] -= 302
            tmp[6] -= 302
            tmp[-1] -= 70000
            self.aligned_triple3_ppp.append(tuple(tmp))
        self.aligned_triple3_ppp = [list(x) for x in list(set(self.aligned_triple3_ppp))]


    def training_run_relation_MtransE(self, ent1_way, enti2_way, relation_train_way, entity_train_way,
                                      epochs=300,
                                      model_type='MtransE_relation',
                                      batch_size=400,
                                      out_file_title='/root/dxh/ccw_workplace/save_weights/'):

        print('begin training...')
        self.get_triples(ent1_way, enti2_way)
        if model_type == 'MtransE' or model_type == 'MtransE_relation':
            self.get_entity_train(entity_train_way)
        if model_type == 'MtransE_relation':
            self.get_relation_train(relation_train_way)
            '''
            这里准备改
            '''
            self.get_aligned_triple1_p()


        for epoch_ in range(epochs):
            self.loss = 0
            start = time.time()
            print(f"epoch {epoch_} begins ...")
            batch_size_1 = 6000
            n_batches_1 = int(len(self.triples_1) / batch_size_1)

            epoch_11 = 10
            print("the number of batches_1: ", n_batches_1)
            for epoch in range(epoch_11):

                for batch in range(1):
                    batch_samples = random.sample(self.triples_1, batch_size_1)

                    current = []
                    corrupted = []
                    for sample in batch_samples:
                        corrupted_sample = copy.deepcopy(sample)
                        pr = np.random.random(1)[0]

                        if pr < 0.5:
                            corrupted_sample[0] = random.sample(self.model.entities_1, 1)[0]
                            while corrupted_sample[0] == sample[0]:
                                corrupted_sample[0] = random.sample(self.model.entities_1, 1)[0]
                        else:
                            corrupted_sample[2] = random.sample(self.model.entities_1, 1)[0]
                            while corrupted_sample[2] == sample[2]:
                                corrupted_sample[2] = random.sample(self.model.entities_1, 1)[0]

                        current.append(sample)
                        corrupted.append(corrupted_sample)
                    current = torch.from_numpy(np.array(current)).long()
                    corrupted = torch.from_numpy(np.array(corrupted)).long()
                    self.train_entity(current, corrupted, 'entity_1')
            print('train kg_1', 'done!')

            batch_size_2 = 6000
            n_batches_2 = int(len(self.triples_2) / batch_size_2)
            epoch_12 = 10

            print("the number of batches_2: ", n_batches_2)
            for epoch in range(epoch_12):

                for batch in range(n_batches_2):

                    batch_samples = random.sample(self.triples_2, batch_size_2)

                    current = []
                    corrupted = []
                    for sample in batch_samples:
                        corrupted_sample = copy.deepcopy(sample)
                        pr = np.random.random(1)[0]

                        if pr < 0.5:
                            corrupted_sample[0] = random.sample(self.model.entities_2, 1)[0]
                            while corrupted_sample[0] == sample[0]:
                                corrupted_sample[0] = random.sample(self.model.entities_2, 1)[0]
                        else:
                            corrupted_sample[2] = random.sample(self.model.entities_2, 1)[0]
                            while corrupted_sample[2] == sample[2]:
                                corrupted_sample[2] = random.sample(self.model.entities_2, 1)[0]

                        current.append(sample)
                        corrupted.append(corrupted_sample)
                    current = torch.from_numpy(np.array(current)).long()
                    corrupted = torch.from_numpy(np.array(corrupted)).long()
                    self.train_entity(current, corrupted, 'entity_2')
            print('train kg_2', 'done!')

            if model_type == 'MtransE' or model_type == 'MtransE_relation':
                batch_siize = 1500
                n_batches_3 = int(len(self.entity_train_data) / batch_siize)
                print("the number of n_batches_3: ", n_batches_3)

                epochs_1 = 10
                for epoch in range(epochs_1 * n_batches_3):
                    training_entity_sample = random.sample(self.entity_train_data, batch_siize)
                    training_entity_sample = torch.from_numpy(np.array(training_entity_sample)).long()
                    self.train_entity_align(training_entity_sample, 'whatever')

                print('train entity seed', 'done!')

            if model_type == 'MtransE_relation':
                # train_relation
                epochs_2 = 50
                for epoch in range(epochs_2):

                    b_size = 5000
                    n_b_size = int(len(self.aligned_triple1_p) / b_size)
                    for i in range(n_b_size):
                        training_entity_sample = random.sample(self.aligned_triple1_p, b_size)
                        training_entity_sample = torch.from_numpy(np.array(training_entity_sample)).long()
                        self.train_relation(training_entity_sample, 'p')

                    b_size_1 = 1000
                    n_b_size_1 = int(len(self.aligned_triple1_n) / b_size_1)
                    for i in range(n_b_size_1):
                        training_entity_sample = random.sample(self.aligned_triple1_n, b_size_1)
                        training_entity_sample = torch.from_numpy(np.array(training_entity_sample)).long()
                        self.train_relation(training_entity_sample, 'n')

                    tmp = torch.from_numpy(np.array(self.aligned_triple2_nn)).long()
                    self.train_relation(tmp, 'nn')

                    tmp = torch.from_numpy(np.array(self.aligned_triple2_np)).long()
                    self.train_relation(tmp, 'np')

                    tmp = torch.from_numpy(np.array(self.aligned_triple2_pn)).long()
                    self.train_relation(tmp, 'pn')

                    b_size_1 = 1000
                    n_b_size_1 = int(len(self.aligned_triple2_pp) / b_size_1)
                    for i in range(n_b_size_1):
                        training_entity_sample = random.sample(self.aligned_triple2_pp, b_size_1)
                        training_entity_sample = torch.from_numpy(np.array(training_entity_sample)).long()
                        self.train_relation(training_entity_sample, 'pp')

                    tmp = torch.from_numpy(np.array(self.aligned_triple3_nnn)).long()
                    self.train_relation(tmp, 'nnn')

                    tmp = torch.from_numpy(np.array(self.aligned_triple3_npn)).long()
                    self.train_relation(tmp, 'npn')

                    tmp = torch.from_numpy(np.array(self.aligned_triple3_npp)).long()
                    self.train_relation(tmp, 'npp')

                    tmp = torch.from_numpy(np.array(self.aligned_triple3_pnn)).long()
                    self.train_relation(tmp, 'pnn')

                    tmp = torch.from_numpy(np.array(self.aligned_triple3_ppn)).long()
                    self.train_relation(tmp, 'ppn')

                    b_size_1 = 1000
                    n_b_size_1 = int(len(self.aligned_triple3_ppp) / b_size_1)
                    for i in range(n_b_size_1):
                        training_entity_sample = random.sample(self.aligned_triple3_ppp, b_size_1)
                        training_entity_sample = torch.from_numpy(np.array(training_entity_sample)).long()
                        self.train_relation(training_entity_sample, 'ppp')

                print('train relation seed', 'done!')

                end = time.time()
                print(f"epoch {epoch_} consuming {end - start} s and the loss is {self.loss}")

            end = time.time()
            # writing log
            with open(out_file_title + 'training_log.txt', 'a') as f:
                f.write(f"epoch {epoch_} consuming {end - start}s loss is {self.loss}")
                f.write('\n')

        print('saving weight to ', out_file_title, '...')
        self.save_weight(out_file_title)
        print('saving weight done!')

        print('all training done!')


if __name__ == '__main__':
    file1 = "/root/dxh/ccw_workplace/final/dbpedia_entity2id.txt"
    file2 = "/root/dxh/ccw_workplace/final/dwy_entity2id.txt"
    file3 = "/root/dxh/ccw_workplace/final/dbpedia_relation2id.txt"
    file4 = "/root/dxh/ccw_workplace/final/dwy_relation2id.txt"
    TRANSE = Train()
    TRANSE.init_model()
    TRANSE.model.prepare_data(file1, file2, file3, file4)
    TRANSE.training_run_relation_MtransE("/root/dxh/ccw_workplace/final/dbpedia_train.txt", "/root/dxh/ccw_workplace/final/dwy_train.txt",
                                         '/root/dxh/ccw_workplace/final/result', '/root/dxh/ccw_workplace/final/entity_align_train.txt',
                                         model_type='MtransE_relation',
                                         out_file_title='/root/dxh/ccw_workplace/final/new_300/')

    a = Entity_Measure("/root/dxh/ccw_workplace/final/dbpedia_entity2id.txt",
                       "/root/dxh/ccw_workplace/final/dwy_entity2id.txt",
                       "/root/dxh/ccw_workplace/final/new_300/MTransE_ent_1",
                       "/root/dxh/ccw_workplace/final/new_300/MTransE_ent_2")
    a.load_dic()
    a.load_vec()
    a.calculate_all_multi('/root/dxh/ccw_workplace/final/entity_align_test.txt',
                          outputfile='/root/dxh/ccw_workplace/final/test_result_norelation/')

