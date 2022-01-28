import gensim
import numpy as np

# 加载预训练模型，注意路径
print('start load vec model')
model = gensim.models.KeyedVectors.load_word2vec_format(
    r'./language_model/GoogleNews-vectors-negative300.bin',
    binary=True)
print('finish load vec model')


def get_vector_split(input_str):
    global model
    """
    :param input_str: 输入一个单词
    :return: 返回词向量，如果单词本身有词向量就直接返回，没有就切成两个计算平均值后返回，如果无法切就返回None
    """
    result = None
    length = len(input_str)
    for i in range(length):
        if i == 0:
            try:
                result = model.get_vector(input_str)
                #print(f"word is {input_str}")
                break
            except:
                pass
        head = input_str[:i]
        tail = input_str[i:]
        try:
            head_vec = model.get_vector(head)
            tail_vec = model.get_vector(tail)
            #print(f"head word is {head}, tail word is {tail}")
            result = np.array([head_vec, tail_vec])
            result = np.average(result, axis=0)
            break
        except:
            pass

    return result


def get_cos_similar(v1, v2):
    num = float(np.dot(v1, v2))  # 向量点乘
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)  # 求模长的乘积
    return 0.5 + 0.5 * (num / denom) if denom != 0 else 0


def get_cos(word_list1, word_list2):
    # print(word_list1, word_list2)
    global model
    vec1 = [0 for x in word_list1]
    for idx, word in enumerate(word_list1):
        try:
            vec1[idx] = get_vector_split(word)
        except:
            pass
    vec1_without_None = [item for item in vec1 if item is not None]
    vec_1_result = np.array(vec1_without_None, dtype=object)
    vec_1_result = np.average(vec_1_result, axis=0)

    vec2 = [0 for x in word_list2]
    for idx, word in enumerate(word_list2):
        try:
            vec2[idx] = get_vector_split(word)
        except:
            pass
    vec2_without_None = [item for item in vec2 if item is not None]
    vec2 = vec2_without_None
    if vec1_without_None == [] or vec2_without_None == []:
        return 0
    vec_2_result = np.array(vec2, dtype=object)
    vec_2_result = np.average(vec_2_result, axis=0)
    # print(vec_1_result, vec_2_result)
    return get_cos_similar(vec_1_result, vec_2_result)


if __name__ == '__main__':
    a = ['residence']
    b = ['lives', 'In', 'is', 'Located', 'In', 'is', 'Located', 'In']

    print(get_cos(a, b))
