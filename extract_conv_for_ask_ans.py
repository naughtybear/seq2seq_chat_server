"""把 word_vec.pkl轉換成可以訓練的格式
"""

import re
import sys
import pickle
import os
import jieba
import numpy as np
from tqdm import tqdm


sys.path.append('..')


def make_split(line):
    """合併兩個句子之間的符號
    """
    if re.match(r'.*([，。…？！～\.,!?])$', ''.join(line)):
        return []
    return ['，']


def good_line(line):
    """判斷一個句子是否好"""
    if len(re.findall(r'[a-zA-Z0-9]', ''.join(line))) > 2:
        return False
    return True


def regular(sen):
    """整理句子"""
    sen = re.sub(r'\.{3,100}', '…', sen)
    sen = re.sub(r'…{2,100}', '…', sen)
    sen = re.sub(r'[,]{1,100}', '，', sen)
    sen = re.sub(r'[\.]{1,100}', '。', sen)
    sen = re.sub(r'[\?]{1,100}', '？', sen)
    sen = re.sub(r'[!]{1,100}', '！', sen)
    return sen


def _ishan(text):
    # for python 3.x
    # sample: ishan('一') == True, ishan('我&&你') == False
    return all('\u4e00' <= char <= '\u9fff' for char in text)


def main(limit=20, x_limit=3, y_limit=6):
    """
    Args:
        limit: 只輸出長度小於limit的句子
    """
    from word_sequence import WordSequence
    print('load pretrained vec')
    word_vec = pickle.load(open('./pickle/word_vec.pkl', 'rb'))

    print('extract lines')
    fp = open('./data/replaced_data.txt', 'r', errors='ignore')
    # last_line = None
    groups = []
    group = []
    for line in tqdm(fp):
        if line.startswith('M '):
            line = line.replace('\n', '')
            if '/' in line:
                line = line[2:].split('/')
            else:
                line = line[2:]

            outline = jieba.lcut(regular(''.join(line)))

            group.append(outline)
        else:  # if line.startswith('E'):
            last_line = None
            if group:
                groups.append(group)
                group = []
    if group:
        groups.append(group)
        group = []
    print('extract groups')
    x_data = []
    y_data = []
    for group in tqdm(groups):
        for i, line in enumerate(group):
            next_line = None
            if i + 1 >= len(group):
                continue
            if i % 2 == 0:
                next_line = group[i + 1]

            if next_line:
                x_data.append(line)
                y_data.append(next_line)

    x_f = open('./data/x_data.txt', 'w')
    y_f = open('./data/y_data.txt', 'w')
    for i in range(len(x_data)-1):
        # x_line = x_data[i]
        # x_line = x_line[:-2]
        x_out = ''.join(list(x_data[i]))
        y_out = ''.join(list(y_data[i]))
        x_f.write(x_out+'\n')
        y_f.write(y_out+'\n')
    print(len(x_data), len(y_data))
    # exit()
    for ask, answer in zip(x_data[:20], y_data[:20]):
        print(''.join(ask))
        print(''.join(answer))
        print('-' * 20)

    data = list(zip(x_data, y_data))
    data = [
        (x, y)
        for x, y in data
        if len(x) < limit
        and len(y) < limit
        and len(y) >= y_limit
        and len(x) >= x_limit
    ]
    x_data, y_data = zip(*data)

    print('refine train data')

    train_data = x_data + y_data

    print('fit word_sequence')

    ws_input = WordSequence()

    ws_input.fit(train_data, max_features=100000)

    print('dump word_sequence')

    pickle.dump(
        (x_data, y_data, ws_input),
        open('./pickle/chatbot.pkl', 'wb')
    )

    print('make embedding vecs')

    emb = np.zeros((len(ws_input), len(word_vec['</s>'])))

    np.random.seed(1)
    for word, ind in ws_input.dict.items():
        if word in word_vec:
            emb[ind] = word_vec[word]
        else:
            emb[ind] = np.random.random_sample(size=(300,)) - 0.5

    print('dump emb')

    pickle.dump(
        emb,
        open('./pickle/emb.pkl', 'wb')
    )

    print('done')


if __name__ == '__main__':
    main()
