"""
讀取一個.vec檔， .vec檔是個已經訓練好的embedding文件

預設使用的檔案: tw_chinese.vec

它的第一行會被忽略
第二行開始，每行是 詞 + 空格 + 詞向量维度0 + 空格 + 詞向量维度1 + ...

参考fasttext的文本格式

https://github.com/facebookresearch/fastText/blob/master/docs/pretrained-vectors.md
"""

import pickle
import numpy as np
from tqdm import tqdm


def read_vector(path='./data/tw_chinese.vec', output_path='./pickle/word_vec.pkl'):
    """
    讀取 path中的 .vec檔，並將其整理成一個 dict並寫入到 output_path中，方便之後模型與資料前處理使用

    格式：
    word_vec = {
        'word_1': np.array(vec_of_word_1),
        'word_2': np.array(vec_of_word_2),
        ...
    }
    """
    fp = open(path, 'r')
    word_vec = {}
    # first_skip = False
    dim = None
    for line in tqdm(fp):
        line = line.strip()
        line = line.split(' ')
        if len(line) >= 2:
            word = line[0]
            vec_text = line[1:]
            vec = np.array([float(v) for v in vec_text])
            word_vec[word] = vec
            if dim is None:
                dim = vec.shape

    np.random.seed(0)
    # PADDING_TAG
    word_vec['<pad>'] = np.random.random_sample(size=(300,)) - 0.5
    # START_TAG
    word_vec['<s>'] = np.random.random_sample(size=(300,)) - 0.5
    # END_TAG
    word_vec['</s>'] = np.random.random_sample(size=(300,)) - 0.5
    # UNKNOWN_TAG
    word_vec['<unk>'] = np.random.random_sample(size=(300,)) - 0.5
    # MEAL_TAG
    word_vec['allkindofmeal'] = np.random.random_sample(size=(300,)) - 0.5

    pickle.dump(word_vec, open(output_path, 'wb'))

if __name__ == '__main__':
    read_vector()
