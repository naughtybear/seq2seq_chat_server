"""
利用字典，將向量轉成字或是將字轉成向量

"""


import numpy as np


class WordSequence(object):
    """一个可以把句子轉為index的class
    """

    PAD_TAG = '<pad>'
    UNK_TAG = '<unk>'
    START_TAG = '<s>'
    END_TAG = '</s>'
    PAD = 0
    UNK = 1
    START = 2
    END = 3


    def __init__(self):
        """初始化基本的dict
        """
        self.dict = {
            WordSequence.PAD_TAG: WordSequence.PAD,
            WordSequence.UNK_TAG: WordSequence.UNK,
            WordSequence.START_TAG: WordSequence.START,
            WordSequence.END_TAG: WordSequence.END,
        }
        self.fited = False


    def to_index(self, word):
        """把一個單字轉換為index
        """
        assert self.fited, 'WordSequence 尚未 fit'
        if word in self.dict:
            return self.dict[word]
        return WordSequence.UNK


    def to_word(self, index):
        """把一個index轉會為單字
        """
        assert self.fited, 'WordSequence 尚未 fit'
        for k, v in self.dict.items():
            if v == index:
                return k
        return WordSequence.UNK_TAG


    def size(self):
        """回傳字典大小
        """
        assert self.fited, 'WordSequence 尚未 fit'
        return len(self.dict) + 1

    def __len__(self):
        """回傳字典大小
        """
        return self.size()


    def fit(self, sentences, min_count=1, max_count=None, max_features=None):
        """訓練 WordSequence
        input:
            min_count 最小出現次數
            max_count 最大出現次數
            max_features 最大特征數

        ws = WordSequence()
        ws.fit([['hello', 'world']])
        """
        assert not self.fited, 'WordSequence 只能 fit 一次'

        count = {}
        for sentence in sentences:
            arr = list(sentence)
            for a in arr:
                if a not in count:
                    count[a] = 0
                count[a] += 1

        if min_count is not None:
            count = {k: v for k, v in count.items() if v >= min_count}

        if max_count is not None:
            count = {k: v for k, v in count.items() if v <= max_count}

        self.dict = {
            WordSequence.PAD_TAG: WordSequence.PAD,
            WordSequence.UNK_TAG: WordSequence.UNK,
            WordSequence.START_TAG: WordSequence.START,
            WordSequence.END_TAG: WordSequence.END,
        }

        if isinstance(max_features, int):
            count = sorted(list(count.items()), key=lambda x: x[1])
            if max_features is not None and len(count) > max_features:
                count = count[-int(max_features):]
            for w, _ in count:
                self.dict[w] = len(self.dict)
        else:
            for w in sorted(count.keys()):
                self.dict[w] = len(self.dict)

        self.fited = True


    def transform(self,
                  sentence, max_len=None):
        """把句子轉換為向量
        例如輸入 ['a', 'b', 'c']
        輸出 [1, 2, 3] 這個數字是字典裡的編號，顺序没有意義
        """
        assert self.fited, 'WordSequence 尚未 fit'

        # if max_len is not None:
        #     r = [self.PAD] * max_len
        # else:
        #     r = [self.PAD] * len(sentence)

        if max_len is not None:
            r = [self.PAD] * max_len
        else:
            r = [self.PAD] * len(sentence)

        for index, a in enumerate(sentence):
            if max_len is not None and index >= len(r):
                break
            r[index] = self.to_index(a)

        return np.array(r)


    def inverse_transform(self, indices,
                          ignore_pad=False, ignore_unk=False,
                          ignore_start=False, ignore_end=False):
        """把向量轉換為句子，和上面的相反
        """
        ret = []
        for i in indices:
            word = self.to_word(i)
            if word == WordSequence.PAD_TAG and ignore_pad:
                continue
            if word == WordSequence.UNK_TAG and ignore_unk:
                continue
            if word == WordSequence.START_TAG and ignore_start:
                continue
            if word == WordSequence.END_TAG and ignore_end:
                continue
            ret.append(word)

        return ret
    
    def printdict(self):
        f = open('./data/dict.txt','w')
        for key, _ in self.dict.items():
            f.write(key)
            f.write('\n')


def test():
    """測試
    """
    ws = WordSequence()
    ws.fit([
        ['第', '一', '句', '話'],
        ['第', '二', '句', '話']
    ])

    indice = ws.transform(['第', '三'])
    print(indice)

    back = ws.inverse_transform(indice)
    print(back)

if __name__ == '__main__':
    test()
