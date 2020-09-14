"""
对SequenceToSequence模型进行基本的参数组合测试
"""

import sys
import random
import pickle

import numpy as np
import tensorflow as tf
import jieba
from opencc import OpenCC
from change_meal_name import replace_meal_line
from collections import defaultdict
# from nltk.tokenize import word_tokenize

sys.path.append('..')


def test(bidirectional, cell_type, depth,
         attention_type, use_residual, use_dropout, time_major, hidden_units, anti):
    """测试不同参数在生成的假数据上的运行结果"""

    from sequence_to_sequence import SequenceToSequence
    from data_utils import batch_flow
    from word_sequence import WordSequence  # pylint: disable=unused-variable

    x_data, _, ws = pickle.load(open('./pickle/chatbot.pkl', 'rb'))
    '''
    for x in x_data[:5]:
        print(' '.join(x))
    '''
    config = tf.ConfigProto(
        device_count={'CPU': 1, 'GPU': 0},
        allow_soft_placement=True,
        log_device_placement=False
    )

    if anti is False:
        save_path = './pickle/s2ss_chatbot.ckpt'
    else:
        save_path = './pickle/s2ss_chatbot_anti.ckpt'

    # 测试部分
    tf.reset_default_graph()
    model_pred = SequenceToSequence(
        input_vocab_size=len(ws),
        target_vocab_size=len(ws),
        batch_size=1,
        mode='decode',
        beam_width=0,
        bidirectional=bidirectional,
        cell_type=cell_type,
        depth=depth,
        attention_type=attention_type,
        use_residual=use_residual,
        use_dropout=use_dropout,
        parallel_iterations=1,
        time_major=time_major,
        hidden_units=hidden_units,
        share_embedding=True,
        pretrained_embedding=True
    )
    init = tf.global_variables_initializer()

    fp = open('meals.txt', 'r')
    meals = fp.readlines()
    order_table = defaultdict(list) # 紀錄所有點餐

    with tf.Session(config=config) as sess:
        sess.run(init)
        model_pred.load(sess, save_path)

        while True:
            order = [] # 紀錄當次點餐
            no_order_ans = ['咖啡', '紅茶', '綠茶'] # 當使用者沒有輸入餐點回答的預設餐點
            user_text = input()
            tmp = user_text.split() # 分開使用者id與問題
            user_id = tmp[0]
            user_text = tmp[1]
            if user_text in ('exit', 'quit'):
                exit(0)
            elif user_text == '#menu':
                print(str(order_table[user_id]) + ' 零。')
                continue
            elif user_text == '#delete':
                del order_table[user_id]
                continue
            s2tw = OpenCC('s2tw')
            user_text = s2tw.convert(user_text)

            order_tmp, user_text = replace_meal_line(user_text, meals)

            for line in iter(order_tmp):
                order.append(line)
            x_test = [jieba.lcut(user_text.lower())]

            bar = batch_flow([x_test], ws, 1)
            x, xl = next(bar)
            x = np.flip(x, axis=1)

            pred = model_pred.predict(
                sess,
                np.array(x),
                np.array(xl)
            )
            s2tw = OpenCC('s2tw')
            out = filter(lambda ch: ch not in '</s><unk>',
                         ws.inverse_transform(pred[0]))

            out = list(out)
            for i, _ in enumerate(out):
                if out[i] == 'allkindofmeal':
                    if not order:
                        out[i] = no_order_ans[0]
                        del no_order_ans[0]
                    else:
                        out[i] = order[0]
                        order_table[user_id].append(order[0])
                        del order[0]
            order_table[user_id].extend(order)
            out = ''.join(list(out))
            print(out)


def main():
    random.seed(0)
    np.random.seed(0)
    tf.set_random_seed(0)
    test(
        bidirectional=True,
        cell_type='lstm',
        depth=4,
        attention_type='Bahdanau',
        use_residual=True,
        use_dropout=True,
        time_major=False,
        hidden_units=256,
        anti = True
    )


if __name__ == '__main__':
    main()
