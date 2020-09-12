#! -*- coding: utf-8 -*-
# 测试代码可用性: 提取特征

from bert4keras.backend import keras
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import to_array
import numpy as np
from config.pathconfig.dirconfig import roberta_dir,data_dir,model_dir

def cosin_distance(vector1, vector2):
    dot_product = 0.0
    normA = 0.0
    normB = 0.0
    for a, b in zip(vector1, vector2):
        dot_product += a * b
        normA += a ** 2
        normB += b ** 2
    if normA == 0.0 or normB == 0.0:
        return None
    else:
        return dot_product / ((normA * normB) ** 0.5)

# config_path = '../config/bert/chinese_L-12_H-768_A-12/bert_config.json'
# checkpoint_path = '/root/kg/bert/chinese_L-12_H-768_A-12/bert_model.ckpt'
# dict_path = '/root/kg/bert/chinese_L-12_H-768_A-12/vocab.txt'
config_path = roberta_dir+'/bert_config.json'
checkpoint_path = roberta_dir+'/bert_model.ckpt'
dict_path = roberta_dir+'/vocab.txt'

tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器
vec_model = build_transformer_model(config_path, checkpoint_path)  # 建立模型，加载权重

def toids(s):
    token_ids, segment_ids = tokenizer.encode(s)
    token_ids, segment_ids = to_array([token_ids], [segment_ids])
    return [token_ids,segment_ids]
# 编码测试
# token_ids, segment_ids = tokenizer.encode(u'姚明的身高是多少')
# token_ids, segment_ids = to_array([token_ids], [segment_ids])
#
# print('\n ===== predicting =====\n')
# a = model.predict([token_ids, segment_ids])


# print(model.predict([token_ids, segment_ids]))
# token_ids, segment_ids = tokenizer.encode(u'身高')
# token_ids, segment_ids = to_array([token_ids], [segment_ids])
# b = model.predict([token_ids, segment_ids])
# token_ids, segment_ids = tokenizer.encode(u'体重')
# token_ids, segment_ids = to_array([token_ids], [segment_ids])
# c = model.predict([token_ids, segment_ids])
# token_ids, segment_ids = tokenizer.encode(u'前队友')
# token_ids, segment_ids = to_array([token_ids], [segment_ids])
# d = model.predict([token_ids, segment_ids])
# token_ids, segment_ids = tokenizer.encode(u'主要奖项')
# token_ids, segment_ids = to_array([token_ids], [segment_ids])
# e = model.predict([token_ids, segment_ids])

# print(cosin_distance(a.mean(axis = 1)[0],e.mean(axis = 1)[0]))
"""
输出：
[[[-0.63251007  0.2030236   0.07936534 ...  0.49122632 -0.20493352
    0.2575253 ]
  [-0.7588351   0.09651865  1.0718756  ... -0.6109694   0.04312154
    0.03881441]
  [ 0.5477043  -0.792117    0.44435206 ...  0.42449304  0.41105673
    0.08222899]
  [-0.2924238   0.6052722   0.49968526 ...  0.8604137  -0.6533166
    0.5369075 ]
  [-0.7473459   0.49431565  0.7185162  ...  0.3848612  -0.74090636
    0.39056838]
  [-0.8741375  -0.21650358  1.338839   ...  0.5816864  -0.4373226
    0.56181806]]]
"""

# print('\n ===== reloading and predicting =====\n')
# # model.save('test.model')
# del model
# model = keras.models.load_model('test.model')
# print(model.predict([token_ids, segment_ids]))
