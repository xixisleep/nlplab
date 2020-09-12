# -*- coding: utf-8 -*-
from django.shortcuts import render
from toolkit.basic_extract_features import cosin_distance,vec_model,toids
import requests


def mention2entity(mention):
    '''
     * mention2entity - 提及->实体
     * @mention: 	[in]提及
     * 根据提及获取歧义关系
    '''
    url = 'https://api.ownthink.com/kg/ambiguous?mention={mention}'.format(mention=mention)  # 知识图谱API，歧义关系
    sess = requests.get(url)  # 请求
    text = sess.text  # 获取返回的数据
    entitys = eval(text)  # 转为字典类型
    return entitys

def entity2knowledge(entity):
    '''
     * entity2knowledge - 实体->知识
     * @entity: 	[in]实体名
     * 根据实体获取实体知识
    '''
    url = 'https://api.ownthink.com/kg/knowledge?entity={entity}'.format(entity=entity)  # 知识图谱API，实体知识
    sess = requests.get(url)  # 请求
    text = sess.text  # 获取返回的数据
    knowledge = eval(text)  # 转为字典类型
    return knowledge


def entity_attribute2value(entity, attribute):
    '''
     * entity_attribute2value - 实体&属性->属性值
     * @entity: 	[in]实体名
     * @attribute:	[in]属性名
     * 根据实体、属性获取属性值
    '''
    knowledge = entity2knowledge(entity)
    avp = knowledge['data']['avp']
    if avp != []:
        for a in avp:
            if attribute == a[0]:
                return a[1]
    else:
        return '空值'


def kgqa(request):
    ctx = {}
    ctx['question'] = "姚明身高多少啊？"
    return render(request, 'kgqa.html', ctx)

def kgqa_question(request):
    ctx = {}
    ctx['question'] = "姚明身高多少啊？"
    if request.POST:
        # 获取输入文本
        key = request.POST['user_text']
        # 中文分词:提前移除空格
        key = key.strip()
        print(key)

        # entity = '姚明'
        # attribute = '身高'
        # values = entity_attribute2value(entity, attribute)  # 根据实体、属性获取属性值
        # print(values)

        entity = '姚明'
        knowledge = entity2knowledge(entity)  # 根据实体获取知识
        print(knowledge['data']['avp'])
        avp = knowledge['data']['avp']
        key_ = toids(key.replace(entity,''))
        key_vec = vec_model.predict(key_)
        maxcos = 0
        attribute = ''
        for a in avp:
            t = toids(a[0])
            t_vec = vec_model.predict(t)
            cos = cosin_distance(key_vec.mean(axis = 1)[0],t_vec.mean(axis = 1)[0])
            if cos > maxcos:
                maxcos = cos
                attribute = a[0]
        ctx['question'] = key
        # values = extract_spoes(key)
        print("maxa = ",attribute)
        values = entity_attribute2value(entity, attribute)  # 根据实体、属性获取属性值
        ctx['rlt'] = values
    return render(request, "kgqa.html", ctx)