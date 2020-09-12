# -*- coding: utf-8 -*-
from django.shortcuts import render
from toolkit.roberta_relation_extraction import extract_spoes


def extract_triple(request):
    ctx = {}
    ctx['key'] = "个人简介杨明丽，女，汉族，生于1954年3月，陕西安康人"
    if request.POST:
        # 获取输入文本
        key = request.POST['user_text']
        # 中文分词:提前移除空格
        key = key.strip()
        print(key)
        ctx['key'] = key
        values = extract_spoes(key)
        ctx['rlt'] = values
    return render(request, "triple.html", ctx)