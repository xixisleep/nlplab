# -*- coding: utf-8 -*-
from django.shortcuts import render
from toolkit.albert_sentiment import sentiment


def sentiment_text(request):
    ctx = {}
    ctx['key'] = "目前不错，希望质量过硬，再追评！"
    if request.POST:
        # 获取输入文本
        key = request.POST['user_text']
        # 中文分词:提前移除空格
        key = key.strip()
        print(key)
        ctx['key'] = key
        values = sentiment(key)
        if values == [1]:
            values = '正面'
        else:
            values = '负面'
        ctx['rlt'] = values
    return render(request, "sentiment.html", ctx)