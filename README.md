# NLPLAB

## 项目介绍：

本项目是一个集知识图谱和nlp中中文分词、命名实体识别、三元组抽取、阅读理解问答系统、情感分析、语义识别等功能于一身的实验项目。
> 本项目的后端使用的是django框架，用它的原因是方便快捷，不考虑高并发和性能，仅仅做Demo。本项目主要是实验nlp相关的功能。
## 目录结构：

```
.
├── bert4keras //开源框架bert的民间keras版推荐使用
├── config //配置文件,包括
│   ├── bert预训练模型和中文字符
│   └──pathconfig 
├── data //存放一下训练数据
├── demo     // django项目路径
├── Model  // 模型层，用于封装Item类，以及neo4j和csv的读取
├── nlplab  // 页面逻辑层（View）
├── static    // 静态资源  
├── templates    // html页面 
├──toolkit   // 工具库，包括预加载，命名实体识别、分词、三元组抽取、阅读理解QA等
└──weights   //bert模型训练保存的权重参数
```


## 项目配置

**0.安装基本环境：**

确保安装好python3.6和Neo4j（任意版本）
 
安装一系列pip依赖： cd至项目根目录，运行 sudo pip3 install -r requirement.txt

**1.导入数据：**

将hudong_pedia.csv导入neo4j：开启neo4j，进入neo4j控制台。将bank.csv放入neo4j安装目录下的/import目录。在控制台依次输入：
百万级以上数据建议官方提供的 neo4j-import 工具，不然会非常慢。
```
1.创建节点
load csv with headers from 'file:///bank1.csv' as line
create(bank:Bank{name:line.bank,count:line.count})
2.创建节点
load csv with headers from 'file:///series.csv' as line
create(series:Series{name:line.series,count:line.count})
3创建索引
CREATE CONSTRAINT ON (p:Bank)
ASSERT p.name IS UNIQUE
4.创建关系
load csv with headers from 'file:///series.csv' as line
match (entity1:Bank{name:line.bank}),(entity2:Series{name:line.series})
create(entity1)-[:subSeries{subSeries:line.relation}]->(entity2)
5.创建节点#这个type的类型不唯一 创建关系的时候回出现问题
load csv with headers from 'file:///type_entity.csv' as line
create(t:CarType{name:line.type})

CREATE CONSTRAINT ON (p:CarType)
ASSERT p.name IS UNIQUE
6创建关系  
load csv with headers from 'file:///series_type_relation.csv' as line
match (entity1:Series{name:line.series}),(entity2:CarType{name:line.type})
create(entity1)-[:Subseries{series_type:line.relation}]->(entity2)
```


*（如果导入的时候出现neo4j jvm内存溢出，可以在导入前，先把neo4j下的conf/neo4j.conf中的dbms.memory.heap.initial_size 和dbms.memory.heap.max_size调大点。导入完成后再把值改回去）*




**2.修改Neo4j用户**

进入demo/Model/neo_models.py,修改第9行的neo4j账号密码，改成你自己的

**3.启动服务**

运行脚本：python manage.py runserver


这样就成功的启动了django。我们进入8000端口主页面，输入文本，即可看到以下命名实体和分词的结果（确保django和neo4j都处于开启状态）
## 1 汽车知识图谱
汽车知识图谱是从汽车之家网站爬下来的数据，有车牌、车系、车型等数据。第一个项目主要是实现垂直领域分词、命名实体识别，实体查询和关系查询。
### 1.1 中文分词和实体识别
这是中文分词和命名实体识别的结果。
使用的是清华大学的thulac分词方法和实体识别。

**实体识别**
![shitishibie.png](https://i.loli.net/2020/09/14/bDAu4PLnfa2JTYM.png)
**分词**
![zhongwenfenci.png](https://i.loli.net/2020/09/14/vyS7VLRKl3ZsaBt.png)


### 1.2 实体查询

实体查询部分，我们能够搜索出与某一实体相关的实体，以及它们之间的关系：

实体查询的结果是：
![shitichaxun.jpg](https://i.loli.net/2020/09/14/RTPaiyzDcGEbUoj.png)

### 1.3 关系查询

关系查询即查询三元组关系entity1-[relation]->entity2 , 分为如下几种情况:

* 指定第一个实体entity1
* 指定第二个实体entity2
* 指定第一个实体entity1和关系relation
* 指定关系relation和第二个实体entity2
* 指定第一个实体entity1和第二个实体entity2
* 指定第一个实体entity1和第二个实体entity2以及关系relation

下图所示，是指定第一个实体entity1和第二个实体entity2的查询结果

![relsearch.png](https://i.loli.net/2020/09/15/El8ukjewXgvUax4.png)
----------------------


## 2 三元组抽取
这个项目使用的数据是https://ai.baidu.com/broad/download?dataset=sked

**A sample：**
```
{
    "text": "《新駌鸯蝴蝶梦》是黄安的音乐作品，收录在《流金十载全记录》专辑中",
    "spo_list": [
        {
            "subject": "新駌鸯蝴蝶梦",
            "predicate": "所属专辑",
            "object": "流金十载全记录",
            "subject_type": "歌曲",
            "object_type": "音乐专辑"
        },
        {
            "subject": "新駌鸯蝴蝶梦",
            "predicate": "歌手",
            "object": "黄安",
            "subject_type": "歌曲",
            "object_type": "人物"
        }
    ]
}
```

就是从这个text中抽取出（新駌鸯蝴蝶梦，所属专辑，流金十载全记录）和（新駌鸯蝴蝶梦，歌手，黄安），这个样本中是一对多的关系。
### 2.1 三元组抽取
下面展示一个抽取示例，使用的算法包含在toolkit目录下面的roberta_relation_extraction.py
使用的robert模型，效果还不错。
基于Bert的三元组抽取模型结构示意图：


![berttriple.png](https://i.loli.net/2020/09/22/Hm7vrt8OMRUkpYN.png)


**结果展示**

![sanyuanzu.png](https://i.loli.net/2020/09/14/3TWyelwJx6kZ1zj.png)

### 3 情感分析
下面展示一个情感分析示例，使用的算法包含在toolkit目录下面的albert_sentiment.py
使用的是albert  base版模型，比较轻快。

示例：

![sentiment.png](https://i.loli.net/2020/09/22/UpxavhJAdHctyzL.png)

### 4 文本生成
下面展示的是一个文本标题生成的任务
coding中