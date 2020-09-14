# NLPLAB

## 项目介绍：

本项目是一个集知识图谱和nlp中中文分词、命名实体识别、三元组抽取、阅读理解问答系统、情感分析、语义识别等功能于一身的实验项目。
> 本项目的后端使用的是django框架，用它的原因是方便快捷，顺手就用了。本项目主要是实验nlp相关的功能，web能看就行。
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
![shitishibie.png](https://i.loli.net/2020/09/14/bDAu4PLnfa2JTYM.png)

![zhongwenfenci.png](https://i.loli.net/2020/09/14/vyS7VLRKl3ZsaBt.png)
实体查询的结果是：
![shitichaxun.jpg](https://i.loli.net/2020/09/14/RTPaiyzDcGEbUoj.png)

### 1.2 实体查询

实体查询部分，我们能够搜索出与某一实体相关的实体，以及它们之间的关系：
![image](https://raw.githubusercontent.com/CrisJk/SomePicture/master/blog_picture/entitySearch.png)

![](https://raw.githubusercontent.com/CrisJk/SomePicture/master/blog_picture/entitySearch2.png)

### 1.3 关系查询

关系查询即查询三元组关系entity1-[relation]->entity2 , 分为如下几种情况:

* 指定第一个实体entity1
* 指定第二个实体entity2
* 指定第一个实体entity1和关系relation
* 指定关系relation和第二个实体entity2
* 指定第一个实体entity1和第二个实体entity2
* 指定第一个实体entity1和第二个实体entity2以及关系relation

下图所示，是指定关系relation和第二个实体entity2的查询结果

![](https://raw.githubusercontent.com/CrisJk/SomePicture/master/blog_picture/relationSearch.png)
----------------------


## 2 三元组查询
这个项目使用的数据是

### 2.1 三元组查询

![sanyuanzu.png](https://i.loli.net/2020/09/14/3TWyelwJx6kZ1zj.png)

### 知识的树形结构

农业知识概览部分，我们能够列出某一农业分类下的词条列表，这些概念以树形结构组织在一起：

![image](https://raw.githubusercontent.com/qq547276542/blog_image/master/agri/6.png)

农业分类的树形图：

![image](https://raw.githubusercontent.com/qq547276542/blog_image/master/agri/5.png)

### 训练集标注

我们还制作了训练集的手动标注页面，每次会随机的跳出一个未标注过的词条。链接：http://localhost:8000/tagging-get , 手动标注的结果会追加到/label_data/labels.txt文件末尾：

我们将这部分做成了小工具，可复用：https://github.com/qq547276542/LabelMarker

![image](https://raw.githubusercontent.com/qq547276542/blog_image/master/agri/4.png)

(update 2018.04.07)  同样的，我们制作了标注关系提取训练集的工具，如下图所示

![](https://raw.githubusercontent.com/CrisJk/SomePicture/master/blog_picture/tagging.JPG)

如果Statement的标签是对的，点击True按钮；否则选择一个关系，或者输入其它关系。若当前句子无法判断，则点击Change One按钮换一条数据。

说明:　Statement是/wikidataSpider/TrainDataBaseOnWiki/finalData中train_data.txt中的数据，我们将它转化成json,导入到mongoDB中。标注好的数据同样存在MongoDB中另一个Collection中。关于Mongo的使用方法可以参考官方tutorial，或者利用这篇文章简单了解一下[MongoDB](http://crisjk.site/2018/04/04/MongoDB-Tutorial/) 

我们在MongoDB中使用两个Collections，一个是train_data，即未经人工标注的数据；另一个是test_data，即人工标注好的数据。

![](https://raw.githubusercontent.com/CrisJk/crisjk.github.io/master/resource/pictures/Agriculture-KnowledgeGraph-Data-README/mongo.png)



**使用方法**: 启动neo4j,mongodb之后，进入demo目录，启动django服务，进入127.0.0.1:8000/tagging即可使用




## 思路

### 命名实体识别:

### 实体分类：

#### 特征提取：

![image](https://raw.githubusercontent.com/qq547276542/blog_image/master/agri/1.png)


#### 分类器：KNN算法

- 无需表示成向量，比较相似度即可
- K值通过网格搜索得到

#### 定义两个页面的相似度sim(p1,p2)：

- 
  title之间的词向量的余弦相似度(利用fasttext计算的词向量能够避免out of vocabulary)
- 2组openType之间的词向量的余弦相似度的平均值
- 相同的baseInfoKey的IDF值之和（因为‘中文名’这种属性贡献应该比较小）
- 相同baseInfoKey下baseInfoValue相同的个数
- 预测一个页面时，由于KNN要将该页面和训练集中所有页面进行比较，因此每次预测的复杂度是O(n)，n为训练集规模。在这个过程中，我们可以统计各个分相似度的IDF值，均值，方差，标准差，然后对4个相似度进行标准化:**(x-均值)/方差**
- 上面四个部分的相似度的加权和为最终的两个页面的相似度，权值由向量weight控制，通过10折叠交叉验证+网格搜索得到


### Labels：（命名实体的分类）

| Label | NE Tags                                  | Example                                  |
| ----- | ---------------------------------------- | ---------------------------------------- |
| 0     | Invalid（不合法）                             | “色调”，“文化”，“景观”，“条件”，“A”，“234年”（不是具体的实体，或一些脏数据） |
| 1     | Person（人物，职位）                            | “袁隆平”，“副市长”                         |
| 2     | Location（地点，区域）                          | “福建省”，“三明市”，“大明湖”                        |
| 3     | Organization（机构，会议）                      | “华东师范大学”，“上海市农业委员会”                      |
| 4     | Political economy（政治经济名词）                | “惠农补贴”，“基本建设投资”                          |



### 关系抽取

使用远程监督方法构建数据集，利用tensorflow训练PCNN模型
详情见： [relationExtraction](https://github.com/qq547276542/Agriculture_KnowledgeGraph/tree/master/relationExtraction)
