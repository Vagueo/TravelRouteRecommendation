# TravelRouteRecommendation
# 目录
  &nbsp;&nbsp;一、[项目概述](#一项目概述)
  &nbsp;&nbsp;二、[环境配置](#二环境配置)
  &nbsp;&nbsp;三、[项目各部分的具体介绍](#三项目各部分的具体介绍)
  &nbsp;&nbsp;&nbsp;&nbsp;1. [mfwscrapy](#1mfwscrapy)
  &nbsp;&nbsp;&nbsp;&nbsp;2. [datasets](#2datasets)
  &nbsp;&nbsp;&nbsp;&nbsp;3. [model](#3model)
## 一、项目概述
该项目共分为如下三个部分：
1. 数据集构建部分：mfwscrapy。
2. 数据集存放的位置以及数据清洗的部分：datasets。
3. 推荐系统的模型部分：model。
## 二、环境配置
- 首先若要正确运行我们的项目，需要配置如下的实验环境：
  - python环境：python3.9
  - javascript环境：node.js
  - 所需依赖：在requirements.txt中，可以通过```pip install -r requirements.txt```命令来安装该项目所需的所有依赖部分。
## 三、项目各部分的具体介绍
### 1.mfwscrapy
- 该部分放置的是用来构建我们自己的数据集的代码部分,配置完上述的实验环境后利用下面的命令在终端运行该项目。
  【注意】需要依次按照下面的顺序进行，这样才能爬取完整
```
  scrapy crawl mfw_mdd_route(弃用)
  scrapy crawl route
  scrapy crawl scenic
  scrapy crawl mdd
  scrapy crawl scenic
```
- mfwscrapy的项目结构：
```
  ./spiders/mfw_mdd_route.py: mdd,scenic和route的爬取部分 (已弃用，因为会导致数据爬得不够完整，于是拆解成了下面三个部分的代码)
  ./spiders/route.py
  ./spiders/scenic.py
  ./spiders/mdd.py
  
  ./myextend.py：自定义扩展，包括代理的部分
  ./middlewares.py： 中间件，包括代理和cookie等，该部分运用selenium获取到页面的cookies
  ./settings.py：配置文件
  ./items.py：mdd,scenic和route的数据结构
  ./pipelines.py：主要处理持久化逻辑，将mdd，scenic和route分别存入到‘datasets/raw/’这一路径的jsonl文件中
  
  ../jscode.js: 用来破解各种ajax请求的参数校验，该部分的代码为_sn生成的代码逻辑
```
- 初始的数据结构：
  - mdd.jsonl:
    ```
    {"mddId": , "mddTitle": , "poi_list": [{"poi_id": , "poi_title": }, ... ,{"poi_id": , "poi_title": }]}
    ```
    - mddId: mdd的唯一标识（爬取到的mdd大部分都是城市）；
    - mddTitle：mdd的名称；
    - poi_list：当前mdd下的具有代表性的poi列表（一般情况下是15个poi），后续利用poi的details聚类得到mdd的details。
  - scenic.jsonl:
    ```
    {"poi_id": , "poi_title": , "details":, "mddId": , "mddTitle": }
    ```
    - poi_id: 景点的唯一标识；
    - poi_title：景点的名称；
    - details：关于该景点的描述；
    - mddId：景点所在城市的id （可以和mddId一一对应）；
    - mddTitle：景点所在城市的名称。
  - route.jsonl:
    ```
    {"routeTitle": , "routeId": , "days": 3, "daily_routes": [{"day": "D1", "poi_list": [{"poi_id": , "poi_title": }, ... ,{"poi_id": , "poi_title": }]}, ... ]}
    ```
    - mddId: 当前route的起始mdd的Id；
    - mddTitle：当前route的起始mdd的名称；
    - routeId：route的唯一标识；
    - routeTitle：route的名称；
    - days：当前旅游线路旅游需要花的天数；
    - daily_routes：当前的旅游路线，列表中的每个元素为每天的线路。
### 2.datasets
- 该部分存放的是我们的数据集的部分，并且这个部分还有数据预处理的代码部分。
- 数据集介绍：
  1. FourSquare数据集：（获取网址```https://sites.google.com/site/yangdingqi/home/foursquare-dataset```）
    此数据集包括从 Foursquare 收集的长期（从 2012 年 4 月到 2013 年 9 月的大约 18 个月）全球规模签到数据。它包含 266,909 名用户在 3,680,126 个场所（77 个国家/地区的 415 个城市）的 33,278,683 次签到。这 415 个城市是世界上 Foursquare 用户检查最多的 415 个城市，每个城市都包含至少 10K 个签到。
    - File dataset_TIST2015_Checkins.txt是签到数据，然后分别包含下面这几列：
      1. User ID (anonymized)
      2. Venue ID (Foursquare)
      3. UTC time
      4. Timezone offset in minutes (The offset in minutes between when this check-sin occurred and the same time in UTC, i.e., UTC time + offset is the local time)
    - File dataset_TIST2015_POIs.txt是poi信息，然后分别包含下面这几列：
      1. Venue ID (Foursquare) 
      2. Latitude
      3. Longitude
      4. Venue category name (Foursquare)
      5. Country code (ISO 3166-1 alpha-2 two-letter country codes)
  
    - File dataset_TIST2015_Cities.txt涉及415个城市的数据，分别包含下面这几列：
      1. City name
      2. Latitude (of City center)
      3. Longitude (of City center)
      4. Country code (ISO 3166-1 alpha-2 two-letter country codes)
      5. Country name
      6. City type (e.g., national capital, provincial capital)
  2. MFW数据集：
     该部分数据集通过mfwscrapy获取得到。
- datasets的项目结构：
```
  ./raw/mdd.jsonl：MFW数据集中的mdd信息。
  ./raw/scenic.jsonl：MFW数据集中的scenic信息。
  ./raw/route.jsonl：MFW数据集中的
  ./raw/dataset_TIST2015：FourSquare数据集，该文件夹中包含上述数据集介绍的文件。
  ./precleaning/FourSquare/trajectories_batch{i}.jsonl（i从0-26）：按照用户划分的线路数据。
  ./precleaning/MFW：包含进行预处理后的MFW数据集。
```
- 清洗后的数据结构：
  1. FourSquare数据集：
     - route：
     ```
      {"routeId": , "trajectory": [...{"type": "mdd", "mddId": , "mddTitle": , "details": }, {"type": "poi", "poi_id": , "poi_title": ,"mddId": ,"mddTitle": , "details": }...]}
     ```
     （将用户的签到数据转换为一条线路数据）
     - routeId: 由于User ID为唯一标识，因此将User ID作为routeId。
     - trajectory：是用户的签到数据，将一个用户的所有签到数据按照时间顺序组合到trajectory中，然后作为一个旅游路线来处理，路线中的每一个地点可以是"mdd"（城市）或是"poi"（景点）。对于mdd，它的details就是City；对于poi，它的details就是它的poi类型（例如：University、Restaurant等等）。
  2. MFW数据集：
     （mdd和poi的各个部分经过清洗后都是不为空的，为空的部分被清理掉了）
    - mdd：
      ```
        {"mddId": , "mddTitle": , "details": }
      ```
      - mddId: mdd的唯一标识（爬取到的mdd大部分都是城市）；
      - mddTitle：mdd的名称；
      - details：将这个mdd下对应的poi_list中的poi的details拼接得到mdd的details。
    - scenic.jsonl:
      ```
      {"poi_id": , "poi_title": , "details":, "mddId": , "mddTitle": }
      ```
      - poi_id: 景点的唯一标识；
      - poi_title：景点的名称；
      - details：关于该景点的描述；
      - mddId：景点所在城市的id （可以和mddId一一对应）；
      - mddTitle：景点所在城市的名称。
    - route:
      ```
      {"routeId": , "trajectory": [...{"type": "mdd", "mddId": , "mddTitle": , "details": }, {"type": "poi", "poi_id": , "poi_title": ,"mddId": ,"mddTitle": , "details": }...]}
      ```
      - routeId：将原来的route按照天数拆分开后，如果拆分开后的地点数量小于3则进行合并操作，经过拆分与合并后的路线的routeId为将合并前的最小天数加入到原routeId的末尾。
        （如：假设有一条线路的routeId为123_45，将第1、2、3天的线路进行了合并后，新的routeId为"123_451"，这样仍然保持了routeId的唯一性。）
      - trajectory：原线路中只有poi，我们将poi对应的mdd先加到这个poi前面，如果有相邻的几个poi的mdd都是相同的不会重复加入。
  ### 3.model
- 该部分存放的是推荐的模型、训练和测试的部分以及自回归得到完整线路的部分。
- 项目结构：
  ```
    ./model/main.py：加载并且划分数据集的部分。
    ./model/model.py：基于城市和景点的双层推荐的模型。
    ./model/train.py：训练一个轮次的函数、评价函数、自回归推理生成的函数以及评估指标计算和打印。
  ```
