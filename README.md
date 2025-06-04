# TravelRouteRecommendation
# 目录
  &nbsp;&nbsp;一、[项目概述](#一项目概述)
  
  &nbsp;&nbsp;二、[环境配置](#二环境配置)
  
  &nbsp;&nbsp;三、[项目各部分的具体介绍](#三项目各部分的具体介绍)
  
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1. [mfwscrapy](#1-mfwscrapy)
  
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2. [datasets](#2-datasets)
  
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3. [model](#3-model)
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
### 1. mfwscrapy
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
### 2. datasets
- 该部分存放的是我们的数据集和数据预处理及清洗的代码部分。
- 数据集介绍：
  1. 预训练数据集：（生成的虚拟路线数据集）
  - 生成步骤：
  （1）先通过datasets/addLoc给清洗后城市数据加上经纬度信息得到mdd_step3_with_details.json；
  （2）再结合mdd_step3_with_details.json和清洗后的景点数据scenic_step2.jsonl，进行路线生成：
      - 起点选择：随机选取一个城市作为出发点；
      - 路径扩展：在每一步中，根据以下两种方式之一扩展路径：第一种城市内跳转，在当前城市内，随机选择一个景点节点进行移动；第二种城市间跳转，跳转至另一个与当前城市地理距离不超过300公里的城市节点。
      - 最终生成5000条虚拟路线，并且每条线路中的地点数为10-12个```（经过数据泄露处理后只有4058条）```
  2. 微调数据集：（真实的热门路线数据集）
     该部分数据集通过mfwscrapy获取得到。
- datasets的项目结构：
    ```
    ./raw/mdd.jsonl：真实MFW数据集中的城市信息。
    ./raw/scenic.jsonl：真实MFW数据集中的景点信息。
    ./raw/route.jsonl：真实MFW数据集中的路线信息。
    ./precleaning/MFW/route_step0_deduplicated.jsonl：去重后线路数据。
    ./precleaning/MFW/mdd_step1.jsonl：清洗后的城市数据。
    ./precleaning/MFW/scenic_step2.jsonl.jsonl：清洗后的景点数据。
    ./precleaning/MFW/mdd_step3_with_details.jsonl：聚合了城市下的景点的details得到城市的details的城市数据。
    ./precleaning/MFW/mdd_step3_with_details_with_location.jsonl：加上了经纬度的城市数据。
    ./precleaning/MFW/route_step4.jsonl：最终得到的真实线路数据。
    ./precleaning/MFW/virtual_routes.jsonl：未经过数据泄露处理的生成的虚拟路线数据。
    ./precleaning/MFW/virtual_routes_filtered.jsonl：经过数据泄露处理后的生成的虚拟路线数据。
    ```
（已弃用）FourSquare数据集：（[获取网址](https://sites.google.com/site/yangdingqi/home/foursquare-dataset)）

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
- 清洗后的数据结构：
  ```
     {
        "route_id": ,
        "trajectory": [
          {
            "type": "mdd",
            "mddId": mddId,
            "mddTitle": mddTitle,
            "details": details
          },
          {
            "type": "poi",
            "poi_id": poi_id,
            "poi_title": poi_title,
            "mddId": mddId,
            "mddTitle": mddTitle,
            "details": details
          },
          ....
        ]
    }
  ```
  ### 3. model
- 该部分存放的是推荐的模型、训练和测试的部分以及自回归得到完整线路的部分。
- 项目结构：
    ```
    ./model/generate_virtual_routes.py：生成虚拟路线的部分。
    ./model/main.py：加载数据集并且进行预训练和微调的部分。
    ./model/model.py：基于城市和景点的双层推荐的模型以及多头注意力机制构建的模型部分。
    ./model/train.py：训练一个轮次的函数、评价函数以及评估指标计算和打印结构的部分。
    ./model/spilt.py：用来划分虚拟数据集的训练和验证集以及真实数据集的训练、验证和测试集的部分。
    ./model/config.py：一些基础参数的设置部分。
    ```
