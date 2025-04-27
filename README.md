## TravelRouteRecommendation
一、 [项目概述](#项目概述)

二、 [环境配置](#环境配置)

三、 [项目各部分的具体介绍](#项目各部分的具体介绍)
1. [mfwscrapy](##mfwscrapy)
# 一、项目概述：
该项目共分为如下三个部分：
1. 数据集构建部分：mfwscrapy。
2. 数据集清洗以及划分出训练集和测试集的部分：datasets。
3. 推荐系统的模型部分。
# 二、环境配置：
- 首先若要正确运行我们的项目，需要配置如下的实验环境：
  - python环境：python3.9
  - 所需依赖：在requirements.txt中，可以通过```pip install -r requirements.txt```命令来安装该项目所需的所有依赖部分。
# 三、项目各部分的具体介绍：
## 1. mfwscrapy:
- 该部分放置的是用来构建我们自己的数据集的部分,配置完上述的实验环境后利用下面的命令在终端运行该项目
```
scrapy crawl mfw_mdd_route
```
- mfwscrapy的项目结构：
```
./spiders/mfw_mdd_route.py: mdd,scenic和route的爬取部分

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
    {"poi_id": , "poi_title": , "city_id": "11252", "details": }
    ```
    - poi_id: 景点的唯一标识；
    - poi_title：景点的名称；
    - city_id：景点所在城市的id （可以和mddId一一对应）；
    - details：关于该景点的描述。
  - route.jsonl:
    ```
    {"mddId": , "mddTitle": , "routeTitle": , "routeId": , "days": 3, "daily_routes": [{"day": "D1", "poi_list": [{"poi_id": , "poi_title": , "time": "1-3小时"}, ... ,{"poi_id": , "poi_title": , "time": "1-3小时"}]}, ... ]}]}

    ```
    - mddId: 当前route的起始mdd的Id；
    - mddTitle：当前route的起始mdd的名称；
    - routeId：route的唯一标识；
    - routeTitle：route的名称；
    - days：当前旅游线路旅游需要花的天数；
    - daily_routes：当前的旅游路线，列表中的每个元素为每天的线路。
