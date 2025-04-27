## TravelRouteRecommendation
一、 [项目概述](#项目概述)

二、 [环境配置](#环境配置)

三、 [项目各部分的具体介绍](#项目各部分的具体介绍)
1. [mfwscrapy](##mfwscrapy)
# 一、项目概述：
该项目共分为三个部分
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


