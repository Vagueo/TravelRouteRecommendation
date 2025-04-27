# 自定义扩展，包括代理和MySql链接
from pymongo import MongoClient
from scrapy import signals
import random
from retrying import retry

class MongoConnect:
    def __init__(self) -> None:
          # Set your username, password, database, and cluster URL
        # username = ""
        # password = ""
        database = "mafengwo"
        port = 27017
        host = "localhost"
        connection_url = f"mongodb://localhost:27017/mafengwo"

        # 连接到Mongo数据库
        client = MongoClient(connection_url)
        self.client = client
        db = client.get_database(database)
        self.db = db
        # Select the database to use
        self.note = db.get_collection("detail")
        self.scenic = db.get_collection("scenic")
        self.mdd = db.get_collection("mdd")
        self.zyx = db.get_collection("zyx")
        self.route = db.get_collection("route")

mongo = MongoConnect()

class Proxy:
    def __init__(self):
        self.refreshProxy()
        # 获取代理时可能出现异常，异常以后进行重试
    @retry(stop_max_attempt_number=3, wait_fixed=1000)
    def refreshProxy(self):
        # 填充代理IP列表
        self._proxy_list = [
            "127.0.0.1:7890"
        ]

    @property
    def proxy(self):
        return self._proxy_list

    @proxy.setter
    def proxy(self, list):
        self._proxy_list = list

    # 随机获取代理IP
    def getProxy(self):
        proxy = random.choice(self._proxy_list)
        # username = ""  # 如果代理需要身份验证，填写用户名
        # password = ""  # 如果代理需要身份验证，填写密码
        return {
            # "http": f"http://{username}:{password}@{proxy}/",
            # "https": f"http://{username}:{password}@{proxy}/"
            "http": f"http://{proxy}/",
            "https": f"http://{proxy}/"
        }
pro = Proxy()

# Scrapy 扩展类
class MyExtend:
    """Scrapy 扩展：在爬虫生命周期中执行特定操作"""

    @classmethod
    def from_crawler(cls, crawler):
        ext = cls()
        crawler.signals.connect(ext.spider_opened, signal=signals.spider_opened)
        crawler.signals.connect(ext.spider_closed, signal=signals.spider_closed)
        return ext

    def spider_opened(self, spider):
        """爬虫启动时执行的操作"""
        spider.logger.info("爬虫启动：初始化数据库连接...")
        # 例如，可以在启动时预加载一些数据
        spider.mongo = mongo

    def spider_closed(self, spider, reason):
        """爬虫关闭时执行的操作"""
        spider.logger.info(f"爬虫关闭，原因：{reason}")
        # 这里可以清理资源，例如关闭数据库连接
        mongo.client.close()