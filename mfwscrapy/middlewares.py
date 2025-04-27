from scrapy import signals
from scrapy.http import Request, Response
from twisted.internet.error import TimeoutError, TCPTimedOutError
from selenium import webdriver
from selenium.webdriver import EdgeOptions
from selenium.webdriver.edge.service import Service
import time
from mfwscrapy.myextend import pro

class MfwscrapyDownloaderMiddleware:
    def __init__(self):
        """ 初始化时获取 Mafengwo Cookies """
        self.cookies_dict = self.get_cookies()
        print(f'cookies:{self.cookies_dict}')

    def get_cookies(self):
        """ 使用 Selenium 访问 Mafengwo 并提取 Cookies """
        option = EdgeOptions()
        option.add_argument("--disable-blink-features=AutomationControlled")  # 防止 Selenium 被检测
        option.add_experimental_option("excludeSwitches", ['enable-automation'])
        option.add_experimental_option("useAutomationExtension", False)
        browser = webdriver.Edge(options=option)
        browser.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
            "source": """
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                })
            """
        })
        time.sleep(5)  # 等待页面加载
        browser.get('https://www.mafengwo.cn/')
        cookies = browser.get_cookies()
        browser.quit()

        return {cookie["name"]: cookie["value"] for cookie in cookies}

    @classmethod
    def from_crawler(cls, crawler):
        """ 连接 Scrapy 信号 """
        s = cls()
        crawler.signals.connect(s.spider_opened, signal=signals.spider_opened)
        return s

    def process_request(self, request: Request, spider):
        """ 在请求前配置代理和 Cookies """
        # 设置代理
        request.meta['proxy'] = pro.getProxy()["https"]

        # 设置 Cookies
        request.cookies = self.cookies_dict

    def process_response(self, request: Request, response: Response, spider):
        """ 直接返回响应 """
        return response

    def process_exception(self, request, exception, spider):
        """ 处理请求异常，进行重试 """
        print(f"Request Exception: {exception}")
        if isinstance(exception, (TimeoutError, TCPTimedOutError)):
            return request  # 重新请求

    def spider_opened(self, spider):
        """ 记录 Spider 启动日志 """
        spider.logger.info("Spider opened: %s" % spider.name)
