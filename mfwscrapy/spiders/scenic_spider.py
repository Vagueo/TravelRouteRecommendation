from scrapy.selector import Selector
import json
import scrapy
from mfwscrapy.items import Scenic
import execjs

class ScenicSpider(scrapy.Spider):
    name = "scenic_spider"
    def __init__(self):
        super().__init__()
        with open("./jscode.js", 'r', encoding='utf-8') as f:
            self.js_ctx = execjs.compile(f.read())
    base_root = "https://www.mafengwo.cn"
    headers = {
        'accept': 'application/json, text/javascript, */*; q=0.01',
        'accept-language': 'zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6',
        'priority': 'u=1, i',
        'referer': 'https://www.mafengwo.cn/mdd/citylist/21536.html',
        'sec-ch-ua': '"Chromium";v="134", "Not:A-Brand";v="24", "Microsoft Edge";v="134"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-origin',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36 Edg/134.0.0.0',
        'x-requested-with': 'XMLHttpRequest',
    }

    def start_requests(self):
        scenic_list = set()
        with open("./datasets/raw/route.jsonl", "r", encoding="utf-8") as route_f:
            for line in route_f:
                route = json.loads(line)
                daily_routes = route["daily_routes"]
                for daily_route in daily_routes:
                    poi_list = daily_route["poi_list"]
                    for poi in poi_list:
                        scenic = Scenic()
                        poi_id = poi["poi_id"]
                        poi_title = poi["poi_title"]

                        if poi_id and poi_title:
                            scenic["poi_id"] = poi_id
                            scenic["poi_title"] = poi_title
                            scenic_list.add(scenic)


        with open("./datasets/raw/mdd.jsonl", "r", encoding="utf-8") as mdd_f:
            for line in mdd_f:
                mdd = json.loads(line)
                print(f"提取出的mdd为{mdd}")
                if mdd:
                    poi_list = mdd["poi_list"]
                    if poi_list:
                        for poi in poi_list:
                            scenic = Scenic()
                            poi_id = poi["poi_id"]
                            poi_title = poi["poi_title"]

                            if poi_id and poi_title:
                                scenic["poi_id"] = poi_id
                                scenic["poi_title"] = poi_title
                                scenic_list.add(scenic)

        for scenic in scenic_list:
            print(f"scenic{scenic}")
            poi_details_url = self.base_root + "/poi/" + scenic["poi_id"] + ".html"
            yield scrapy.Request(poi_details_url, callback=self.parse_poi_detail,headers=self.headers, meta={"scenic": scenic})

    # 得到景点的详情介绍
    def parse_poi_detail(self, response):
        scenic = response.meta["scenic"]
        selector = Selector(text=response.text)

        # 提取 details 内容
        detail_text = self.extract_details(selector)
        if detail_text == None:
            detail_text = scenic["poi_title"]
        scenic["details"] = detail_text

        # 提取 city_href（先第二种，再第一种）
        city_below_href = selector.xpath('//div[@class="crumb"]//span/a/@href').get()  # 地级市的位置
        if city_below_href is None:
            city_below_href = selector.xpath('//div[contains(@class, "top-info")]//span/a/@href').get()

        if city_below_href:
            belong_url = self.base_root + city_below_href
            yield scrapy.Request(belong_url, callback=self.parse_poi_city, headers=self.headers,
                                 meta={"scenic": scenic})
    # 获得景点对应的城市的id
    def parse_poi_city(self, response):
        scenic = response.meta["scenic"]
        selector = Selector(text=response.text)
        city_href = selector.xpath('//*[@id="container"]/div[1]/div/div[2]/div[3]/div/span/a/@href').get()
        city_id = city_href.split('/')[-1].replace('.html', '')
        city_name = selector.xpath('//*[@id="container"]/div[1]/div/div[2]/div[3]/div/span/a/text()').get()
        scenic["mddId"] = city_id
        scenic["mddTitle"] = city_name
        yield scenic

    def extract_details(self, selector):
        # 第二种结构（container -> summary）
        raw = selector.xpath('//div[contains(@class, "summary")]//text()').getall()
        if raw and any(text.strip() for text in raw):
            return ''.join([text.strip() for text in raw if text.strip()])

        # 第一种结构（comment_header 下 poi-info 块中第二段文字）
        raw = selector.xpath(
            '//*[@id="comment_header"]//div[contains(@class, "poi-info")]//div[@class="bd"]/p[2]//text()').getall()
        if raw and any(text.strip() for text in raw):
            return ''.join([text.strip() for text in raw if text.strip()])

        return ''

    def closed(self, reason):
        print("scenic的爬虫任务已完成！结束运行。")



