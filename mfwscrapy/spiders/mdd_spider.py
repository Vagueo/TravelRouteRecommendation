from scrapy.selector import Selector
from mfwscrapy.items import Mdd, Scenic
import time
import json
import scrapy
import execjs

# 爬取首页的热门mdd
class MddSpider(scrapy.Spider):
    name = "mdd_spider"
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
        mdd_list = set()
        with open("./datasets/raw/scenic.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                scenic = json.loads(line)
                mddId = scenic["mddId"]
                mddTitle = scenic["mddTitle"]
                if mddId and mddTitle:
                    mdd = Mdd()
                    mdd["mddId"] = mddId
                    mdd["mddTitle"] = mddTitle
                    mdd_list.add(mdd)

        for mdd in mdd_list:
            mdd_details_url = self.base_root + "/jd/" + mdd["mddId"] + "/gonglve.html"
            print(f"mdd: {mdd}, mdd_details_url: {mdd_details_url}")
            yield scrapy.Request(mdd_details_url, callback=self.parse_mdd, headers=self.headers, meta={"mdd": mdd})

    # 发送post请求获得页面源码
    def parse_mdd(self, response):
        mdd = response.meta["mdd"]
        data = {
            'sAct': 'KMdd_StructWebAjax|GetPoisByTag',
            'iTagId': '0',
            'iMddid': mdd["mddId"],
            'iPage': '1',
            '_ts': str(int(time.time() * 1000))
        }
        data['_sn'] = self.js_ctx.call('getSn', data)
        yield scrapy.FormRequest("https://www.mafengwo.cn/ajax/router.php",
                          headers=self.headers,
                          callback=self.parse_mdd_poi,
                          formdata=data,
                          meta={"mdd": mdd})

    # 获得该目的地的15个景点
    def parse_mdd_poi(self, response):
        mdd = response.meta["mdd"]
        data = json.loads(response.text)["data"]  # 解析出的text是json格式，先将json解析为字典
        html_content = data["list"]  # 字典中只有一个元素，其中list对应的为html文档
        selector = Selector(text=html_content)
        # 该城市的景点总数
        li_lists = selector.xpath("//li")
        poi_list = []
        for li in li_lists:
            scenic = Scenic()
            poi_url = li.xpath('./a/@href').get()
            poi_title_raw = li.xpath('./a/@title').get()

            # 解码 title
            if poi_title_raw and "\\u" in poi_title_raw:
                try:
                    poi_title = poi_title_raw.encode().decode("unicode_escape")
                except Exception as e:
                    print(f"解码失败: {poi_title_raw}, 错误: {e}")
                    poi_title = poi_title_raw
            else:
                poi_title = poi_title_raw

            # 空值判断
            if poi_url and poi_title:
                poi_id = poi_url.split('/')[-1].replace('.html', '')
                scenic["poi_id"] = poi_id
                scenic["poi_title"] = poi_title
                scenic["mddId"] = mdd["mddId"]
                scenic["mddTitle"] = mdd["mddTitle"]

                poi_list.append({"poi_id": scenic["poi_id"], "poi_title": scenic["poi_title"]})
            else:
                print(f"Missing poi_url or poi_title in li: {li.get()}")
        mdd["poi_list"] = poi_list
        yield mdd

    def closed(self, reason):
        print("mdd的爬虫任务已完成！结束运行。")
