from scrapy.selector import Selector
from mfwscrapy.items import Route, Scenic
import time
import json
import scrapy
import re
import execjs

class RouteSpider(scrapy.Spider):
    name = "route_spider"
    start_urls = "https://www.mafengwo.cn/mdd/route/21536.html"

    def __init__(self):
        super().__init__()
        with open("./jscode.js", 'r', encoding='utf-8') as f:
            self.js_ctx = execjs.compile(f.read())
    base_root = "https://www.mafengwo.cn"
    headers = {
        'accept': 'application/json, text/javascript, */*; q=0.01',
        'accept-language': 'zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6',
        'priority': 'u=1, i',
        'referer': 'https://www.mafengwo.cn/travel-scenic-spot/mafengwo/21536.html',
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
        yield scrapy.Request(self.start_urls,headers=self.headers,callback=self.parse_all_route)

    def parse_all_route(self, response):
        selector = Selector(text=response.text)
        li_list = selector.xpath('//*[@id="container"]/div[2]/div/div[2]/ul/li')
        for li in li_list:
            route_page_url_href = li.xpath('./div[@class="img"]/a/@href').get()
            if route_page_url_href:
                route_page_url = self.base_root + route_page_url_href
                mddId = route_page_url.split('/')[-1].replace('.html','')
                yield scrapy.Request(route_page_url,headers=self.headers,callback=self.parse_route_pages,meta={"mddId":mddId})

    # 爬取该页面的所有线路的详情信息
    def parse_route_pages(self, response):
        mddId = response.meta["mddId"]
        selector = Selector(text=response.text)
        num_page_text = selector.xpath('//*[@id="routelistpagination"]/div/span[@class="count"]/text()').get()
        print(f'当前爬取的线路: {num_page_text}')
        if num_page_text:
            # num_pages = int(re.search(r'\d+', num_page_text).group())
            num_pages = int(num_page_text[1])
            print(f"共{num_pages}页")
        else:
            print(f'未找到页数信息：{response.url}')
            num_pages = 1  # 默认设为1页防止出错
        if num_pages == 1:  # 若只有一页那么页面数据就在page_text中
            div_list = selector.xpath('//*[@id="routebody"]/div[1]')
            for div in div_list:
                route = Route()
                route_detail_url_href = div.xpath('./div/div/dl[1]/dd/p/a/@href').get()
                route["routeTitle"] = div.xpath('./div/div/dl[1]/dt/a/h2/text()').get()

                # route["days"] = int(re.search(r'\d+', div.xpath('./div/div/dl[1]/dt/div/span[1]/strong/text()').get()).group())
                # 1. 首先尝试从默认位置提取天数
                day_text = div.xpath('./div/div/dl[1]/dt/div/span[1]/strong/text()').get()
                day_num = None

                if day_text:
                    match = re.search(r'\d+', day_text)
                    if match:
                        day_num = int(match.group())

                # 2. 如果失败，再从标题中提取
                if day_num is None:
                    title_text = selector.xpath('/html/body/div[2]/div[1]/div/dl/dt/h1/text()').get()
                    if title_text:
                        match = re.search(r'(\d+)[天日]', title_text)
                        if match:
                            day_num = int(match.group(1))

                if route_detail_url_href != None and day_num != None:
                    route["routeId"] = route_detail_url_href.split("/")[-1].replace('.html', '')
                    route["days"] = day_num
                    route_detail_url = self.base_root + route_detail_url_href
                    yield scrapy.Request(route_detail_url, callback=self.parse_route_detail, meta={"route": route})
        else:  # 若页面数大于1，那么页面数据在ajax请求中，需要像前面的目的地页面的请求过程一样处理
            for page in range(1, num_pages + 1):
                data = {
                    "mddid": mddId,
                    "page": str(page),
                    "_ts": str(int(time.time() * 1000)),
                }
                data['_sn'] = self.js_ctx.call('getSn', data)
                headers = {
                    'accept': 'application/json, text/javascript, */*; q=0.01',
                    'accept-language': 'zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6',
                    'cache-control': 'no-cache',
                    'content-type': 'application/x-www-form-urlencoded; charset=UTF-8',
                    'origin': 'https://www.mafengwo.cn',
                    'pragma': 'no-cache',
                    'priority': 'u=1, i',
                    'referer': 'https://www.mafengwo.cn/mdd/route/' + mddId + '.html',
                    'sec-ch-ua': '"Microsoft Edge";v="135", "Not-A.Brand";v="8", "Chromium";v="135"',
                    'sec-ch-ua-mobile': '?0',
                    'sec-ch-ua-platform': '"Windows"',
                    'sec-fetch-dest': 'empty',
                    'sec-fetch-mode': 'cors',
                    'sec-fetch-site': 'same-origin',
                    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36 Edg/135.0.0.0',
                    'x-requested-with': 'XMLHttpRequest',
                }
                yield scrapy.FormRequest(
                    url="https://www.mafengwo.cn/mdd/base/routeline/pagedata_routelist",
                    headers=headers,
                    formdata=data,
                    callback=self.parse_ajax_route_list,
                )

    # 依次获取多页的每一页的信息，然后像单页一样处理
    def parse_ajax_route_list(self, response):
        try:
            json_data = json.loads(response.text)
        except json.JSONDecodeError:
            print("JSON decode failed: %s", response.text[:200])
            return

        html_content = json_data.get("list", "")
        selector = Selector(text=html_content)
        div_list = selector.xpath('//div[contains(@class,"row row-line")]')
        for div in div_list:
            route = Route()
            route["routeTitle"] = div.xpath('./div/div/dl[1]/dt/a/h2/text()').get()
            href = div.xpath('./div/div/dl[1]/dd/p/a/@href').get()
            # 1. 首先尝试从默认位置提取天数
            day_text = div.xpath('./div/div/dl[1]/dt/div/span[1]/strong/text()').get()
            day_num = None

            if day_text:
                match = re.search(r'\d+', day_text)
                if match:
                    day_num = int(match.group())

            # 2. 如果失败，再从标题中提取
            if day_num is None:
                title_text = selector.xpath('/html/body/div[2]/div[1]/div/dl/dt/h1/text()').get()
                if title_text:
                    match = re.search(r'(\d+)[天日]', title_text)
                    if match:
                        day_num = int(match.group(1))

            if href != None and day_num != None:
                route["routeId"] = href.split("/")[-1].replace('.html', '')
                route["days"] = day_num
                route["routeId"] = href.split('/')[-1].replace('.html', '')
                route_detail_url = "https://www.mafengwo.cn" + href
                yield scrapy.Request(route_detail_url, callback=self.parse_route_detail, meta={"route": route})

    # 处理单页线路的详情页的内容
    def parse_route_detail(self, response):
        route = response.meta["route"]  # 获取通过meta传递的route对象
        selector = Selector(text=response.text)
        day_divs = selector.xpath('.//div[@class="container"]/div[@class="row row-lineDetail"]/div[@class="wrapper"]/div[@class="day-list"]/div[@class="day-item"]')
        daily_routes = []
        for idx, day_div in enumerate(day_divs):
            day_route = {}  # 每一天的路线
            day = f'D{idx + 1}'  # 第几天，例如:D1
            # 获取当天所有的景点
            poi_nodes = day_div.xpath('.//div[@class="poi-name"]//li/span[@class="place"]')
            poi_list = []
            for poi in poi_nodes:
                scenic = Scenic()
                poi_href = poi.xpath('./a/@href').get()
                poi_id = poi_href.split('/')[-1].replace('.html', '')
                poi_title = poi.xpath('./a/text()').get()
                poi_title = re.sub(r'\s+', ' ', poi_title).strip()  # 清理多余空白字符

                if poi_id and poi_title:
                    scenic["poi_id"] = poi_id
                    scenic["poi_title"] = poi_title
                    # yield scrapy.Request(self.base_root + poi_href, callback=self.parse_poi_detail,
                    #                      headers=self.headers, meta={"scenic": scenic})
                    poi_list.append({"poi_id": poi_id, "poi_title": poi_title})
            day_route["day"] = day
            day_route["poi_list"] = poi_list
            daily_routes.append(day_route)

        route["daily_routes"] = daily_routes
        yield route

    def closed(self, reason):
        print("route_spider的爬虫任务已完成！结束运行。")