import scrapy
import execjs
import time
import json
import re
from scrapy import FormRequest
from scrapy.selector import Selector
from mfwscrapy.items import Mdd, Route, Scenic


class MfwMddRouteSpider(scrapy.Spider):
    def __init__(self):
        super().__init__()
        with open("./jscode.js", 'r', encoding='utf-8') as f:
            self.js_ctx = execjs.compile(f.read())

    name = "mfw_mdd_route"
    start_urls = "https://www.mafengwo.cn/mdd/base/list/pagedata_citylist"
    # route_url = "https://www.mafengwo.cn/mdd/base/routeline/pagedata_routelist"
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
        page = 1
        yield from self.fetch_page(page)

    def fetch_page(self,page):
        if page > 15: return
        data = {
            "mddid": "21536",
            "page": str(page),
            "_ts": str(int(time.time() * 1000)),
        }
        data['_sn'] = data['_sn'] = self.js_ctx.call('getSn', data)
        yield scrapy.FormRequest(self.start_urls,
                          headers=self.headers,
                          callback=self.parse,
                          formdata=data,
                          meta={"page":page})
    def parse(self, response):
        page = response.meta["page"]
        data = json.loads(response.text)    # 解析出的text是json格式，先将json解析为字典
        html_content = data["list"]         # 字典中只有一个元素，其中list对应的为html文档
        selector = Selector(text=html_content)

        # 获得所有li标签，每个li标签里存着目的地信息
        li_list = selector.xpath("//li")
        for li in li_list:
            mdd = Mdd()
            mdd["mddId"] = li.xpath("./div/a/@data-id").get()
            mdd["mddTitle"] = li.xpath("./div//div/text()").get().strip()  # 使用strip去掉两端的空白字符和换行符
            mdd_url = "https://www.mafengwo.cn/jd/"+mdd["mddId"]+"/gonglve.html"
            yield scrapy.Request(mdd_url,callback=self.parse_mdd,headers=self.headers,meta={"mdd":mdd})
            # 该目的地下的旅游路线的详情页
            route_url = "https://www.mafengwo.cn/mdd/route/"+mdd["mddId"]+".html"
            headers = {
                'accept': 'application/json, text/javascript, */*; q=0.01',
                'accept-language': 'zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6',
                'priority': 'u=1, i',
                'referer': 'https://www.mafengwo.cn/mdd/route/'+mdd["mddId"]+'.html',
                'sec-ch-ua': '"Chromium";v="135", "Not:A-Brand";v="8", "Microsoft Edge";v="1345',
                'sec-ch-ua-mobile': '?0',
                'sec-ch-ua-platform': '"Windows"',
                'sec-fetch-dest': 'empty',
                'sec-fetch-mode': 'cors',
                'sec-fetch-site': 'same-origin',
                'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36 Edg/135.0.0.0',
                'x-requested-with': 'XMLHttpRequest',
            }
            yield scrapy.Request(route_url,callback=self.parse_route_pages,headers=headers,meta={"mdd":mdd})

        next_page = page + 1
        if next_page <= 15:
            yield from self.fetch_page(next_page)
        else:
            # with open(self.save_path, "a", encoding="utf-8") as f:
            #     f.write("</body></html>")
            print(f"所有页面爬取完成热门城市目的地的数据爬取完毕")

    # 先post得到包含了景点的页面源码数据
    def parse_mdd(self, response):
        mdd = response.meta["mdd"]
        data = {
            'sAct':'KMdd_StructWebAjax|GetPoisByTag',
            'iTagId':'0',
            'iMddid':mdd["mddId"],
            'iPage':'1',
            '_ts':str(int(time.time() * 1000))
        }
        data['_sn'] = self.js_ctx.call('getSn', data)
        yield FormRequest("https://www.mafengwo.cn/ajax/router.php",
                          headers=self.headers,
                          callback=self.parse_mdd_poi,
                          formdata=data,
                          meta={"mdd": mdd})
    # 处理ajax请求,获得该目的地的15个景点
    def parse_mdd_poi(self,response):
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
            # poi_title = None
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
                yield scrapy.Request(
                    url="https://www.mafengwo.cn/" + poi_url,
                    callback=self.parse_poi_detail,
                    headers=self.headers,
                    meta={"scenic": scenic}
                )
            else:
                print(f"Missing poi_url or poi_title in li: {li.get()}")
        mdd["poi_list"] = poi_list
        # print(mdd)
        yield mdd

    # def parse_mdd(self,response):
    #     mdd = response.meta["mdd"]
    #     selector = Selector(text=response.text)
    #     div_lists = selector.xpath('//*[@id="container"]/div[@class="row row-top5"]/div/div')
    #     TOP5 = []
    #     for div in div_lists:
    #         scenic = Scenic()
    #         poi_url = div.xpath('./div[1]/div/h3/a[1]/@href').get()
    #         poi_title = div.xpath('./div[1]/div/h3/a[1]/@title').get()
    #
    #         # 空值判断
    #         if poi_url and poi_title:
    #             poi_id = poi_url.split('/')[-1].replace('.html', '')
    #             scenic["poi_id"] = poi_id
    #             scenic["poi_title"] = poi_title
    #             scenic["mddId"] = mdd["mddId"]
    #
    #             TOP5.append({"poi_id": scenic["poi_id"], "poi_title": scenic["poi_title"]})
    #             yield scrapy.Request(
    #                 url="https://www.mafengwo.cn/" + poi_url,
    #                 callback=self.parse_poi_detail,
    #                 headers=self.headers,
    #                 meta={"scenic": scenic}
    #             )
    #         else:
    #             print(f"Missing poi_url or poi_title in div: {div.get()}")
    #     mdd["TOP5_list"] = TOP5
    #     # print(mdd)
    #     yield mdd

    # 爬取该页面的所有线路的详情信息
    def parse_route_pages(self, response):
        mdd = response.meta["mdd"]
        page_text = response.text
        selector = Selector(text=page_text)
        # num_page_text = selector.xpath('//*[@id="routelistpagination"]/div/span[@class="count"]/text()').get()
        # num_page_text = selector.xpath('//*[@id="routelistpagination"]/div/span[1]/text()').get()
        # print(f'当前爬取的线路: {num_page_text}')
        # num_pages = int(re.search(r'\d+', num_page_text).group())    # 得到页面数量
        num_page_text = selector.xpath('//*[@id="routelistpagination"]/div/span[@class="count"]/text()').get()
        print(f'当前爬取的线路: {num_page_text}')
        if num_page_text:
            num_pages = int(re.search(r'\d+', num_page_text).group())
        else:
            self.logger.warning(f'未找到页数信息：{response.url}')
            with open(f'debug_{mdd["mddTitle"]}.html', 'w', encoding='utf-8') as f:
                f.write(response.text)
            num_pages = 1  # 默认设为1页防止出错
        print(f'当前爬取的线路的起点是{mdd["mddTitle"]}，当前要爬取的线路详情页共有{num_pages}页')
        if num_pages == 1:      # 若只有一页那么页面数据就在page_text中
            div_list = selector.xpath('//*[@id="routebody"]/div[1]')
            for div in div_list:
                route = Route()

                route["mddId"] = mdd["mddId"]
                route["mddTitle"] = mdd["mddTitle"]
                route["routeTitle"] = div.xpath('./div/div/dl[1]/dt/a/h2/text()').get()
                route["routeId"] = div.xpath('./div/div/dl[1]/dd/p/a/@href').get().split("/")[-1].replace('.html','')
                route["days"] = int(re.search(r'\d+',div.xpath('./div/div/dl[1]/dt/div/span[1]/strong/text()').get()).group())

                route_detail_url = "https://www.mafengwo.cn/" + div.xpath('./div/div/dl[1]/dd/p/a/@href').get()
                yield scrapy.Request(route_detail_url,callback=self.parse_route_detail,meta={"route":route})
        else:       # 若页面数大于1，那么页面数据在ajax请求中，需要像前面的目的地页面的请求过程一样处理
            for page in range(1, num_pages + 1):
                data = {
                    "mddid": str(mdd["mddId"]),
                    "page": str(page),
                    "_ts": str(int(time.time() * 1000)),
                }
                data['_sn'] = self.js_ctx.call('getSn', data)
                # data['_sn'] = execjs.compile(open("../../jscode.js").read()).call('getSn',data)
                headers = {
                    'accept': 'application/json, text/javascript, */*; q=0.01',
                    'accept-language': 'zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6',
                    'cache-control': 'no-cache',
                    'content-type': 'application/x-www-form-urlencoded; charset=UTF-8',
                    'origin': 'https://www.mafengwo.cn',
                    'pragma': 'no-cache',
                    'priority': 'u=1, i',
                    'referer': 'https://www.mafengwo.cn/mdd/route/' + mdd["mddId"] + '.html',
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
                    meta={"mdd": mdd}
                )

    # 依次获取多页的每一页的信息，然后像单页一样处理
    def parse_ajax_route_list(self, response):
        mdd = response.meta["mdd"]
        try:
            json_data = json.loads(response.text)
        except json.JSONDecodeError:
            self.logger.warning("JSON decode failed: %s", response.text[:200])
            return

        html_content = json_data.get("list", "")
        selector = Selector(text=html_content)
        div_list = selector.xpath('//div[contains(@class,"row row-line")]')

        for div in div_list:
            route = Route()
            route["mddId"] = mdd["mddId"]
            route["mddTitle"] = mdd["mddTitle"]
            route["routeTitle"] = div.xpath('./div/div/dl[1]/dt/a/h2/text()').get()
            href = div.xpath('./div/div/dl[1]/dd/p/a/@href').get()
            if not href:
                continue
            route["routeId"] = href.split('/')[-1].replace('.html', '')
            days_str = div.xpath('./div/div/dl[1]/dt/div/span[1]/strong/text()').get()
            route["days"] = int(re.search(r'\d+', days_str).group()) if days_str else 0
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
            day = f'D{idx+1}'  # 第几天，例如:D1

            # mdd_href = day_div.xpath('.//div[@class="day-hd"]//span[contains(@class, "place ")]/a/@href').get()
            # mdd_name = day_div.xpath('.//div[@class="day-hd"]//span[contains(@class, "place ")]/a/text()').get()
            # 获取当天的所有 mdd（目的地）
            # mdd_nodes = day_div.xpath('.//div[@class="day-hd"]//span[contains(@class, "place ")]/a')
            # mdd_list = []
            # if not mdd_nodes:
            #     mdd_list.append({"mddId": route["mddId"], "mddTitle": route["mddTitle"]})
            # for mdd_node in mdd_nodes:
            #     mdd_href = mdd_node.xpath('./@href').get()
            #     mdd_name = mdd_node.xpath('./text()').get()
            #
            #     if idx == 0:
            #         # 如果第一天出发地一定是这个线路的mddId和mddTitle
            #         day_route["day"] = day
            #         mdd_from_id = route["mddId"]
            #         mdd_from_name = route["mddTitle"]
            #         mdd_list.append({"mddId":mdd_from_id,"mddTitle":mdd_from_name})
            #     else:
            #         # 第二天及以后，前一天的目的地为mdd_from
            #         prev_day = daily_routes[idx - 1]
            #         mdd_from_id = prev_day["mdd_list"][-1].get("mddId")
            #         mdd_from_name = prev_day["mdd_list"][-1].get("mddTitle")
            #         mdd_list.append({"mddId": mdd_from_id, "mddTitle": mdd_from_name})
            #     if mdd_href and mdd_name:
            #         mddId = mdd_href.split('/')[-1].replace('.html', '').strip()
            #         mddTitle = re.sub(r'\s+', ' ', mdd_name).strip()  # 清理多余空白字符
            #         mdd_list.append({"mddId": mddId, "mddTitle": mddTitle})
            #         mdd = Mdd()
            #         mdd["mddId"] = mddId
            #         mdd["mddTitle"] = mddTitle
            #         mdd_url = "https://www.mafengwo.cn/jd/" + mdd["mddId"] + "/gonglve.html"
            #         yield scrapy.Request(mdd_url, callback=self.parse_mdd, headers=self.headers, meta={"mdd": mdd})

            # 获取当天所有的景点
            poi_nodes = day_div.xpath('.//div[@class="poi-name"]//li/span[@class="place"]')
            poi_list = []
            for poi in poi_nodes:
                scenic = Scenic()
                poi_href = poi.xpath('./a/@href').get()
                poi_id = poi_href.split('/')[-1].replace('.html', '')
                poi_title = poi.xpath('./a/text()').get()
                poi_title = re.sub(r'\s+', ' ', poi_title).strip()  # 清理多余空白字符
                data_time = poi.xpath('./a/@data-time').get()

                if poi_id and poi_title:
                    scenic["poi_id"] = poi_id
                    scenic["poi_title"] = poi_title
                    yield scrapy.Request("https://www.mafengwo.cn/" + poi_href, callback=self.parse_poi_detail,headers=self.headers, meta={"scenic": scenic})
                    poi_list.append({"poi_id": poi_id, "poi_title": poi_title,"time":data_time})
            day_route["day"] = day
            # day_route["mdd_list"] = mdd_list
            day_route["poi_list"] = poi_list
            daily_routes.append(day_route)

        route["daily_routes"] = daily_routes
        yield route

        # profile_routes = []
        # # 处理线路概况
        # for day_div in day_divs:
        #     day_route = []
        #     spot_links = day_div.xpath('.//a')
        #     for spot in spot_links:
        #         title = spot.xpath('./text()').get()
        #         href = spot.xpath('./@href').get()
        #
        #         if href:
        #             id = href.split('/')[-1].replace(".html",'')
        #             if "/travel-scenic-spot/mafengwo/" in href: # 属于mdd
        #                 mdd = Mdd()
        #                 mdd["mddId"] = id
        #                 mdd["mddTitle"] = title
        #                 day_route.append({'mddId': mdd["mddId"], 'title': mdd["mddTitle"]})
        #                 mdd_url = "https://www.mafengwo.cn/jd/"+mdd["mddId"]+"/gonglve.html"
        #                 yield scrapy.Request(mdd_url,callback=self.parse_mdd,headers=self.headers,meta={"mdd":mdd})
        #             elif "/poi/" in href:   # 属于景点
        #                 scenic = Scenic()
        #                 scenic["poi_id"] = id
        #                 normalized_title = re.sub(r"\s*[\(\（].*?[\)\）]", "", title)
        #                 scenic["poi_title"] = normalized_title
        #                 day_route.append({'poi_id': scenic["poi_id"], 'poi_title': scenic["poi_title"]})
        #                 scenic_url = "https://www.mafengwo.cn/"+href
        #                 yield scrapy.Request(scenic_url,callback=self.parse_poi_detail,headers=self.headers,meta={"scenic":scenic})
        #     # 如果这一天至少包含一个景点，则加入总列表
        #     if day_route:
        #         profile_routes.append(day_route)
        # # 最终 yield 路线对象
        # # print(route)
        # route["profile_routes"] = profile_routes
        # yield route
    # 得到景点的详情介绍
    def parse_poi_detail(self, response):
        scenic = response.meta["scenic"]
        selector = Selector(text=response.text)

        # 提取 details 内容
        detail_text = self.extract_details(selector)
        scenic["details"] = detail_text

        # 提取 city_href（先第二种，再第一种）
        city_below_href = selector.xpath('//div[@class="crumb"]//span/a/@href').get()   # 地级市的位置
        if city_below_href is None:
            city_below_href = selector.xpath('//div[contains(@class, "top-info")]//span/a/@href').get()

        if city_below_href:
            belong_url = 'https://www.mafengwo.cn/'+city_below_href
            yield scrapy.Request(belong_url,callback=self.parse_poi_city,headers=self.headers,meta={"scenic":scenic})
            # mddId = city_href.split('/')[-1].replace('.html', '')
            # scenic["mddId"] = mddId
            #
            # # 请求 MDD 页面
            # mdd = Mdd()
            # mdd["mddId"] = mddId
            # mdd_url = f"https://www.mafengwo.cn/jd/{mddId}/gonglve.html"
            # yield scrapy.Request(mdd_url, callback=self.parse_mdd, headers=self.headers, meta={"mdd": mdd})
        # else:
        #     scenic["mddId"] = None  # 防止后续 KeyError

    def parse_poi_city(self, response):
        scenic = response.meta["scenic"]
        # print(response.text)
        selector = Selector(text=response.text)
        city_href = selector.xpath('//*[@id="container"]/div[1]/div/div[2]/div[3]/div/span/a/@href').get()
        mddId = city_href.split('/')[-1].replace('.html', '')
        city_name = selector.xpath('//*[@id="container"]/div[1]/div/div[2]/div[3]/div/span/a/text()').get()
        scenic["mddId"] = mddId
        yield scenic
        # 请求 MDD 页面
        mdd = Mdd()
        mdd["mddId"] = mddId
        mdd["mddTitle"] = city_name
        mdd_url = f"https://www.mafengwo.cn/jd/{mddId}/gonglve.html"
        yield scrapy.Request(mdd_url, callback=self.parse_mdd, headers=self.headers, meta={"mdd": mdd})
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
    # def parse_poi_detail(self, response):
    #     scenic = response.meta["scenic"]
    #     selector = Selector(text=response.text)
    #     city_href = selector.xpath('//div[@class="crumb"]//span/a/@href').get()
    #     if city_href is None:
    #         city_href = selector.xpath('//div[contains(@class, "top-info")]//span/a/@href').get()
    #
    #     if city_href:
    #         mddId = city_href.split('/')[-1].replace('.html', '')
    #         scenic["mddId"] = mddId
    #
    #         # 继续处理
    #         mdd = Mdd()
    #         mdd["mddId"] = mddId
    #         mdd_url = f"https://www.mafengwo.cn/jd/{mddId}/gonglve.html"
    #         yield scrapy.Request(mdd_url, callback=self.parse_mdd, headers=self.headers, meta={"mdd": mdd})
    #     else:
    #         self.logger.warning(f"[parse_poi_detail] 未找到 city_href，URL: {response.url}")
    #
    #     raw_text_list = selector.xpath('./body/div[2]/div[3]/div[2]/div[@class="summary"]//text()').getall()
    #     clean_text = ''.join([text.strip() for text in raw_text_list if text.strip()])
    #     scenic["details"] = clean_text
    #     # 尝试从第一个 XPath 表达式提取评论数量
    #     # num_str = selector.xpath('//*[@id="poi-navbar"]/ul/li[3]/a/span/text()').get()
    #     # # 如果第一个 XPath 提取结果为 None，再尝试第二个 XPath 表达式
    #     # if not num_str:
    #     #     num_str = selector.xpath('//*[@id="comment_header"]/div[1]/span[1]/text()').get()
    #     # # 如果两个 XPath 都没有提取到数据，则设置为默认值 0 或其他标志值
    #     # num_str = num_str.strip() if num_str else '0'
    #     # scenic["favorable_num"] = int(re.search(r'\d+', num_str).group())
    #     # print(scenic)
    #     yield scenic