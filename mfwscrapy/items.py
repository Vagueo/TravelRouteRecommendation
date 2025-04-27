import scrapy
# 目的地
class Mdd(scrapy.Item):
    # _id = scrapy.Field()
    mddId = scrapy.Field()      # 目的地的ID
    mddTitle = scrapy.Field()   # 目的地的名称
    # details = scrapy.Field()
    poi_list = scrapy.Field()   # 目的地的景点列表
    # status = scrapy.Field()
    # createTime = scrapy.Field()
    # updateTime = scrapy.Field()

# 旅游路线
class Route(scrapy.Item):
    # _id = scrapy.Field()  # 路线唯一ID
    routeId = scrapy.Field()  # 旅游路线ID
    routeTitle = scrapy.Field()  # 旅游路线名称
    mddId = scrapy.Field()  # 目的地ID（例如北京 10065）
    mddTitle = scrapy.Field()   # 目的地名称
    # profile_routes = scrapy.Field()  # 该路线的概况
    daily_routes = scrapy.Field()   # 该线路每天的详细路线
    days = scrapy.Field()  # 行程天数

    # status = scrapy.Field()  # 记录爬取状态
    # createTime = scrapy.Field()  # 记录创建时间
    # updateTime = scrapy.Field()  # 记录更新时间

# 景点
class Scenic(scrapy.Item):
    # _id=scrapy.Field()
    poi_id = scrapy.Field()     # 景点唯一ID
    poi_title = scrapy.Field()  # 景点名称
    details = scrapy.Field()    # 景点详细描述
    city_id = scrapy.Field()    # 景点所在城市的ID
    # favorable_num = scrapy.Field()
    # status = scrapy.Field()
    # createTime = scrapy.Field()
    # updateTime = scrapy.Field()





