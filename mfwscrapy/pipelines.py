import json
from mfwscrapy.items import Mdd, Route, Scenic
from scrapy.exceptions import DropItem

class BasePipeline:
    def __init__(self, file_path, unique_key, required_fields, allowed_item_type=None, buffer_size=10):
        self.file_path = file_path
        self.unique_key = unique_key
        self.required_fields = required_fields
        self.allowed_item_type = allowed_item_type      # 判断是哪种item
        self.buffer_size = buffer_size

        self.seen_ids = self.load_seen_ids()
        self.file = open(self.file_path, "a", encoding="utf-8")
        self.buffer = []

    def load_seen_ids(self):
        seen_ids = set()
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    seen_ids.add(data.get(self.unique_key))
        except FileNotFoundError:
            pass
        return seen_ids

    def process_item(self, item, spider):
        # 跳过不是指定类型的 item
        if self.allowed_item_type and not isinstance(item, self.allowed_item_type):
            return item

        item_dict = dict(item)
        item_id = item_dict.get(self.unique_key)

        print(f"[DEBUG] 处理 item: {item}")

        missing_fields = [field for field in self.required_fields if field not in item_dict]
        if missing_fields:
            msg = f"[ERROR]缺字段：{missing_fields} in item: {item}"
            print(msg)
            raise DropItem(msg)  # 正式丢弃

        if item_id not in self.seen_ids:
            self.seen_ids.add(item_id)
            self.buffer.append(json.dumps(item_dict, ensure_ascii=False) + "\n")
            if len(self.buffer) >= self.buffer_size:
                self.flush_buffer()
        return item

    def flush_buffer(self):
        self.file.writelines(self.buffer)
        self.file.flush()
        self.buffer.clear()

    def close_spider(self, spider):
        if self.buffer:
            self.flush_buffer()
        self.file.close()

class MddPipeline(BasePipeline):
    def __init__(self):
        super().__init__(
            file_path="D:/Project/pythonProject/mfwscrapy/datasets/raw/mdd.jsonl",
            unique_key="mddId",
            required_fields=["mddId", "mddTitle"],
            allowed_item_type=Mdd
        )

    def process_item(self, item, spider):
        # 如果 poi_list 存在但为空，则丢弃该 item
        if isinstance(item, Mdd) and (not item.get("poi_list")):
            msg = f"[DROP] mddId={item.get('mddId')} 的 poi_list 为空，已丢弃"
            print(msg)
            raise DropItem(msg)

        return super().process_item(item, spider)

class RoutePipeline(BasePipeline):
    def __init__(self):
        super().__init__(
            file_path="D:/Project/pythonProject/mfwscrapy/datasets/raw/route.jsonl",
            unique_key="routeId",
            required_fields=["routeId", "routeTitle"],
            allowed_item_type=Route
        )

class ScenicPipeline(BasePipeline):
    def __init__(self):
        super().__init__(
            file_path="D:/Project/pythonProject/mfwscrapy/datasets/raw/scenic.jsonl",
            unique_key="poi_id",
            required_fields=["poi_id", "city_id", "poi_title"],
            allowed_item_type=Scenic
        )

    def process_item(self, item, spider):
        # 丢弃 details 字段为空或无实际描述的 scenic 项
        if isinstance(item, Scenic):
            details = item.get("details", "").strip()
            title = item.get("poi_title", "").strip()
            if not details or len(details) <= 5 or details == title:
                msg = f"[DROP] poi_id={item.get('poi_id')} 的 details 无效（为空/与标题相同），已丢弃"
                print(msg)
                raise DropItem(msg)

        return super().process_item(item, spider)
