import os
import json
import time
import requests

BAIDU_API_URL = "http://api.map.baidu.com/geocoding/v3/"
AK = ""  # <-- 替换为你自己的 AK

def get_location(name):
    try:
        params = {
            'address': name,
            'output': 'json',
            'ak': AK
        }
        response = requests.get(BAIDU_API_URL, params=params, timeout=5)
        result = response.json()
        if result.get("status") == 0:
            location = result["result"]["location"]
            return location["lat"], location["lng"]
        else:
            print(f"❌ 未找到：{name}，返回：{result}")
            return None, None
    except Exception as e:
        print(f"⚠️ 错误处理 {name}：{e}")
        return None, None

def process_jsonl_file(file_path, type_key, delay=1):
    seen_ids = set()

    # 构建输出路径
    base_name = os.path.basename(file_path).replace('.jsonl', '_with_location.jsonl')
    output_path = os.path.join("./precleaning/MFW", base_name)

    # 如果已有输出文件，则先加载已存在的 ID，避免重复写入
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f_existing:
            for line in f_existing:
                try:
                    obj = json.loads(line.strip())
                    seen_ids.add(obj["id"])
                except:
                    continue

    with open(file_path, 'r', encoding='utf-8') as f_in, open(output_path, 'a', encoding='utf-8') as f_out:
        for idx, line in enumerate(f_in, 1):
            try:
                data = json.loads(line.strip())
                if type_key == 'poi':
                    if 'poi_title' not in data or 'poi_id' not in data:
                        print(f"⚠️ 第{idx}行缺少 poi_title 或 poi_id，跳过。内容：{data}")
                        continue
                    name = data["poi_title"]
                    id = data["poi_id"]
                    details = data["details"]
                    poi_list = data["poi_list"]
                elif type_key == 'mdd':
                    if 'mddTitle' not in data or 'mddId' not in data or 'details' not in data:
                        print(f"⚠️ 第{idx}行缺少 mddTitle 或 mddId或details，跳过。内容：{data}")
                        continue
                    name = data["mddTitle"]
                    id = data["mddId"]
                    details = data["details"]
                else:
                    continue

                if id in seen_ids:
                    print(f"⚠️ 第{idx}行重复 ID：{id}，跳过")
                    continue
                seen_ids.add(id)

                lat, lng = get_location(name)
                if lat is not None and lng is not None:
                    loc = {
                        "mddId": id,
                        "mddTitle": name,
                        "poi_list": poi_list,
                        "details": details,
                        "latitude": lat,
                        "longitude": lng
                    }
                    json.dump(loc, f_out, ensure_ascii=False)
                    f_out.write('\n')
                    print(f"✅ 成功写入第{idx}条数据：{id} - {name}，({lat}, {lng})")
                time.sleep(delay)

            except Exception as e:
                print(f"❌ 第{idx}行处理失败：{e}，原始内容：{line.strip()}")

    print(f"✅ 文件处理完成，输出位置：{output_path}")

# 执行
process_jsonl_file("./precleaning/MFW/mdd_step3_with_details.jsonl", "mdd", delay=1)
