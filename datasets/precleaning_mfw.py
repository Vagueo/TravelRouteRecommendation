'''
    处理过程：【注意是有先后顺序的，第一步删除完后的文件再来处理第二步】
    （1）把mdd中没有poi_list的mdd删除;mdd中的poi_list中的poi不在scenic中的poi删除
    （2）把scenic.jsonl中的poi对应的city_id不在mdd.jsonl中的poi删除
    【舍弃】（3）给刚刚处理完的mdd.jsonl和scenic.jsonl中的mdd和poi加上经纬度信息，没有的会自动抛弃
    （4）把route.jsonl中mdd_list和poi_list分别在mdd.jsonl和scenic.jsonl没有的mdd和poi删除,并且删除完后poi_list < 3的部分都删除
    （5）实现将每一天的线路拆分开，七天的线路拆成七条单独的线路：
        1）其中mdd在scenic前面，mdd不是mdd_list中的，而是poi对应的city_id,若相邻的几个地点的mdd都相同，那么就把这个mdd插在最前面的这个poi前面
        （例如：有一个poi_list依次为a，b，c，其中a属于A这个城市，b，c都属于B这个城市，那么就把A这个城市插到a前面，B插入到b前面）
        2）拆分出来的线路的地点数量 < 3的话通过merge_routes合并
        3）并且拆分出来的线路的routeId改为原routeId_x（x为天数）
    【舍弃，只先进行数据清洗】（6）加入BERT模型处理details得到三个符合poi的描述，mdd的描述就通过该城市的poi_list的所有描述聚类得到三个最符合的描述，将其作为新的details
    （7）最后得到处理完后的数据，数据的结构为：
        1）user_id是我们自行从0到1生成，并不代表这是用户数据，只是为了统一处理，将热门旅游线路和FourSquare统一处理
        2）trajectory为线路
    {
        "route_id": ,
        "trajectory": [
          {
            "type": "mdd",
            "mddId": mddId,
            "mddTitle": mddTitle,
            "latitude": 39.929004,
            "longitude": 32.852998
            "details": details
            
          },
          {
            "type": "poi",
            "poi_id": poi_id,
            "poi_title": poi_title,
            "mddId": mddId,
            "mddTitle": mddTitle,
            "latitude": 39.926852000000004,
            "longitude": 32.851782
            "details": details

          },
          ....
        ]
    }
'''
import json

# Step 0: 加载数据
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# 加载初始数据
mdd_data = load_jsonl("./raw/mdd.jsonl")
scenic_data = load_jsonl("./raw/scenic.jsonl")
route_data = load_jsonl("./raw/route.jsonl")


# >>> 新增：Step 0.5 清洗重复的路线（routeTitle相同且daily_routes相同的，只保留一条）
def remove_duplicate_routes(route_data):
    title_to_route = {}
    for route in route_data:
        title = route.get("routeTitle", "")
        daily_routes = route.get("daily_routes", [])
        # 把 daily_routes 转成字符串来比较是否相同
        daily_routes_str = json.dumps(daily_routes, sort_keys=True, ensure_ascii=False)

        if title not in title_to_route:
            title_to_route[title] = (daily_routes_str, route)
        else:
            existing_daily_routes_str, _ = title_to_route[title]
            if existing_daily_routes_str != daily_routes_str:
                # 如果标题一样但是内容不同，可以选择保留多个，或者打警告。这里暂时保留第一个。
                print(f"[警告] routeTitle重复但daily_routes不同：{title}")

    deduplicated_routes = [item[1] for item in title_to_route.values()]
    return deduplicated_routes


route_data = remove_duplicate_routes(route_data)
save_jsonl("./precleaning/MFW/route_step0_deduplicated.jsonl", route_data)
print(f"完成去除重复路线，保留了 {len(route_data)} 条路线，输出位置：./precleaning/MFW/route_step0_deduplicated.jsonl")

# Step 1：清洗不在 scenic 中的 poi以及mdd 中没有 poi_list 的城市
# 提取 scenic_data 中的合法 poi_id
scenic_poi_ids = {poi["poi_id"] for poi in scenic_data}
clean_mdd_data = []
for mdd in mdd_data:
    if "poi_list" in mdd:
        # 过滤掉不在 scenic_poi_ids 中的 poi
        filtered_poi_list = [poi for poi in mdd["poi_list"] if poi.get("poi_id") in scenic_poi_ids]
        if filtered_poi_list:  # 只保留 poi_list 非空的条目
            mdd["poi_list"] = filtered_poi_list
            clean_mdd_data.append(mdd)
# 保存清洗后的结果
save_jsonl("./precleaning/MFW/mdd_step1.jsonl", clean_mdd_data)
print("完成对 mdd 的清洗，输出位置：./precleaning/MFW/mdd_step1.jsonl")

# Step 2：清洗 scenic 中 mddId 不在 mdd 中的 poi
valid_mdd_ids = {mdd["mddId"] for mdd in clean_mdd_data}
clean_scenic_data = [poi for poi in scenic_data if poi["mddId"] in valid_mdd_ids]
save_jsonl("./precleaning/MFW/scenic_step2.jsonl", clean_scenic_data)
print(f"完成清洗 city_id 不在 mdd 中的 poi，输出位置：./precleaning/MFW/scenic_step2.jsonl")

# Step 3：为 MDD 添加 details 字段，聚合其 poi_list 中所有 POI 的 details
def ensure_period(text):
    return text if text.endswith("。") else text + "。"
poi_details_map = {poi["poi_id"]: poi["details"] for poi in clean_scenic_data}

for mdd in clean_mdd_data:
    mdd["details"] = "".join(
        [ensure_period(poi_details_map[poi["poi_id"]])
         for poi in mdd["poi_list"] if poi["poi_id"] in poi_details_map])
    # mdd["details"] = " ".join(
    #     [poi_details_map[poi["poi_id"]] for poi in mdd["poi_list"] if poi["poi_id"] in poi_details_map])

save_jsonl("./precleaning/MFW/mdd_step3_with_details.jsonl", clean_mdd_data)
print(f"完成为 MDD 聚合 POI 描述，输出位置：./precleaning/MFW/mdd_step3_with_details.jsonl")

# Step 4：按天拆线路，插入 mdd，合并短线路
def split_and_format_routes(routes, poi_to_mdd):
    final_routes = []

    for route in routes:
        route_id = route["routeId"]
        daily_lists = route.get("daily_routes", [])
        num_days = len(daily_lists)
        day = 0

        while day < num_days:
            merged_traj = []
            start_day = day
            prev_mdd = None

            # 尝试从 start_day 开始合并若干天，直到轨迹数 ≥ 3 或天数用尽
            while day < num_days:
                daily_route = daily_lists[day]

                if len(daily_route.get("poi_list", [])) < 1:
                    print(f"[调试] 路线 {route_id} 第 {day + 1} 天的 POI 列表为空，跳过该天")
                    day += 1
                    continue

                for poi in daily_route["poi_list"]:
                    poi_id = poi["poi_id"]
                    mdd_id = poi_to_mdd.get(poi_id)

                    # 插入 mdd
                    if mdd_id != prev_mdd:
                        mdd_obj = next((m for m in clean_mdd_data if m["mddId"] == mdd_id), None)
                        if mdd_obj and mdd_obj.get("mddId") and mdd_obj.get("mddTitle"):
                            merged_traj.append({
                                "type": "mdd",
                                "mddId": mdd_obj["mddId"],
                                "mddTitle": mdd_obj["mddTitle"],
                                "details": mdd_obj.get("details", "")
                            })
                            prev_mdd = mdd_id

                    # 插入 poi
                    poi_obj = next((p for p in clean_scenic_data if p["poi_id"] == poi_id), None)
                    if poi_obj and poi_obj.get("poi_id") and poi_obj.get("poi_title") and poi_obj.get("details"):
                        merged_traj.append({
                            "type": "poi",
                            "poi_id": poi_obj["poi_id"],
                            "poi_title": poi_obj["poi_title"],
                            "mddId": poi_obj["mddId"],
                            "mddTitle": poi_obj["mddTitle"],
                            "details": poi_obj["details"]
                        })
                    else:
                        print(
                            f"[警告] 路线 {route_id} 第 {day + 1} 天，跳过无效 POI：{poi_id} - {poi.get('poi_title', '未知')}")
                day += 1

                if len(merged_traj) >= 3:
                    final_routes.append({
                        "route_id": f"{route_id}_{start_day + 1}",  # 注意 day+1 是合并起始天数
                        "trajectory": merged_traj
                    })
                    break  # 成功合并一条，跳出合并循环，继续下一段

            # 如果合并到最后都不够3，放弃该段
            if len(merged_traj) < 3:
                print(f"[调试] 路线 {route_id} 第 {start_day + 1} 天起始合并后仍不足 3 个点，丢弃该段")

    return final_routes

# def split_and_format_routes(routes, poi_to_mdd):
#     final_routes = []
#     for route in routes:
#         route_id = route["routeId"]
#         daily_lists = route.get("daily_routes", [])
#         for day_idx, daily_route in enumerate(daily_lists):
#             if len(daily_route["poi_list"]) < 1:
#                 print(f"[调试] 路线 {route_id} 第 {day_idx + 1} 天的 POI 列表为空，跳过该天")
#                 continue
#
#             traj = []
#             prev_mdd = None
#             for poi in daily_route["poi_list"]:
#                 poi_id = poi["poi_id"]
#                 mdd_id = poi_to_mdd.get(poi_id)
#
#                 if mdd_id != prev_mdd:
#                     mdd_obj = next((m for m in clean_mdd_data if m["mddId"] == mdd_id), None)
#                     if mdd_obj and mdd_obj.get("mddId") and mdd_obj.get("mddTitle"):
#                         traj.append({
#                             "type": "mdd",
#                             "mddId": mdd_obj["mddId"],
#                             "mddTitle": mdd_obj["mddTitle"],
#                             "details": mdd_obj.get("details", "")
#                         })
#                         prev_mdd = mdd_id
#
#                 poi_obj = next((p for p in clean_scenic_data if p["poi_id"] == poi_id), None)
#                 if poi_obj and poi_obj.get("poi_id") and poi_obj.get("poi_title") and poi_obj.get("details") and poi_obj.get("mddId") and poi_obj.get("mddTitle"):
#                     traj.append({
#                         "type": "poi",
#                         "poi_id": poi_obj["poi_id"],
#                         "poi_title": poi_obj["poi_title"],
#                         "mddId": poi_obj["mddId"],
#                         "mddTitle": poi_obj["mddTitle"],
#                         "details": poi_obj["details"]
#                     })
#                 else:
#                     print(
#                         f"[警告] 路线 {route_id} 第 {day_idx + 1} 天，跳过无效 POI：{poi_id} - {poi.get('poi_title', '未知')}")
#
#             if len(traj) < 3:
#                 print(f"[调试] 路线 {route_id} 第 {day_idx + 1} 天的 POI 数量少于 3，跳过该天")
#                 continue
#
#             final_routes.append({
#                 "route_id": f"{route_id}_{day_idx + 1}",
#                 "trajectory": traj
#             })
#
#     return final_routes


poi_to_mdd = {poi["poi_id"]: poi["mddId"] for poi in clean_scenic_data}
processed_routes = split_and_format_routes(route_data, poi_to_mdd)
save_jsonl("./precleaning/MFW/route_step4.jsonl", processed_routes)
print("线路按天数拆分完成，输出位置：./precleaning/MFW/route_step4.jsonl")
