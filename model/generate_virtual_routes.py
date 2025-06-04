import json
import random
import math
import os

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def build_poi_dict(poi_list):
    """用poi_id为key建立poi详情字典，方便查询"""
    return {poi['poi_id']: poi for poi in poi_list}

def haversine(lat1, lon1, lat2, lon2):
    """计算两个坐标之间的球面距离（单位：km）"""
    R = 6371.0  # 地球半径，单位km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = math.sin(d_phi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c

def generate_virtual_routes(
    mdd_data,
    poi_dict,
    num_routes=100,
    max_mdd_per_route=3,
    max_poi_per_mdd=5,
    distance_threshold_km=1500,
    min_total_nodes=10,
    max_total_nodes=12
):
    """
    生成虚拟线路（限制地理距离，并将每条线路节点总数限制在 [10, 12] 区间内）
    """
    routes = []
    all_mdds = [mdd for mdd in mdd_data if 'latitude' in mdd and 'longitude' in mdd and mdd.get('poi_list')]
    attempts = 0
    max_attempts = num_routes * 20
    min_mdd_count = 1

    while len(routes) < num_routes and attempts < max_attempts:
        attempts += 1
        trajectory = []
        visited_mdd_ids = set()
        total_node_count = 0

        current_mdd = random.choice(all_mdds)
        visited_mdd_ids.add(current_mdd['mddId'])

        for _ in range(max_mdd_per_route):
            if total_node_count >= max_total_nodes:
                break

            # 添加当前城市节点
            trajectory.append({
                "type": "mdd",
                "mddId": current_mdd['mddId'],
                "mddTitle": current_mdd['mddTitle'],
                "details": current_mdd.get('details', "")
            })
            total_node_count += 1

            # 添加当前城市的 POI 节点
            mdd_poi_list = current_mdd.get('poi_list', [])
            if mdd_poi_list and total_node_count < max_total_nodes:
                num_poi = random.randint(1, min(len(mdd_poi_list), max_poi_per_mdd))
                num_poi = min(num_poi, max_total_nodes - total_node_count)  # 确保不超过最大节点数
                selected_pois = random.sample(mdd_poi_list, num_poi)

                for poi_ref in selected_pois:
                    if total_node_count >= max_total_nodes:
                        break

                    poi_id = poi_ref['poi_id']
                    if poi_id in poi_dict:
                        poi = poi_dict[poi_id]
                        trajectory.append({
                            "type": "poi",
                            "poi_id": poi['poi_id'],
                            "poi_title": poi['poi_title'],
                            "mddId": current_mdd['mddId'],
                            "mddTitle": current_mdd['mddTitle'],
                            "details": poi.get('details', "")
                        })
                        total_node_count += 1

            # 选择下一个城市（在距离限制内，且未访问过）
            candidates = [
                m for m in all_mdds
                if m['mddId'] not in visited_mdd_ids and
                   haversine(current_mdd['latitude'], current_mdd['longitude'], m['latitude'], m['longitude']) < distance_threshold_km
            ]
            if not candidates:
                break
            current_mdd = random.choice(candidates)
            visited_mdd_ids.add(current_mdd['mddId'])

        # 筛选满足最小节点数量要求的轨迹
        if min_total_nodes <= len(trajectory) <= max_total_nodes:
            routes.append({
                "route_id": len(routes),
                "trajectory": trajectory
            })

    return routes

def load_real_routes(file_path):
    """加载真实轨迹数据"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    real_routes = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            real_routes.append(json.loads(line))
    return real_routes

def extract_node_ids(route):
    """提取MDD和POI节点ID集合，用于相似度计算"""
    node_ids = set()
    for item in route.get('trajectory', []):
        if item['type'] == 'mdd':
            node_ids.add(f"mdd_{item['mddId']}")
        elif item['type'] == 'poi':
            node_ids.add(f"poi_{item['poi_id']}")
    return node_ids

def jaccard_similarity(set1, set2):
    """计算两个集合的 Jaccard 相似度"""
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union != 0 else 0.0

def extract_node_ids_sequence(route):
    """提取轨迹中节点 ID 的有序序列"""
    ids = []
    for item in route.get('trajectory', []):
        if item['type'] == 'mdd':
            ids.append(f"mdd_{item['mddId']}")
        elif item['type'] == 'poi':
            ids.append(f"poi_{item['poi_id']}")
    return ids

# def sliding_window_jaccard(seq1, seq2, window_size, threshold):
#     """
#     在seq1上滑动窗口与seq2做Jaccard相似度匹配
#     """
#     len1 = len(seq1)
#     if len1 < window_size:
#         return 0.0
#
#     max_sim = 0.0
#     for i in range(len1 - window_size + 1):
#         window = set(seq1[i:i + window_size])
#         sim = jaccard_similarity(window, set(seq2))
#         if sim > max_sim:
#             max_sim = sim
#         if max_sim >= threshold:
#             break  # 提前退出
#     return max_sim
#
# def filter_similar_routes(virtual_routes, real_routes, threshold=0.8):
#     """使用滑动窗口+Jaccard过滤相似虚拟轨迹"""
#     filtered = []
#     real_sequences = [extract_node_ids_sequence(r) for r in real_routes]
#
#     for v_route in virtual_routes:
#         v_seq = extract_node_ids_sequence(v_route)
#         keep = True
#         for r_seq in real_sequences:
#             w_size = len(r_seq)
#             max_sim = sliding_window_jaccard(v_seq, r_seq, w_size, threshold)
#             if max_sim >= threshold:
#                 keep = False
#                 break
#         if keep:
#             filtered.append(v_route)
#
#     return filtered
def extract_all_subsequences(seq, window_size=2):
    return set(tuple(seq[i:i+window_size]) for i in range(len(seq) - window_size + 1))
def filter_exact_subsequence_routes(virtual_routes, real_routes, window_size=2):
    """
    过滤掉那些包含真实轨迹中连续子序列的虚拟路线（子序列必须完全匹配，且顺序一致）
    """
    # 提取所有真实路线中连续子序列集合
    real_subseq_set = set()
    for r in real_routes:
        r_seq = extract_node_ids_sequence(r)
        real_subseq_set.update(extract_all_subsequences(r_seq, window_size))

    # 过滤虚拟轨迹
    filtered_routes = []
    for v in virtual_routes:
        v_seq = extract_node_ids_sequence(v)
        v_subseqs = extract_all_subsequences(v_seq, window_size)
        if not v_subseqs & real_subseq_set:
            filtered_routes.append(v)

    return filtered_routes
def generate():
    random.seed(42)
    mdd_file = '../datasets/precleaning/MFW/mdd_step3_with_details_with_location.jsonl'
    poi_file = '../datasets/precleaning/MFW/scenic_step2.jsonl'
    # route_file = '../datasets/finetuning/mfw_real_train_val_routes.jsonl'
    #
    # routes = load_jsonl(route_file)
    mdd_data = load_jsonl(mdd_file)
    poi_data = load_jsonl(poi_file)
    # mdd_data_cleaned, poi_data_cleaned = [], []
    #
    # for route in routes:
    #     trajs = route["trajectory"]
    #     for traj in trajs:
    #         type = traj["type"]
    #         if type == "mdd":
    #             mddId = traj["mddId"]
    #             for mdd in mdd_data:
    #                 if mdd["mddId"] == mddId:
    #                     mdd_data_cleaned.append(mdd)
    #         elif type == "poi":
    #             poi_id = traj["poi_id"]
    #             for poi in poi_data:
    #                 if poi["poi_id"] in poi_id:
    #                     poi_data_cleaned.append(poi)
    poi_dict = build_poi_dict(poi_data)

    virtual_routes = generate_virtual_routes(
        mdd_data,
        poi_dict,
        num_routes=5000,
        max_mdd_per_route=3,
        max_poi_per_mdd=5,
        distance_threshold_km=300,  # 限制相邻城市在300km以内
        min_total_nodes=10,
        max_total_nodes=12
    )

    with open('../datasets/precleaning/MFW/virtual_routes.jsonl', 'w', encoding='utf-8') as f:
        for route in virtual_routes:
            f.write(json.dumps(route, ensure_ascii=False) + '\n')

    print(f"生成了 {len(virtual_routes)} 条虚拟路线，保存到 virtual_routes.jsonl")

    real_route_file = '../datasets/finetuning/mfw_real_test_routes.jsonl'  # 真实的轨迹测试集文件路径
    real_routes = load_real_routes(real_route_file)

    # 过滤与真实轨迹的测试集高度重合的虚拟路线
    virtual_routes_filtered = filter_exact_subsequence_routes(virtual_routes, real_routes, window_size=2)

    # 保存最终版本
    with open('../datasets/precleaning/MFW/virtual_routes_filtered.jsonl', 'w', encoding='utf-8') as f:
        for route in virtual_routes_filtered:
            f.write(json.dumps(route, ensure_ascii=False) + '\n')

    print(f"原始生成 {len(virtual_routes)} 条，去重后剩余 {len(virtual_routes_filtered)} 条虚拟路线")

# def generate_virtual_routes(mdd_data, poi_dict, num_routes=100, max_mdd_per_route=2, max_poi_per_mdd=5, distance_threshold_km=1500):
#     """
#     生成虚拟线路（限制地理距离，确保每条线路 mdd + poi 节点总数不小于10）
#     """
#     routes = []
#     all_mdds = [mdd for mdd in mdd_data if 'latitude' in mdd and 'longitude' in mdd and mdd.get('poi_list')]
#
#     attempts = 0
#     max_attempts = num_routes * 20
#     min_total_nodes = 10
#     min_mdd_count = 2
#
#     while len(routes) < num_routes and attempts < max_attempts:
#         attempts += 1
#         trajectory = []
#         visited_mdd_ids = set()
#         total_poi_count = 0
#         mdd_count = 0
#
#         current_mdd = random.choice(all_mdds)
#         visited_mdd_ids.add(current_mdd['mddId'])
#
#         for _ in range(random.randint(min_mdd_count, max_mdd_per_route)):
#             # 添加当前城市节点
#             trajectory.append({
#                 "type": "mdd",
#                 "mddId": current_mdd['mddId'],
#                 "mddTitle": current_mdd['mddTitle'],
#                 "details": current_mdd.get('details', "")
#             })
#             mdd_count += 1
#
#             # 添加当前城市的POI（不为空才添加）
#             mdd_poi_list = current_mdd.get('poi_list', [])
#             if mdd_poi_list:
#                 num_poi = random.randint(1, min(len(mdd_poi_list), max_poi_per_mdd))  # 最少1个poi
#                 selected_pois = random.sample(mdd_poi_list, num_poi)
#
#                 for poi_ref in selected_pois:
#                     poi_id = poi_ref['poi_id']
#                     if poi_id in poi_dict:
#                         poi = poi_dict[poi_id]
#                         trajectory.append({
#                             "type": "poi",
#                             "poi_id": poi['poi_id'],
#                             "poi_title": poi['poi_title'],
#                             "mddId": current_mdd['mddId'],
#                             "mddTitle": current_mdd['mddTitle'],
#                             "details": poi.get('details', "")
#                         })
#                         total_poi_count += 1
#                     else:
#                         print(f"Missing poi_id: {poi_id}")
#                         continue
#
#             # 选下一个城市（满足地理距离且未访问）
#             candidates = [
#                 m for m in all_mdds
#                 if m['mddId'] not in visited_mdd_ids and
#                    haversine(current_mdd['latitude'], current_mdd['longitude'], m['latitude'], m['longitude']) < distance_threshold_km
#             ]
#             if not candidates:
#                 break
#             current_mdd = random.choice(candidates)
#             visited_mdd_ids.add(current_mdd['mddId'])
#
#         # 确保轨迹总数 ≥ 10，包含多个 mdd 和 poi
#         if len(trajectory) >= min_total_nodes:
#             routes.append({
#                 "route_id": len(routes),
#                 "trajectory": trajectory
#             })
#
#     return routes
# def generate_virtual_routes(mdd_data, poi_dict, num_routes=100, max_mdd_per_route=5, max_poi_per_mdd=5, distance_threshold_km=300):
#     """
#     生成虚拟线路（限制地理距离）
#     """
#     routes = []
#     all_mdds = [mdd for mdd in mdd_data if 'latitude' in mdd and 'longitude' in mdd]
#
#     for route_idx in range(num_routes):
#         trajectory = []
#         visited_mdd_ids = set()
#
#         # 随机选第一个城市作为起点
#         current_mdd = random.choice(all_mdds)
#         visited_mdd_ids.add(current_mdd['mddId'])
#
#         for _ in range(random.randint(2, max_mdd_per_route)):
#             # 添加当前城市节点
#             trajectory.append({
#                 "type": "mdd",
#                 "mddId": current_mdd['mddId'],
#                 "mddTitle": current_mdd['mddTitle'],
#                 "details": current_mdd.get('details', ""),
#             })
#
#             # 添加当前城市的POI
#             mdd_poi_list = current_mdd.get('poi_list', [])
#             num_poi = random.randint(0, min(len(mdd_poi_list), max_poi_per_mdd))
#             selected_pois = random.sample(mdd_poi_list, num_poi) if num_poi > 0 else []
#
#             for poi_ref in selected_pois:
#                 poi_id = poi_ref['poi_id']
#                 if poi_id in poi_dict:
#                     poi = poi_dict[poi_id]
#                     trajectory.append({
#                         "type": "poi",
#                         "poi_id": poi['poi_id'],
#                         "poi_title": poi['poi_title'],
#                         "mddId": current_mdd['mddId'],
#                         "mddTitle": current_mdd['mddTitle'],
#                         "details": poi.get('details', "")
#                     })
#
#             # 选下一个城市（必须在阈值距离以内，且尽量不重复）
#             candidates = [
#                 m for m in all_mdds
#                 if m['mddId'] not in visited_mdd_ids and
#                    haversine(current_mdd['latitude'], current_mdd['longitude'], m['latitude'], m['longitude']) < distance_threshold_km
#             ]
#             if not candidates:
#                 break
#             current_mdd = random.choice(candidates)
#             visited_mdd_ids.add(current_mdd['mddId'])
#
#         routes.append({
#             "route_id": route_idx,
#             "trajectory": trajectory
#         })
#
#     return routes