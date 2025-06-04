'''
    将mfw的数据集进行BERT训练后得到一个句向量，并且将
    （7）最后得到处理完后的数据，数据的结构为：
        1）route_id: 为了将热门旅游线路和FourSquare统一处理，将FourSquare的user_id改为route_id
        2）trajectory为线路，每条线路中的地点分为两种类型，一种为mdd，一种为poi，然后我们做基于mdd和poi的双层推荐。
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
# import os
# import json
# import re
# import random
# from tqdm import tqdm
# from sentence_transformers import SentenceTransformer
#
# # =========================
# # 配置路径
# # =========================
# fsq_data_dir = './precleaning/FourSquare'
# mfw_data_dir = './precleaning/MFW'
# save_dir = './precleaning/Final'
# os.makedirs(save_dir, exist_ok=True)
#
# # 加载模型
# model_en = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# model_zh = SentenceTransformer('shibing624/text2vec-base-multilingual')
#
# # =========================
# # 工具函数
# # =========================
# def get_sentence_embedding(text, lang='en'):
#     if not text.strip():
#         return None
#     if lang == 'zh':
#         return model_zh.encode(text.strip())
#     else:
#         return model_en.encode(text.strip())
#
# def is_address(text):
#     address_keywords = ['路', '街', '大道', '巷', '区', '镇', '县', '村', '弄', '胡同']
#     if any(kw in text for kw in address_keywords) and len(text) < 30:
#         return True
#     if re.fullmatch(r'[\d\s\-]+', text):
#         return True
#     return False
#
# def deduplicate(details_list):
#     return list(set(details_list))
#
# def extract_keywords(text, lang='en', topk=5):
#     if not text:
#         return []
#     sentences = [s.strip() for s in re.split(r'[，。,.\n]', text) if s.strip()]
#     if len(sentences) > topk:
#         return random.sample(sentences, topk)
#     return sentences
#
# # =========================
# # 加载马蜂窝 POI 和 MDD
# # =========================
# def load_mafengwo_poi_mdd(poi_path, mdd_path):
#     poi_data = {}
#     mdd_data = {}
#
#     with open(poi_path, 'r', encoding='utf-8') as f:
#         for line in f:
#             item = json.loads(line)
#             poi_id = item.get('id')
#             name = item.get('name', '').strip()
#             details = item.get('details', '').strip()
#             if not name:
#                 continue
#             if not details or is_address(details):
#                 text = name
#             else:
#                 text = f"{name}，{details}"
#             poi_data[poi_id] = {
#                 'name': name,
#                 'details': extract_keywords(text, lang='zh')
#             }
#
#     with open(mdd_path, 'r', encoding='utf-8') as f:
#         for line in f:
#             item = json.loads(line)
#             mdd_id = item.get('id')
#             name = item.get('name', '').strip()
#             poi_list = item.get('scenic_list', [])
#             scenic_texts = []
#             for scenic in poi_list:
#                 pid = scenic.get('id')
#                 if pid and pid in poi_data:
#                     scenic_texts.extend(poi_data[pid]['details'])
#             mdd_data[mdd_id] = {
#                 'name': name,
#                 'details': deduplicate(scenic_texts[:5]) if scenic_texts else []
#             }
#
#     return poi_data, mdd_data
#
# # =========================
# # 加载 FourSquare POI
# # =========================
# def load_foursquare_poi(poi_path):
#     poi_data = {}
#
#     with open(poi_path, 'r', encoding='utf-8') as f:
#         for line in f:
#             item = json.loads(line)
#             poi_id = item.get('id')
#             name = item.get('name', '').strip()
#             details = item.get('details', '').strip()
#             if not name:
#                 continue
#             text = f"{name}. {details}" if details else name
#             poi_data[poi_id] = {
#                 'name': name,
#                 'details': extract_keywords(text, lang='en')
#             }
#
#     return poi_data
#
# # =========================
# # 处理马蜂窝路线
# # =========================
# def process_mfw_routes(route_path, poi_data, mdd_data, save_path):
#     with open(route_path, 'r', encoding='utf-8') as f_in, open(save_path, 'w', encoding='utf-8') as f_out:
#         for line in f_in:
#             route = json.loads(line)
#             route_id = route.get('id')
#             trajectory = []
#
#             days = route.get('days', [])
#             for day in days:
#                 day_items = day.get('items', [])
#                 for item in day_items:
#                     if item['type'] == 'mdd':
#                         mdd_id = item.get('mddId')
#                         mdd_info = mdd_data.get(mdd_id, {})
#                         trajectory.append({
#                             "type": "mdd",
#                             "mddId": mdd_id,
#                             "mddTitle": mdd_info.get('name', ''),
#                             "details": mdd_info.get('details', [])
#                         })
#                     elif item['type'] == 'poi':
#                         poi_id = item.get('poi_id')
#                         mdd_id = item.get('mddId')
#                         poi_info = poi_data.get(poi_id, {})
#                         mdd_info = mdd_data.get(mdd_id, {})
#                         trajectory.append({
#                             "type": "poi",
#                             "poi_id": poi_id,
#                             "poi_title": poi_info.get('name', ''),
#                             "mddId": mdd_id,
#                             "mddTitle": mdd_info.get('name', ''),
#                             "details": poi_info.get('details', [])
#                         })
#
#             if trajectory:
#                 out_item = {
#                     "route_id": route_id,
#                     "trajectory": trajectory
#                 }
#                 f_out.write(json.dumps(out_item, ensure_ascii=False) + '\n')
#
# # =========================
# # 处理 FourSquare 路线
# # =========================
# # 处理 FourSquare 数据
# def process_foursquare_data(foursquare_dir="./precleaning/FourSquare", batch_num=27):
#     all_mdd_info = dict()  # 保存 mddId -> {mddTitle, latitude, longitude, details(list)}
#     all_poi_info = dict()  # 保存 poi_id -> {poi_title, latitude, longitude, details(list), mddId}
#     all_trajectories = []  # 保存每条轨迹
#
#     for i in tqdm(range(batch_num), desc="Processing FourSquare batches"):
#         batch_path = os.path.join(foursquare_dir, f"trajectories_batch{i}.jsonl")
#         if not os.path.exists(batch_path):
#             print(f"Warning: {batch_path} not found, skip")
#             continue
#
#         with open(batch_path, 'r', encoding='utf-8') as f:
#             for line in f:
#                 line = line.strip()
#                 if not line:
#                     continue
#                 data = json.loads(line)
#                 route_id = data.get("route_id")
#                 trajectory = data.get("trajectory", [])
#
#                 traj = []  # 当前轨迹
#
#                 for item in trajectory:
#                     if item['type'] == 'mdd':
#                         mdd_id = item['mddId']
#                         if mdd_id not in all_mdd_info:
#                             all_mdd_info[mdd_id] = {
#                                 'mddTitle': item.get('mddTitle', ''),
#                                 'latitude': item.get('latitude', None),
#                                 'longitude': item.get('longitude', None),
#                                 'details': item.get('details', [])
#                             }
#                         traj.append({'type': 'mdd', 'id': mdd_id})
#                     elif item['type'] == 'poi':
#                         poi_id = item['poi_id']
#                         if poi_id not in all_poi_info:
#                             all_poi_info[poi_id] = {
#                                 'poi_title': item.get('poi_title', ''),
#                                 'latitude': item.get('latitude', None),
#                                 'longitude': item.get('longitude', None),
#                                 'details': item.get('details', []),
#                                 'mddId': item.get('mddId', None)  # 所属MDD
#                             }
#                         traj.append({'type': 'poi', 'id': poi_id})
#                 if traj:
#                     all_trajectories.append({'route_id': route_id, 'trajectory': traj})
#
#     return all_mdd_info, all_poi_info, all_trajectories
#
# def split_and_save(data_list, save_dir, prefix):
#     random.shuffle(data_list)
#     n = len(data_list)
#     train_end = int(0.8 * n)
#     val_end = int(0.9 * n)
#
#     train_set = data_list[:train_end]
#     val_set = data_list[train_end:val_end]
#     test_set = data_list[val_end:]
#
#     def save_jsonl(dataset, path):
#         with open(path, 'w', encoding='utf-8') as f:
#             for item in tqdm(dataset, desc=f"Saving {os.path.basename(path)}"):
#                 f.write(json.dumps(item, ensure_ascii=False) + '\n')
#
#     save_jsonl(train_set, os.path.join(save_dir, f'{prefix}_train.jsonl'))
#     save_jsonl(val_set, os.path.join(save_dir, f'{prefix}_val.jsonl'))
#     save_jsonl(test_set, os.path.join(save_dir, f'{prefix}_test.jsonl'))
#
# def main():
#     ## Mafengwo
#     mfw_poi_path = os.path.join(mfw_data_dir, 'scenic_step2.jsonl')
#     mfw_mdd_path = os.path.join(mfw_data_dir, 'mdd_step1.jsonl')
#     mfw_route_path = os.path.join(mfw_data_dir, 'route_step3.jsonl')
#     mfw_save_path = os.path.join(save_dir, 'mfw_final_routes.jsonl')
#
#     poi_data_mfw, mdd_data_mfw = load_mafengwo_poi_mdd(mfw_poi_path, mfw_mdd_path)
#     process_mfw_routes(mfw_route_path, poi_data_mfw, mdd_data_mfw, mfw_save_path)
#
#     ## FourSquare
#     mdd_info, poi_info, trajectories = process_foursquare_data()
#
#     print(f"总共提取了 {len(mdd_info)} 个MDD，{len(poi_info)} 个POI，{len(trajectories)} 条轨迹")
#
#     # 读回 Mafengwo Final routes
#     mfw_routes = []
#     with open(mfw_save_path, 'r', encoding='utf-8') as f:
#         for line in tqdm(f, desc="Loading MFW final routes"):
#             mfw_routes.append(json.loads(line.strip()))
#
#     # 保存拆分好的 Mafengwo
#     split_and_save(mfw_routes, save_dir, prefix='mfw')
#
#     # 保存拆分好的 FourSquare
#     split_and_save(trajectories, save_dir, prefix='fsq')
#
#     print("✅ Mafengwo 和 FourSquare 路线处理 & 数据集划分完成！")

# # ==============================
# # 配置
# # ==============================
# fsq_data_dir = './precleaning/FourSquare'
# mfw_data_dir = './precleaning/MFW'
# save_dir = './precleaning/Final'
# os.makedirs(save_dir, exist_ok=True)
#
# # 加载模型
# model_en = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# model_zh = SentenceTransformer('shibing624/text2vec-base-multilingual')
#
# # ==============================
# # 工具函数
# # ==============================
# def get_sentence_embedding(text, lang='en'):
#     if not text.strip():
#         return None
#     if lang == 'zh':
#         return model_zh.encode(text.strip())
#     else:
#         return model_en.encode(text.strip())
#
# def is_address(text):
#     address_keywords = ['路', '街', '大道', '巷', '区', '镇', '县', '村', '弄', '胡同']
#     if any(kw in text for kw in address_keywords) and len(text) < 30:
#         return True
#     if re.fullmatch(r'[\d\s\-]+', text):
#         return True
#     return False
#
# # 利用集合去重
# def deduplicate(details_list):
#     return list(set(details_list))
#
# def split_dataset(texts, embeddings, ratio=0.8):
#     idxs = list(range(len(texts)))
#     random.shuffle(idxs)
#     split = int(len(idxs) * ratio)
#     train_idx, test_idx = idxs[:split], idxs[split:]
#     train_texts = [texts[i] for i in train_idx]
#     train_embeddings = embeddings[train_idx]
#     test_texts = [texts[i] for i in test_idx]
#     test_embeddings = embeddings[test_idx]
#     return (train_texts, train_embeddings), (test_texts, test_embeddings)
#
# # ==============================
# # FourSquare 处理（英文）
# # ==============================
# def load_foursquare_trajectories(data_dir, batch_range=range(27)):
#     all_poi_details = []
#     mdd_to_poi = {}
#
#     for i in batch_range:
#         batch_path = os.path.join(data_dir, f'trajectories_batch{i}.jsonl')
#         if not os.path.exists(batch_path):
#             print(f"Warning: {batch_path} not found.")
#             continue
#
#         with open(batch_path, 'r', encoding='utf-8') as f:
#             for line in f:
#                 traj = json.loads(line)
#                 trajectory = traj.get('trajectory', [])
#
#                 # Process each item in the trajectory
#                 poi_details = []
#                 for item in trajectory:
#                     if 'details' in item and item['details']:
#                         detail = item['details'][0].strip()  # 假设details字段是列表，只取第一个
#                         if detail:
#                             all_poi_details.append(detail)
#                             if item['type'] == 'mdd':
#                                 city = item.get('mddTitle', '').strip()
#                                 if city:
#                                     if city not in mdd_to_poi:
#                                         mdd_to_poi[city] = []
#                                     mdd_to_poi[city].append(detail)
#                             elif item['type'] == 'poi':
#                                 poi_details.append(detail)
#
#     return all_poi_details, mdd_to_poi
#
# def process_mdd_details(mdd_to_poi, topk=5):
#     mdd_sentences = {}
#     for mdd, poi_details in mdd_to_poi.items():
#         freq = {}
#         for detail in poi_details:
#             freq[detail] = freq.get(detail, 0) + 1
#         sorted_details = sorted(freq.items(), key=lambda x: x[1], reverse=True)
#         selected = [d[0] for d in sorted_details[:topk]]
#         merged_text = '，'.join(selected)
#         mdd_sentences[mdd] = merged_text
#     return mdd_sentences
#
# # ==============================
# # 马蜂窝 处理（中文）
# # ==============================
# def load_mafengwo_poi_mdd(poi_path, mdd_path):
#     poi_data = {}
#     mdd_to_poi = {}
#
#     with open(poi_path, 'r', encoding='utf-8') as f:
#         for line in f:
#             item = json.loads(line)
#             poi_id = item.get('id')
#             name = item.get('name', '').strip()
#             details = item.get('details', '').strip()
#             if not name:
#                 continue
#             if not details or is_address(details):
#                 text = name
#             else:
#                 text = f"{name}，{details}"
#             poi_data[poi_id] = text
#
#     with open(mdd_path, 'r', encoding='utf-8') as f:
#         for line in f:
#             item = json.loads(line)
#             mdd_name = item.get('name', '').strip()
#             scenic_list = item.get('scenic_list', [])
#             scenic_ids = [scenic.get('id') for scenic in scenic_list if 'id' in scenic]
#             if mdd_name:
#                 mdd_to_poi[mdd_name] = [poi_data.get(pid) for pid in scenic_ids if pid in poi_data]
#
#     return poi_data, mdd_to_poi
#
# def process_mfw_mdd_details(mdd_to_poi, topk=5):
#     mdd_sentences = {}
#     for mdd, poi_texts in mdd_to_poi.items():
#         freq = {}
#         for text in poi_texts:
#             if text:
#                 freq[text] = freq.get(text, 0) + 1
#         sorted_texts = sorted(freq.items(), key=lambda x: x[1], reverse=True)
#         selected = [d[0] for d in sorted_texts[:topk]]
#         merged_text = '，'.join(selected)
#         mdd_sentences[mdd] = merged_text
#     return mdd_sentences
#
# # ==============================
# # 主流程
# # ==============================
# # 1. FourSquare
# fsq_poi_details, fsq_mdd_to_poi = load_foursquare_trajectories(fsq_data_dir)
# fsq_poi_details = deduplicate(fsq_poi_details)
# fsq_mdd_sentences = process_mdd_details(fsq_mdd_to_poi)
#
# fsq_poi_embeddings = [get_sentence_embedding(text, lang='en') for text in tqdm(fsq_poi_details, desc='Encoding FSQ POIs')]
# fsq_mdd_embeddings = [get_sentence_embedding(text, lang='en') for text in tqdm(fsq_mdd_sentences.values(), desc='Encoding FSQ MDDs')]
#
# # 2. 马蜂窝
# mfw_poi_path = os.path.join(mfw_data_dir, 'scenic_step2.jsonl')
# mfw_mdd_path = os.path.join(mfw_data_dir, 'mdd_step1.jsonl')
#
# mfw_poi_data, mfw_mdd_to_poi = load_mafengwo_poi_mdd(mfw_poi_path, mfw_mdd_path)
# mfw_poi_texts = list(mfw_poi_data.values())
# mfw_mdd_sentences = process_mfw_mdd_details(mfw_mdd_to_poi)
#
# mfw_poi_embeddings = [get_sentence_embedding(text, lang='zh') for text in tqdm(mfw_poi_texts, desc='Encoding MFW POIs')]
# mfw_mdd_embeddings = [get_sentence_embedding(text, lang='zh') for text in tqdm(mfw_mdd_sentences.values(), desc='Encoding MFW MDDs')]
#
# # 3. 合并
# all_poi_texts = fsq_poi_details + mfw_poi_texts
# all_poi_embeddings = np.vstack(fsq_poi_embeddings + mfw_poi_embeddings)
#
# all_mdd_texts = list(fsq_mdd_sentences.keys()) + list(mfw_mdd_sentences.keys())
# all_mdd_embeddings = np.vstack(fsq_mdd_embeddings + mfw_mdd_embeddings)
#
# # 4. 划分训练/测试集
# (poi_train_texts, poi_train_embeddings), (poi_test_texts, poi_test_embeddings) = split_dataset(all_poi_texts, all_poi_embeddings)
# (mdd_train_texts, mdd_train_embeddings), (mdd_test_texts, mdd_test_embeddings) = split_dataset(all_mdd_texts, all_mdd_embeddings)
#
# # 5. 保存
# np.savez_compressed(os.path.join(save_dir, 'poi_train.npz'), texts=poi_train_texts, embeddings=poi_train_embeddings)
# np.savez_compressed(os.path.join(save_dir, 'poi_test.npz'), texts=poi_test_texts, embeddings=poi_test_embeddings)
# np.savez_compressed(os.path.join(save_dir, 'mdd_train.npz'), texts=mdd_train_texts, embeddings=mdd_train_embeddings)
# np.savez_compressed(os.path.join(save_dir, 'mdd_test.npz'), texts=mdd_test_texts, embeddings=mdd_test_embeddings)
#
# print("\nAll processing done! ✅ 训练/测试集已保存.")



