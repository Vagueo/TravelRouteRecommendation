Len = 0
with open("./virtual_routes_filtered.jsonl",'r',encoding="utf-8")as f:
    for line in f:
        Len += 1

print(Len)