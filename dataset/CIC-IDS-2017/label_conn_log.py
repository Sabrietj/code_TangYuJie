'''Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv 文件后面有一个空行有非法字符，请手动删除'''

import os
import csv
import json
from datetime import datetime

folder_path = '/root/autodl-fs/CIC-IDS-2017/'
w = 0
l = 0
zidian = {}
# 获取文件夹下所有条目


for root, dirs, files in os.walk(folder_path):
    if root == folder_path:
        continue

    if files:
        for filename in files:
            if filename.endswith(".csv"):
                # print(filename)
                csvpath = f"{root}/{filename}"
                # 读取csv文件
                # 第一列和最后一列的值
                with open(csvpath, "rt", encoding="utf-8") as f:
                    csv_reader = csv.reader(f)
                    header = next(csv_reader)  # 跳过表头
                    print(csvpath)

                    for row in csv_reader:
                        try:
                            if len(row) == 0 or row[0] == "" or row[0].split("-")[-1] == "0":
                                continue
                            # print(row)
                            w += 1
                            if len(row[6]) == 19:
                                row[6] = row[6][:-3]
                                row[6] = row[6].lstrip("0").replace("/0", "/").replace(" 0", " ")
                                # print(row[6])

                            key = f"{row[0]}-{row[6]}"

                            label = row[-1]
                            if label == "Web Attack � Brute Force":
                                label = "Web Attack – Brute Force"
                            elif label == "Web Attack � XSS":
                                label = "Web Attack – XSS"
                            elif label == "Web Attack � Sql Injection":
                                label = "Web Attack – Sql Injection"
                            if key in zidian:
                                zidian[key].append(label)
                            else:
                                zidian[key] = [label]
                                # print(zidian[key])
                        except BaseException:
                            print(row)

from collections import Counter

# 示例列表
datas = []

# 统计频次


num = 0
'''"id.orig_h":"192.168.10.3","id.orig_p":445,"id.resp_h":"192.168.10.14","id.resp_p":57702,"proto":"17"'''
newzidian = {}
for key in zidian:
    # if len(set(zidian[key]))==1:
    #     print(key,zidian[key])
    #     break
    if len(set(zidian[key])) > 1:  #
        # print(key,zidian[key])
        l += len(zidian[key])
        datas += zidian[key]
        num += 1
print(w, num, l)  #
print(Counter(datas))
# exit(2)
# 哪怕就是删了影响也不大，就是占了6万条  csv里有6万条 同一六元组但是混装多个标签   方案1、如果混装，取比例最高的   方案2、放弃这6万条混装的
# 先执行方案2 方案2匹配到的太少了   决定添加方案1
'''
label
BENIGN           219478
PortScan         158827
DDoS              65243
SSH-Patator        2926
Infiltration         21
DoS slowloris         2
Heartbleed            1

label
BENIGN           219478
PortScan         158951
DDoS              86456
SSH-Patator        2979
Infiltration         21
DoS slowloris         3
Heartbleed            1

BENIGN                        466682
DoS Hulk                      162954
PortScan                      159108
DDoS                           86466
DoS Slowhttptest               16045
DoS GoldenEye                   7607
FTP-Patator                     3986
DoS slowloris                   3877
SSH-Patator                     2979
Bot                             2208
Web Attack – Brute Force        1364
Web Attack – XSS                 629
Infiltration                      21
Web Attack – Sql Injection        12
Heartbleed                         1
'''
ports = set()
for key in zidian:
    ports.add(key.split("-")[-2])
    if len(set(zidian[key])) == 1:
        newzidian[key] = zidian[key][0]
    else:
        # 反正不选BENIGN
        # 哪个攻击比例高，选哪个
        while "BENIGN" in zidian[key]:
            zidian[key].remove("BENIGN")
        counter = Counter(zidian[key])
        most_common_attack, count = counter.most_common(1)[0]  # 返回频数最高的 (元素, 次数)
        print(f"最高频攻击类型: {most_common_attack}, 出现次数: {count}")
        newzidian[key] = most_common_attack
        # 统计频数最高的攻击类型

print(len(newzidian))
print(ports)


# exit(2)


def getFormatted_time(a):
    results = []
    # 转换为本地 datetime 对象
    for zhi in [0, 60, -60, 120, -120, 180, -180]:
        dt = datetime.fromtimestamp(float(a) + zhi - 23 * 3600)

        # 格式化为 "日/月/年 小时:分钟"：5/7/2017 4:48
        formatted_time = dt.strftime("%d/%m/%Y %H:%M")

        # 去除前导零（例如 05 → 5，07 → 7）
        formatted_time = formatted_time.lstrip("0").replace("/0", "/").replace(" 0", " ")
        results.append(formatted_time)
    for zhi in [0, 60, -60, 120, -120, 180, -180]:
        dt = datetime.fromtimestamp(float(a) + zhi - 11 * 3600)

        # 格式化为 "日/月/年 小时:分钟"：5/7/2017 4:48
        formatted_time = dt.strftime("%d/%m/%Y %H:%M")

        # 去除前导零（例如 05 → 5，07 → 7）
        formatted_time = formatted_time.lstrip("0").replace("/0", "/").replace(" 0", " ")
        results.append(formatted_time)
    # print(a,b,formatted_time)  # 输出：5/7/2017 4:48
    return results


# conn取出一条到字典里面找

current_dir = '/root/autodl-fs/CIC-IDS-2017/'
items = os.listdir(current_dir)

file_names = [item for item in items if os.path.isdir(os.path.join(current_dir, item))]

for i in range(len(file_names)):
    print(' 📁 处理设备:' + file_names[i])

    conn_file = os.path.join(current_dir, file_names[i], "conn.log")
    output_file = os.path.join(current_dir, file_names[i], "conn_label.log")

    if not os.path.exists(conn_file):
        print(f"❌ 文件不存在: {conn_file}")
        continue

    # 打开输入文件进行读取
    # 流式逐行读取，不加载整个文件
    truenum = 0
    mapnums = 0
    with open(conn_file, 'r', encoding="utf-8") as f_in, \
            open(output_file, 'w', encoding="utf-8") as f_out:

        for line in f_in:  # ⬅️ 逐行读取，不会全加载到内存！
            line = line.strip()
            if not line:  # 跳过空行
                continue
            truenum += 1
            try:
                # 解析每一行的JSON对象
                data = json.loads(line.strip())
                # 构建新的JSON对象
                src_ip = data.get('id.orig_h', '').strip()
                src_port = str(data.get('id.orig_p', '')).strip()
                dst_ip = data.get('id.resp_h', '').strip()
                dst_port = str(data.get('id.resp_p', '')).strip()
                proto = str(data.get('proto', '')).strip().lower()
                if proto == "tcp":
                    proto = "6"
                elif proto == "udp":
                    proto = "17"
                else:
                    # print("报错啦",proto)
                    continue

                for shijian in getFormatted_time(data.get("ts")):
                    # 构建key

                    # 构造匹配键
                    key = "-".join([src_ip, dst_ip, src_port, dst_port, proto, shijian])
                    # print(key)
                    if key in newzidian:
                        # print("有")
                        # print(key)
                        mapnums += 1
                        # 将新的JSON对象写入输出文件
                        data["label"] = newzidian[key]
                        f_out.write(json.dumps(data) + '\n')
                        break
                else:
                    # 没有配对上的流
                    # print("No match for conn entry (CIC-IDS-2017):" + str(key))
                    pass

            except json.JSONDecodeError:
                print(f"无法解析的行: {line}")

    print(f"[Summary] Matched: {mapnums}/{truenum} ({mapnums / truenum:.2%})")