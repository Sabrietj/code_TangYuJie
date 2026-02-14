# -*- coding: utf-8 -*-

import time
import os
import sys
import pandas as pd
from print_manager import __PrintManager__
# 添加../utils目录到Python搜索路径
utils_path = os.path.join(os.path.dirname(__file__), '..', 'utils')
sys.path.insert(0, utils_path)
from config_manager import read_session_tuple_mode
from evaluate_data import EvaluateData
from session_tuple import SessionTuple, FlowTuple
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import traceback
from datetime import datetime

verbose = False

def is_empty_result(result):
    if result is None:
        return True
    if isinstance(result, pd.DataFrame):
        return result.empty
    if isinstance(result, dict):
        return len(result) == 0
    if isinstance(result, list):
        return len(result) == 0
    return False

class LogAnalyzer(EvaluateData):
    def __init__(self):
        super(LogAnalyzer, self).__init__()

        # log file path
        self.conn_log_path = None
        self.dns_log_path = None
        self.ssl_log_path = None
        self.x509_log_path = None
        self.flowmeter_log_path = None
        self.http_log_path = None
        self.ftp_log_path = None
        self.mqtt_log_path = {'connect': None, 'subscribe': None, 'publish': None}

        # log file data
        self.conn_dict = dict()
        self.dns_frame = pd.DataFrame()
        self.ssl_dict = dict()
        self.x509_dict = dict()
        self.flowmeter_dict = dict()
        self.http_dict = dict()
        self.ftp_dict = dict()
        self.mqtt_dict = {'connect': dict(), 'subscribe': dict(), 'publish': dict()}

        # connection tuple (could be 1-tuple with client IP, or
        # 2 tuple with client and server IPs) after merging logs
        self.session_tuple = dict()
        self.open_time = None

        self.debug_uid = "COIC5J39wSkYdzNLah"


    def load_files_parallel(self):
        """并行加载所有日志文件"""
        with ThreadPoolExecutor(max_workers=5) as executor:
            # 提交所有文件加载任务
            print("\n>>> load %s" % self.conn_log_path)
            conn_future = executor.submit(self.read_log, self.conn_log_path)
            print("\n>>> load %s" % self.ssl_log_path)
            ssl_future = executor.submit(self.read_log, self.ssl_log_path)
            print("\n>>> load %s" % self.x509_log_path)
            x509_future = executor.submit(self.read_log, self.x509_log_path, False, True) # date = False, to_dict=True, allow_1toN=False
            print("\n>>> load %s" % self.dns_log_path)
            dns_future = executor.submit(self.read_log, self.dns_log_path, False, False) # date = False, to_dict=False, allow_1toN=False
            print("\n>>> load %s" % self.flowmeter_log_path)
            flowmeter_future = executor.submit(self.read_log, self.flowmeter_log_path)

            # 当前这段read_log代码在“一个 uid 对应多条 http.log 记录”时：
            # 最终字典里保留的是——“最后一次读到的那一条 http 记录”。
            print("\n>>> load %s" % self.http_log_path)
            http_future = executor.submit(self.read_log, self.http_log_path, False, True, True) # date = False, to_dict=True, allow_1toN=True

            print("\n>>> load %s" % self.ftp_log_path)
            ftp_future = executor.submit(self.read_log, self.ftp_log_path, False, True, True) # date = False, to_dict=True, allow_1toN=True

            mqtt_futures = dict()
            for log_type, mqtt_path in self.mqtt_log_path.items():
                print("\n>>> load %s" % mqtt_path)
                mqtt_future = executor.submit(self.read_log, mqtt_path, False, True, True) # date = False, to_dict=True, allow_1toN=True
                mqtt_futures[log_type] = mqtt_future

            # 获取结果
            self.conn_dict = conn_future.result()
            self.ssl_dict = ssl_future.result()
            self.x509_dict = x509_future.result()
            self.dns_frame = dns_future.result()
            self.flowmeter_dict = flowmeter_future.result()
            self.http_dict = http_future.result()
            self.ftp_dict = ftp_future.result()
            self.mqtt_dict = {log_type: future.result() for log_type, future in mqtt_futures.items()}


    def load_conn_file(self):
        self.conn_dict = self.read_log(self.conn_log_path)
        print("\n>>> load %s" % self.conn_log_path)

    def load_ssl_file(self):
        self.ssl_dict = self.read_log(self.ssl_log_path)
        print("\n>>> load %s" % self.ssl_log_path)
        # 详细调试信息
        if verbose and self.ssl_dict:
            sample_uid = list(self.ssl_dict.keys())[0]
            sample_data = self.ssl_dict[sample_uid]
            print(f">>> SSL样本数据 - UID: {sample_uid}")
            print(f">>> SSL样本内容: {list(sample_data.keys())}")

    def load_x509_file(self):
        self.x509_dict = self.read_log(self.x509_log_path, date=False, to_dict=True)
        print("\n>>> load %s" % self.x509_log_path)

    def load_dns_file(self):
        result = self.read_log(self.dns_log_path, date=False, to_dict=False)
        if is_empty_result(result):
            self.dns_frame = pd.DataFrame()
            print(f"\n>>> DNS文件不存在或为空: {self.dns_log_path}")
            return

        df = result

        # === 1. 时间校正 ===
        if 'ts' in df.columns:
            df['ts'] = df['ts'].apply(self.time_correction)
        else:
            print(f">>> 警告: DNS文件 {self.dns_log_path} 没有 'ts' 列")

        # === 2. 只过滤“无 answers 的垃圾行”，不丢 CNAME ===
        df = df[
            df['qtype_name'].isin(['A', 'AAAA', 'CNAME']) &
            df['answers'].notna() &
            (df['answers'] != '-') &
            (df['answers'] != '')
        ].copy()

        # ts 转 float，避免后面反复 astype
        df['ts'] = df['ts'].astype(float)

        self.dns_frame = df

        # === 3. 构建 DNS 索引（IP + CNAME） ===
        self._build_dns_answer_index()
        # 现实 Zeek DNS 中很常见的是：
        # domain.com → CNAME → cdn.xxx.net → A → server IP
        self._build_dns_cname_index()

        print(f"\n>>> load {self.dns_log_path}, after filter: {len(df)} records")

    def _build_dns_answer_index(self):
        """
        构建：
        self.dns_answer_index[ip] = [row1, row2, ...]  # 按 ts 升序
        """
        self.dns_answer_index = defaultdict(list)

        for _, row in self.dns_frame.iterrows():
            answers = row.get('answers')
            if answers is None:
                continue

            # 情况 1：JSON Zeek，answers 是 list
            if isinstance(answers, list):
                ans_list = answers

            # 情况 2：TSV / 老 Zeek，answers 是字符串
            elif isinstance(answers, str):
                if answers in ('-', ''):
                    continue
                ans_list = [a.strip() for a in answers.split(',') if a and a != '-']

            else:
                continue

            for ans in ans_list:
                # 只索引 IP（与 slow 逻辑一致）
                # 这里不做复杂校验，简单判断即可
                if isinstance(ans, str) and (
                    ans.count('.') == 3 or ':' in ans   # IPv4 / IPv6
                ):
                    self.dns_answer_index[ans].append(row)

        # 按时间排序（fast 查询依赖这个）
        for ip in self.dns_answer_index:
            self.dns_answer_index[ip].sort(key=lambda r: r['ts'])

        print(f"\n>>> DNS answer index built: {len(self.dns_answer_index)} IPs")

    def _build_dns_cname_index(self):
        """
        构建反向 CNAME 索引：
        self.dns_cname_reverse_index[target_domain] = [row1, row2, ...]
        其中 row.query 是 “parent/alias”，target_domain 是 “canonical/target”
        """
        import ipaddress
        from collections import defaultdict

        def norm_name(x: str) -> str:
            # Zeek 里域名可能带结尾 '.'
            return x.strip().strip('.').lower()

        def is_ip(x: str) -> bool:
            try:
                ipaddress.ip_address(x.strip())
                return True
            except Exception:
                return False

        self.dns_cname_reverse_index = defaultdict(list)
        self.dns_cname_forward_index = defaultdict(list)

        for _, row in self.dns_frame.iterrows():
            query = row.get('query')
            answers = row.get('answers')
            if not query or not answers:
                continue

            # answers 可能是 list 或 str
            if isinstance(answers, list):
                ans_list = answers
            elif isinstance(answers, str):
                if answers in ('-', ''):
                    continue
                ans_list = [a.strip() for a in answers.split(',') if a and a != '-']
            else:
                continue

            qn = norm_name(query)

            # 关键：从 answers 里提取“域名型答案”（非 IP）作为 CNAME target
            for ans in ans_list:
                if not isinstance(ans, str):
                    continue
                an = ans.strip()
                if not an or an in ('-', ''):
                    continue
                if is_ip(an):
                    continue  # IP 走 dns_answer_index，不是 CNAME target

                tn = norm_name(an)
                if not tn or tn == qn:
                    continue

                # 用 target 做 key（反向索引），row 里保留原始 query 供回溯
                self.dns_cname_reverse_index[tn].append(row)
                self.dns_cname_forward_index[qn].append(tn)

        print(f"\n>>> DNS CNAME reverse index built: {len(self.dns_cname_reverse_index)} targets")


    def _resolve_cname_chain(self, dns_row, max_depth=5):
        def norm_name(x: str) -> str:
            return x.strip().strip('.').lower()

        chain = []
        q = dns_row.get('query')
        if not q:
            return chain

        cur = norm_name(q)
        chain.append(cur)
        visited = {cur}

        for _ in range(max_depth):
            next_names = []
            for target in self.dns_cname_forward_index.get(cur, []):
                if target not in visited:
                    visited.add(target)
                    chain.append(target)
                    next_names.append(target)

            if not next_names:
                break

            # 一般 DNS 是一条链，取第一个即可
            cur = next_names[0]

        return chain


    def load_flowmeter_file(self):
        self.flowmeter_dict = self.read_log(self.flowmeter_log_path)
        print("\n>>> load %s" % self.flowmeter_log_path)

    def load_http_file(self):
        # 当前这段read_log代码在“一个 uid 对应多条 http.log 记录”时：
        # 最终字典里保留的是——“最后一次读到的那一条 http 记录”。
        self.http_dict = self.read_log(self.http_log_path, date=False, to_dict=True, allow_1toN=True)
        # TODO: preprocess
        print("\n>>> load %s" % self.http_log_path)

    def load_ftp_file(self):
        self.ftp_dict = self.read_log(self.ftp_log_path, date=False, to_dict=True, allow_1toN=True)
        # TODO: preprocess
        print("\n>>> load %s" % self.ftp_log_path)

    def load_mqtt_file(self):
        mqtt_dict = dict()
        for log_type, mqtt_path in self.mqtt_log_path.items():
            mqtt_dict[log_type] = self.read_log(mqtt_path, date=False, to_dict=True, allow_1toN=True)
            # TODO: preprocess
            print("\n>>> load %s" % mqtt_path)
        self.mqtt_dict = mqtt_dict

    def load_files_sequential(self):
        """顺序加载所有日志文件"""
        self.load_conn_file()
        self.load_dns_file()
        self.load_ssl_file()
        self.load_x509_file()
        self.load_flowmeter_file()
        self.load_http_file()
        self.load_ftp_file()
        self.load_mqtt_file()

    def evaluate_features(self, path_to_dataset):
        self.path_to_dataset = path_to_dataset
        self.conn_log_path = os.path.join(self.path_to_dataset, 'conn_label.log')
        self.ssl_log_path = os.path.join(self.path_to_dataset, 'ssl.log')
        self.x509_log_path = os.path.join(self.path_to_dataset, 'x509.log')
        self.dns_log_path = os.path.join(self.path_to_dataset, 'dns.log')
        self.flowmeter_log_path = os.path.join(self.path_to_dataset, 'flowmeter.log')
        self.http_log_path = os.path.join(self.path_to_dataset, 'http.log')
        self.ftp_log_path = os.path.join(self.path_to_dataset, 'ftp.log')
        self.mqtt_log_path = { log_type:os.path.join(self.path_to_dataset, f"mqtt_{log_type}.log") for log_type in self.mqtt_log_path.keys()}

        # 添加调试信息
        debug_lines = [
            ">>> 调试信息 - 文件路径检查:",
            f"    conn_log_path      : {self.conn_log_path} - 存在吗？{os.path.exists(self.conn_log_path)}",
            f"    ssl_log_path       : {self.ssl_log_path} - 存在吗？{os.path.exists(self.ssl_log_path)}",
            f"    x509_log_path      : {self.x509_log_path} - 存在吗？{os.path.exists(self.x509_log_path)}",
            f"    dns_log_path       : {self.dns_log_path} - 存在吗？{os.path.exists(self.dns_log_path)}",
            f"    flowmeter_log_path : {self.flowmeter_log_path} - 存在吗？{os.path.exists(self.flowmeter_log_path)}",
            f"    http_log_path      : {self.http_log_path} - 存在吗？{os.path.exists(self.http_log_path)}",
            f"    ftp_log_path       : {self.ftp_log_path} - 存在吗？{os.path.exists(self.ftp_log_path)}",
        ]

        for mqtt_name, mqtt_path in self.mqtt_log_path.items():
            debug_lines.append(
                f"    mqtt_log_path[{mqtt_name}] : {mqtt_path} - 存在吗？{os.path.exists(mqtt_path)}"
            )

        print("\n" + "\n".join(debug_lines))

        # 重置所有数据容器
        self.conn_dict = dict()
        self.dns_frame = pd.DataFrame()
        self.ssl_dict = dict()
        self.x509_dict = dict()
        self.flowmeter_dict = dict()
        self.http_dict = dict()
        self.ftp_dict = dict()
        self.mqtt_dict = {'connect': dict(), 'subscribe': dict(), 'publish': dict()}
        self.session_tuple = dict()

        try:
            self.load_files_sequential()

            # 检查必要文件是否存在
            if not self.conn_dict:
                print(f">>> 错误: 连接日志文件为空或无法读取: {self.conn_log_path}")
                return False

            # 添加SSL字典内容检查
            if verbose:
                print(f">>> Conn字典包含 {len(self.conn_dict)} 条记录")
                print(f">>> SSL字典包含 {len(self.ssl_dict)} 条记录")

                if self.conn_dict:
                    # 打印前几个UID看看
                    print(">>> Conn字典中的前5个UID:")
                    for i, uid in enumerate(list(self.conn_dict.keys())[:5]):
                        print(f"    {i+1}. {uid}")

                if self.ssl_dict:
                    print(">>> SSL字典中的前5个UID:")
                    for i, uid in enumerate(list(self.ssl_dict.keys())[:5]):
                        print(f"    {i+1}. {uid}")
                else:
                    print(">>> SSL字典为空")

            self.create_session_tuple_sequential()
            return True

        except Exception as e:
            print(f">>> 处理目录时发生错误 {path_to_dataset}: {e}")
            traceback.print_exc()
            return False


    def build_session_tuple_index(self, conn_log):
        mode = read_session_tuple_mode()
        src_ip = conn_log.get('id.orig_h')
        src_port = conn_log.get('id.orig_p')
        dst_ip = conn_log.get('id.resp_h')
        dst_port = conn_log.get('id.resp_p')
        proto = conn_log.get('proto')

        if mode == 'srcIP':
            return (src_ip,)
        elif mode == 'dstIP':
            return (dst_ip,)
        elif mode == 'srcIP_dstIP':
            return (src_ip, dst_ip)
        elif mode == 'srcIP_dstIP_proto':
            return (src_ip, dst_ip, proto)
        elif mode == 'srcIP_dstIP_dstPort':
            return (src_ip, dst_ip, dst_port)
        elif mode == 'srcIP_dstIP_dstPort_proto':
            return (src_ip, dst_ip, dst_port, proto)
        elif mode == 'srcIP_srcPort_dstIP_dstPort_proto':
            return (src_ip, src_port, dst_ip, dst_port, proto)
        else:
            # 未知模式，使用默认四元组
            print(f"Warning: Unknown session_tuple_mode '{mode}', using default: srcIP_dstIP_dstPort_proto")
            return (src_ip, dst_ip, dst_port, proto)


    def create_session_tuple_sequential(self):
        """顺序创建session tuple，带进度显示"""
        number_of_x509_log = 0
        number_of_ssl_log = 0
        number_of_not_ssl_log = 0
        number_of_background_flow = 0

        print(f"\n>>> 开始顺序处理会话元组，path_to_dataset = {self.path_to_dataset}")

        # 获取连接日志总数用于进度显示
        total_conn = len(self.conn_dict)
        processed_conn = 0

        # 特定UID调试
        debug_uid = self.debug_uid
        debug_uid_found = False

        def update_progress():
            """更新进度显示"""
            nonlocal processed_conn
            processed_conn += 1
            progress = (processed_conn / total_conn) * 100
            bar_length = 40
            filled_length = int(bar_length * processed_conn // total_conn)
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            print(f'\r>>> 进度，path_to_dataset = {self.path_to_dataset}: |{bar}| {progress:.1f}% ({processed_conn}/{total_conn} 条连接)', end='', flush=True)

        print(f"\n>>> 开始处理 {total_conn} 条连接日志...")

        for conn_uid, conn_log in self.conn_dict.items():
            # 1. 强力过滤无效 UID
            if not conn_uid:  # 过滤 None 和 空字符串
                continue
            if str(conn_uid).strip() == "" or str(conn_uid).lower() in ['-', 'none', 'nan']:
                continue
            # 特定UID调试
            if conn_uid == debug_uid and not debug_uid_found:
                debug_uid_found = True
                print(f"\n>>> 🔍 找到目标UID: {debug_uid}")
                print(f">>>   Conn日志内容: {conn_log}")
                print(f">>>   SSL字典中是否存在: {conn_uid in self.ssl_dict}")

                if conn_uid in self.ssl_dict:
                    ssl_log = self.ssl_dict[conn_uid]
                    print(f">>>   SSL日志内容: {ssl_log}")
                    print(f">>>   SSL版本: {ssl_log.get('version', 'N/A')}")
                    print(f">>>   服务名: {ssl_log.get('server_name', 'N/A')}")

            label = conn_log.get('label', '')
            if 'Background' in label:
                number_of_background_flow += 1

            flowmeter_log = None
            if self.flowmeter_dict:
                # 直接获取，找不到就为 None，不要 else，不要 print 警告
                flowmeter_log = self.flowmeter_dict.get(conn_uid)

            http_log = None
            if self.http_dict:
                http_log = self.http_dict.get(conn_uid)

            ftp_log = None
            if self.ftp_dict:
                ftp_log = self.ftp_dict.get(conn_uid)

            mqtt_log = None
            if self.mqtt_dict:
                mqtt_log = { log_type:mqtt_dict.get(conn_uid) for log_type, mqtt_dict in self.mqtt_dict.items()}

            tuple_index = self.build_session_tuple_index(conn_log)
            # 特定UID调试：会话索引
            if conn_uid == debug_uid:
                print(f">>>   会话索引: {tuple_index}")
                print(f">>>   会话字典中是否存在: {tuple_index in self.session_tuple}")

            if tuple_index not in self.session_tuple:
                self.session_tuple[tuple_index] = SessionTuple(tuple_index)
                if conn_uid == debug_uid:
                    print(f">>>   创建新会话: {tuple_index}")

            # add not-ssl flow to session_tuple
            if conn_uid not in self.ssl_dict:
                # 特定UID调试：非SSL流路径
                if conn_uid == debug_uid:
                    print(f">>>   ⚠️  UID不在SSL字典中，走非SSL路径")

                # 非SSL流
                # http.log 存在于明文流量，非SSL流中
                flow = FlowTuple(conn_uid, conn_log, flowmeter_log=flowmeter_log, http_log=http_log, ftp_log=ftp_log, mqtt_log=mqtt_log)
                self.session_tuple[tuple_index].flow_list.append(flow)
                self.session_tuple[tuple_index].flow_list.sort(key=lambda f: f.start_time)
                self.session_tuple[tuple_index].add_not_ssl_flow(conn_log)
                number_of_not_ssl_log += 1

                # 特定UID调试：非SSL流创建结果
                if conn_uid == debug_uid:
                    print(f">>>   非SSL流创建完成")
                    print(f">>>   会话中SSL流数: {self.session_tuple[tuple_index].get_number_of_ssl_flows()}")
                    print(f">>>   会话中非SSL流数: {self.session_tuple[tuple_index].number_of_not_ssl_flows}")
            else:
                # SSL flow
                ssl_log = self.ssl_dict[conn_uid]

                # 特定UID调试：SSL流路径
                if conn_uid == debug_uid:
                    print(f">>>   ✅ UID在SSL字典中，走SSL路径")
                    print(f">>>   SSL日志版本: {ssl_log.get('version', 'N/A')}")

                if not ssl_log.get("version"):
                    #TCP链接建立了，但是Originator abort了这个连接，后面没有TLS握手 这种情况TLS握手字段都不能用
                    # 即使SSL无效，仍然创建非SSL流
                    if verbose:
                        print(f">>> 跳过无效SSL日志: UID={conn_uid}, 无版本信息")
                    # 即使SSL无效，仍然创建非SSL流
                    flow = FlowTuple(conn_uid, conn_log, flowmeter_log=flowmeter_log, http_log=http_log, ftp_log=ftp_log, mqtt_log=mqtt_log)
                    self.session_tuple[tuple_index].flow_list.append(flow)
                    self.session_tuple[tuple_index].add_not_ssl_flow(conn_log)
                    number_of_not_ssl_log += 1

                    if conn_uid == debug_uid:
                        print(f">>>   无效SSL流创建为非SSL流")
                else:
                    # 有效的SSL日志
                    ssl_log['ts'] = self.time_correction(ssl_log['ts'])

                    # 特定UID调试：SSL处理前
                    if conn_uid == debug_uid:
                        print(f">>>   开始处理有效SSL日志")
                        print(f">>>   调用add_ssl_flow前")

                    # 添加到session tuple
                    self.session_tuple[tuple_index].add_ssl_flow(conn_log)
                    if conn_uid == debug_uid:
                        self.session_tuple[tuple_index].add_ssl_log(ssl_log, debug_uid=debug_uid)
                    else:
                        self.session_tuple[tuple_index].add_ssl_log(ssl_log)
                    number_of_ssl_log += 1

                    # 特定UID调试：SSL处理后
                    if conn_uid == debug_uid:
                        print(f">>>   SSL流添加完成")
                        print(f">>>   会话中SSL流数: {self.session_tuple[tuple_index].get_number_of_ssl_flows()}")

                    # 处理X509证书
                    cert_chain_fuids = ssl_log.get('cert_chain_fps',[])
                    x509_logs = []

                    # 特定UID调试：X509证书
                    if conn_uid == debug_uid:
                        print(f">>>   证书链指纹: {cert_chain_fuids}")
                        print(f">>>   X509字典键数: {len(self.x509_dict)}")

                    for x509_uid in cert_chain_fuids:
                        if x509_uid in self.x509_dict:
                            x509_log = self.x509_dict[x509_uid]
                            x509_log['ts'] = self.time_correction(x509_log['ts'])
                            x509_logs.append(x509_log)
                            # 添加到 session_tuple
                            self.session_tuple[tuple_index].add_x509_log(x509_log)
                            # SNI 检查可以只对第一个证书做，也可以对每个证书做
                            self.session_tuple[tuple_index].is_SNI_in_cert(ssl_log, x509_log)
                            number_of_x509_log += 1

                            # 特定UID调试：X509添加
                            if conn_uid == debug_uid:
                                print(f">>>   添加X509证书: {x509_uid}")
                        else:
                            if conn_uid == debug_uid:
                                print(f">>>   ⚠️ X509证书不存在: {x509_uid}")

                    # 创建FlowTuple
                    # http.log 可能存在，zeek对ssl解密有限支持
                    flow = FlowTuple(conn_uid, conn_log, ssl_log=ssl_log, x509_logs=x509_logs, \
                                     flowmeter_log=flowmeter_log, http_log=http_log, ftp_log=ftp_log, mqtt_log=mqtt_log)
                    self.session_tuple[tuple_index].flow_list.append(flow)
                    self.session_tuple[tuple_index].flow_list.sort(key=lambda f: f.start_time)

                    # 特定UID调试：FlowTuple创建
                    if conn_uid == debug_uid:
                        print(f">>>   FlowTuple创建完成")
                        print(f">>>   会话中流数量: {len(self.session_tuple[tuple_index].flow_list)}")
                        # 检查FlowTuple中的SSL信息
                        for i, flow_item in enumerate(self.session_tuple[tuple_index].flow_list):
                            if flow_item.uid == debug_uid:
                                print(f">>>   第{i}个FlowTuple SSL日志: {flow_item.ssl_log is not None}")
                                if flow_item.ssl_log:
                                    print(f">>>     SSL版本: {flow_item.ssl_log.get('version', 'N/A')}")

            # 更新进度条
            update_progress()

        # 特定UID调试：最终检查
        if debug_uid_found:
            print(f"\n>>> 🔍 目标UID处理完成检查")
            for tuple_idx, session in self.session_tuple.items():
                for flow in session.flow_list:
                    if flow.uid == debug_uid:
                        print(f">>>   在会话 {tuple_idx} 中找到目标Flow")
                        print(f">>>     Flow SSL日志存在: {flow.ssl_log is not None}")
                        if flow.ssl_log:
                            print(f">>>     SSL版本: {flow.ssl_log.get('version', 'N/A')}")
                        break

        # 进度条完成
        print(f"\n>>> 连接日志处理完成，开始处理DNS日志，path_to_dataset = {self.path_to_dataset}...")

        # add dns log to conn tuple，也加上进度显示
        # number_of_dns_log_slow = self.add_dns_log_with_progress_slow()
        number_of_dns_log_fast = self.add_dns_log_with_progress_fast()

        # print("number_of_dns_log_slow =", number_of_dns_log_slow, "number_of_dns_log_fast =", number_of_dns_log_fast)
        # assert number_of_dns_log_slow == number_of_dns_log_fast, "DNS日志匹配结果不一致！"

        # number_of_dns_log = number_of_dns_log_slow
        number_of_dns_log = number_of_dns_log_fast

        print(f"\n>>> 所有处理完成，开始统计信息，path_to_dataset = {self.path_to_dataset}...")

        self.statistic_of_session_tuple(number_of_ssl_log, number_of_x509_log, number_of_dns_log, number_of_not_ssl_log)

        print(f"\n>>> 顺序处理会话元组完成，path_to_dataset = {self.path_to_dataset}")
        return

    # 保留原有的add_dns_log函数作为兼容
    def add_dns_log(self):
        """兼容原有接口，调用带进度显示的版本"""
        # return self.add_dns_log_with_progress_slow()
        return self.add_dns_log_with_progress_fast()

    def add_dns_log_with_progress_fast(self):
        number_of_dns_log = 0

        if not hasattr(self, 'dns_answer_index'):
            print("\n>>> DNS answer index not found")
            return 0

        total_conn = len(self.conn_dict)
        processed_conn = 0
        last_print_ratio = 0.0

        print(f"\n>>> 开始DNS日志匹配（fast），共 {total_conn} 条连接...")

        for conn_uid, conn_log in self.conn_dict.items():
            processed_conn += 1
            ratio = processed_conn / total_conn
            percent = ratio * 100

            # 每 0.5% 打印一次，或最后一次强制打印
            if ratio - last_print_ratio >= 0.005 or processed_conn == total_conn:
                last_print_ratio = ratio

                bar_length = 40
                filled = int(bar_length * ratio)
                bar = '█' * filled + '-' * (bar_length - filled)

                print(
                    f'\r>>> 进度，path_to_dataset = {self.path_to_dataset}: '
                    f'|{bar}| {percent:.1f}% ({processed_conn}/{total_conn} 条连接)',
                    end='', flush=True
                )

            tuple_index = self.build_session_tuple_index(conn_log)
            if tuple_index not in self.session_tuple:
                continue

            server_ip = conn_log.get('id.resp_h')
            flow_time = float(conn_log.get('ts', 0))

            dns_list = self.dns_answer_index.get(server_ip)
            if not dns_list:
                continue

            # 选最近且 ts <= flow_time 的 DNS
            best_dns_log = None
            for row in reversed(dns_list):
                if row['ts'] <= flow_time:
                    best_dns_log = row
                    break

            if best_dns_log is None:
                continue

            # TTL check（完全复用 slow 逻辑）
            if not self._check_dns_ttl(best_dns_log, server_ip, flow_time):
                continue

            if best_dns_log is not None:
                # CNAME chain resolution
                cname_chain = self._resolve_cname_chain(best_dns_log)

                best_dns_log = (
                    best_dns_log.to_dict()
                    if isinstance(best_dns_log, pd.Series)
                    else dict(best_dns_log)
                )

                best_dns_log['cname_chain'] = cname_chain if cname_chain else []

                # attach
                self.session_tuple[tuple_index].add_dns_log(conn_log, best_dns_log)
                for flow in self.session_tuple[tuple_index].flow_list:
                    if flow.uid == conn_uid:
                        flow.dns_log = (
                            best_dns_log.to_dict()
                            if isinstance(best_dns_log, pd.Series)
                            else best_dns_log
                        )
                        break

                number_of_dns_log += 1

        print()  # 先从进度条，执行换行
        print(f"\n>>> DNS日志匹配完成（fast），共匹配 {number_of_dns_log} 条")
        return number_of_dns_log

    def _check_dns_ttl(self, dns_log, server_ip, flow_time):
        """
        返回 True 表示 TTL 有效
        返回 False 表示 TTL 过期 / 不匹配
        """
        dns_ts_col = 'ts'
        dns_ttl_col = 'TTLs'

        if dns_ttl_col not in dns_log or dns_log[dns_ttl_col] in ('', '-', None):
            return True  # 没有 TTL，当作有效（与 slow 行为一致）

        try:
            answers = dns_log.get('answers')
            ttls = dns_log.get('TTLs')

            # answers
            if isinstance(answers, list):
                ans_list = answers
            elif isinstance(answers, str):
                ans_list = [a.strip() for a in answers.split(',') if a and a != '-']
            else:
                return True

            # ttls
            if isinstance(ttls, list):
                ttl_list = ttls
            elif isinstance(ttls, str):
                ttl_list = [
                    float(x) for x in ttls.split(',')
                    if x.replace('.', '', 1).isdigit()
                ]
            else:
                return True

            if server_ip not in ans_list:
                return True

            pos = ans_list.index(server_ip)
            if pos >= len(ttl_list):
                return True

            ttl_val = ttl_list[pos]
            dns_time = float(dns_log[dns_ts_col])

            TTL_GRACE = 5.0  # 秒
            return (dns_time + ttl_val + TTL_GRACE) >= flow_time

        except Exception:
            # 保守策略：异常时不丢
            traceback.print_exc()
            return True


    def add_dns_log_with_progress_slow(self):
        """添加DNS日志，带进度显示"""
        number_of_dns_log = 0

        # 检查DNS数据是否有效
        if self.dns_frame is None or self.dns_frame.empty:
            print(f">>> add_dns_log_with_progress(): DNS数据为空或不存在")
            return number_of_dns_log

        # 检查必要的列是否存在
        required_columns = ['ts', 'answers', 'qtype_name']
        missing_columns = [col for col in required_columns if col not in self.dns_frame.columns]

        if missing_columns:
            print(f"\n>>> DNS数据缺少必要列: {missing_columns}，跳过DNS匹配")
            return number_of_dns_log

        # 常用字段名（可根据实际日志修改）
        dns_answer_col = "answers"
        dns_query_col = "query"
        dns_ts_col = "ts"
        dns_ttl_col = "TTLs"

        # 获取需要处理的连接总数
        total_conn = len(self.conn_dict)
        processed_conn = 0

        def update_dns_progress():
            """更新DNS处理进度"""
            nonlocal processed_conn
            processed_conn += 1
            progress = (processed_conn / total_conn) * 100
            bar_length = 30
            filled_length = int(bar_length * processed_conn // total_conn)
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            print(f'\r>>> DNS进度，path_to_dataset = {self.path_to_dataset}: |{bar}| {progress:.1f}% ({processed_conn}/{total_conn} 条连接)', end='', flush=True)

        def answer_contains_ip(ans, ip):
            if ans is None:
                return False

            # 情况 1：answers 是 list（JSON dns.log，最常见）
            if isinstance(ans, list):
                return ip in ans

            # 情况 2：answers 是字符串（TSV / 老 Zeek）
            if isinstance(ans, str):
                return ip in [a.strip() for a in ans.split(',') if a and a != '-']

            return False

        print(f"\n>>> 开始DNS日志匹配，共 {total_conn} 条连接需要处理...")

        for conn_uid, conn_log in self.conn_dict.items():
            label = conn_log.get('label', '')
            tuple_index = self.build_session_tuple_index(conn_log)
            if tuple_index in self.session_tuple:
                server_ip = conn_log['id.resp_h']
                try:
                    flow_time = float(conn_log.get('ts', 0))
                except Exception:
                    traceback.print_exc()
                    flow_time = 0.0

                # DNS 文件必须有 answers 列才能继续
                if dns_answer_col not in self.dns_frame.columns:
                    continue

                # ① 找到所有包含该 server_ip 的 DNS 记录（仅保留 A/AAAA 类型）；
                candidates = self.dns_frame[
                    self.dns_frame['qtype_name'].isin(['A', 'AAAA']) &
                    self.dns_frame[dns_answer_col].apply(lambda x: answer_contains_ip(x, server_ip))
                    # self.dns_frame[dns_answer_col].astype(str).str.contains(str(server_ip), na=False)
                ]
                if dns_ts_col not in candidates.columns:
                    continue

                # ② 保留查询时间 <= flow_time 的记录
                candidates = candidates[candidates[dns_ts_col] <= flow_time]

                if candidates.empty:
                    continue

                # ③ 选择距离 flow_time 最近的一条
                candidates = candidates.copy()
                candidates["time_diff"] = flow_time - candidates[dns_ts_col]
                best_dns_log = candidates.loc[candidates["time_diff"].idxmin()].to_dict()

                # ④ TTL 检查（如果存在）
                if dns_ttl_col in best_dns_log and best_dns_log[dns_ttl_col] not in ["-", ""]:
                    try:
                        answers = best_dns_log.get('answers')
                        ttls = best_dns_log.get('TTLs')

                        if isinstance(answers, list):
                            ans_list = answers
                        elif isinstance(answers, str):
                            ans_list = [a.strip() for a in answers.split(',') if a and a != '-']
                        else:
                            ans_list = []

                        if isinstance(ttls, list):
                            ttl_list = ttls
                        elif isinstance(ttls, str):
                            ttl_list = [float(x) for x in ttls.split(',') if x.replace('.', '', 1).isdigit()]
                        else:
                            ttl_list = []

                        TTL_GRACE = 5.0  # 秒
                        if server_ip in ans_list:
                            pos = ans_list.index(server_ip)
                            ttl_val = ttl_list[pos] if pos < len(ttl_list) else None
                            if ttl_val is not None:
                                dns_time = float(best_dns_log[dns_ts_col])
                                if (dns_time + ttl_val + TTL_GRACE) < flow_time:
                                    continue  # TTL 已过期，跳过
                    except Exception:
                        traceback.print_exc()
                        pass

                best_dns_log = (
                    best_dns_log.to_dict()
                    if isinstance(best_dns_log, pd.Series)
                    else dict(best_dns_log)
                )

                # CNAME chain resolution
                cname_chain = self._resolve_cname_chain(best_dns_log)
                best_dns_log['cname_chain'] = cname_chain if cname_chain else []

                # ⑤ 添加到 session_tuple，及其维护的 flow 列表
                if verbose:
                    print(f"[DNS] Best match found for {conn_uid}: {best_dns_log}")
                self.session_tuple[tuple_index].add_dns_log(conn_log, best_dns_log)
                number_of_dns_log += 1

                if verbose:
                    print(f"[DEBUG] Looking for flow with UID: {conn_uid}")

                for flow in self.session_tuple[tuple_index].flow_list:
                    if flow.uid == conn_uid:
                        if verbose:
                            print(f"[DEBUG] Found the flow with UID: {conn_uid} in session tuple {tuple_index}")
                        flow.dns_log = best_dns_log
                        break

            # 更新DNS处理进度
            update_dns_progress()

        print(f"\n>>> DNS日志匹配完成，共匹配 {number_of_dns_log} 条DNS记录")
        return number_of_dns_log

    def detect_log_format(self, filename):
        """改进的日志格式检测"""
        if not os.path.exists(filename):
            print(f">>> 警告: 文件不存在: {filename}")
            return "text"

        try:
            with open(filename, 'r', encoding='utf-8', errors='ignore') as file:
                for line in file:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    # 检查是否是JSON格式
                    if line.startswith('{') and line.endswith('}'):
                        try:
                            json.loads(line)
                            print(f"\n>>> 检测到JSON格式: {filename}")
                            return "json"
                        except:
                            pass
                    break
            print(f"\n>>> 检测到文本格式: {filename}")
            return "text"
        except Exception as e:
            print(f">>> 文件格式检测错误 {filename}: {e}")
            traceback.print_exc()
            return "text"

    def read_log(self, filename, date=False, to_dict=True, allow_1toN=False):
        """
        返回约定：
        - conn / ssl / flowmeter / dns / x509:
            uid -> dict
        - http / ftp / mqtt:
            allow_1toN=False: uid -> dict
            allow_1toN=True : uid -> list[dict]
        """
        # 检查文件是否存在
        if not os.path.exists(filename):
            print(f"\nWarning: File {filename} does not exist.")
            # 根据 to_dict 参数返回适当的空值
            if to_dict:
                return {}
            else:
                return pd.DataFrame()

        log_format = self.detect_log_format(filename)

        if log_format == "json":
            with open(filename, 'r') as file:
                log_lines = file.readlines()
                # 将每一行日志转换为字典
            log_dicts = [json.loads(line) for line in log_lines]

            if 'x509' in filename:
                data = {}
                # ===== x509.log 特例 =====
                # x509.log 中 fingerprint 本身是“证书级唯一标识”
                # 一个 fingerprint 对应一张证书
                # 这里做 fingerprint -> record 的一对一映射是【语义正确】的
                for k in log_dicts:
                    fp = k.get("fingerprint")
                    if not fp:
                        continue

                    k = dict(k)
                    del k["fingerprint"]

                    if date and "ts" in k:
                        k["ts_datetime"] = datetime.utcfromtimestamp(k["ts"])

                    data[fp] = k
            else:
                # ===== 非 x509.log（如 http.log / ftp.log / mqtt.log）=====
                # ⚠️ 可能存在严重语义问题从这里开始 ⚠️
                #
                # Zeek 中：
                #   - uid 是“连接级（connection-level）标识”
                #   - 一个 uid 下，可能有多条应用层事件记录
                #     * http.log: 多个 request / response
                #     * ftp.log : 多个 FTP 命令
                #     * mqtt.log: 多个 publish / subscribe
                #
                # 但下面这段代码：
                #   data[uid] = k
                # 强制把 uid -> 单条 log 记录
                #
                # 后果：
                #   - 同一个 uid 出现多次时，前面的记录会被后面的覆盖
                #   - 最终 data[uid] 只保留“最后一次出现的那条 log”
                #
                # 即：
                #   uid -> 最后一条应用层事件（HTTP / FTP / MQTT）
                #
                # 这会系统性丢失：
                #   - HTTP keep-alive 中的早期请求
                #   - FTP 会话中的多条命令
                #   - MQTT 长连接中的消息序列
                #
                if allow_1toN:
                    data = defaultdict(list)
                    for k in log_dicts:
                        uid = k.get("uid")
                        if not uid or str(uid).strip() == "" or str(uid).lower() in ['-', 'none']:
                            continue
                        if not uid:
                            continue

                        k = dict(k)          # 防止原地修改
                        del k["uid"]

                        if date and "ts" in k:
                            k["ts_datetime"] = datetime.utcfromtimestamp(k["ts"])

                        data[uid].append(k)

                    # 强烈建议：按 ts 排序，保证语义稳定
                    for uid in data:
                        data[uid].sort(key=lambda x: x.get("ts", 0))

                else:
                    data = {}
                    for k in log_dicts:
                        uid = k.get("uid")
                        if not uid or str(uid).strip() == "" or str(uid).lower() in ['-', 'none']:
                            continue
                        if not uid:
                            continue

                        k = dict(k)
                        del k["uid"]

                        if date and "ts" in k:
                            k["ts_datetime"] = datetime.utcfromtimestamp(k["ts"])

                        data[uid] = k   # last-write-wins


            if to_dict:
                return data
            else:
                # 统一成 DataFrame 返回
                df = pd.DataFrame.from_dict(data, orient="index")
                return df
        else:
            fields = None
            with open(filename) as f:
                for line in f:
                    if date and '#open' in line:
                        self.open_time = line.strip().split('\t')[1]
                    if '#fields' in line or '#field' in line:
                        fields = line.strip().split('\t')[1:]
                        break

            if fields is None:
                print(f"Warning: {filename} has no field definitions, skip.")
                return pd.DataFrame() if not to_dict else {}

            data = pd.read_csv(
                filename,
                sep='\t',
                comment='#',
                engine='python'
            )

            if 'x509' in filename:
                index_col = 'fingerprint' if 'fingerprint' in data.columns else None
                if index_col:
                    data = data.drop_duplicates(subset=index_col).set_index(index_col)
            else:
                for col in ['uid', 'id']:
                    if col in data.columns:
                        data = data.drop_duplicates(subset=col).set_index(col)
                        break

            if to_dict:
                return data.to_dict('index')
            else:
                return data

    def time_correction(self, current_time):
        """校正时间戳，支持多种时间格式"""
        if not self.open_time:
            try:
                return float(current_time)
            except (ValueError, TypeError):
                return 0.0

        try:
            current_time = float(current_time)

            # 解析数据开始时间
            open_time = time.mktime(time.strptime(self.open_time, "%Y-%m-%d-%H-%M-%S"))

            # 判断时间戳类型
            # 假设2010年之前的时间戳都是相对时间（1262304000 = 2010-01-01）
            if current_time < 1262304000:  # 2010年之前
                return open_time + current_time
            else:
                return current_time

        except Exception as e:
            print(f"时间校正失败: {e}")
            traceback.print_exc()
            return current_time

    def statistic_of_session_tuple(self, number_of_ssl_log, number_of_x509_log,
                                number_of_dns_log, not_ssl_flow):
        malicious_tuples = 0
        normal_tuples = 0
        malicious_flows = 0
        normal_flows = 0

        for key in self.session_tuple:
            if self.session_tuple[key].is_malicious():
                malicious_tuples += 1
                malicious_flows += self.session_tuple[key].number_of_flows()
            else:
                normal_tuples += 1
                normal_flows += self.session_tuple[key].number_of_flows()

        print(
            f"\n>>> statistic result of session_tuple:\n"
            f"\tssl flow : {number_of_ssl_log}, not ssl flow : {not_ssl_flow}\n"
            f"\tmalicious flow : {malicious_flows}, normal flow : {normal_flows}\n"
            f"\tmalicious tuple : {malicious_tuples}, normal tuple : {normal_tuples}\n"
            f"\tadd x509 log : {number_of_x509_log}, add dns log : {number_of_dns_log}"
        )
