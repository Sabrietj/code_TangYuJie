# -*- coding: utf-8 -*-

from print_manager import __PrintManager__
import os
import csv
import traceback
import sys

# 保持原有的路径注入方式
utils_path = os.path.join(os.path.dirname(__file__), '..', 'utils')
sys.path.insert(0, utils_path)
from zeek_columns import conn_columns, flowmeter_columns, ssl_columns, x509_columns, dns_columns, http_columns, \
    max_x509_cert_chain_len, \
    ftp_columns, mqtt_connect_columns, mqtt_subscribe_columns, mqtt_publish_columns


def reduce_events_aligned(events, columns):
    """辅助函数保持不变"""
    if not events:
        return {k: "" for k in columns}
    if len(events) == 1:
        ev = events[0]
        return {k: ev.get(k, "") for k in columns}
    result = {}
    for k in columns:
        result[k] = [ev.get(k, None) for ev in events]
    return result


class EvaluateData(object):
    def __init__(self):
        self.session_tuple = dict()
        self.cert_dict = dict()

    def create_plot_data(self, path, filename):
        print(f"\nCreating plot data for {filename} in {path}")
        __PrintManager__.evaluate_creating_plot()
        self.create_session_csv(path, filename + "-session" + ".csv")
        self.create_flow_list_csv(self.session_tuple, path, filename + "-flow" + ".csv")
        __PrintManager__.succ_evaluate_data()

    def create_flow_list_csv(self, session_tuple_dict, output_path, filename="flow_list.csv"):
        """创建流列表CSV文件，按时间升序排列"""
        print(f"\n>>> 开始创建流列表CSV: {filename} (Time Ascending)")

        target_uid = "COIC5J39wSkYdzNLah"

        os.makedirs(output_path, exist_ok=True)
        output_file = os.path.join(output_path, filename)

        headers = ["uid", "label", "is_malicious"] + \
                  [f"conn.{col}" for col in conn_columns] + \
                  [f"flowmeter.{col}" for col in flowmeter_columns] + \
                  [f"dns.{col}" for col in dns_columns] + \
                  [f"ssl.{col}" for col in ssl_columns] + \
                  [f"x509.cert{i}.{col}" for i in range(max_x509_cert_chain_len) for col in x509_columns] + \
                  [f"http.{col}" for col in http_columns] + \
                  [f"ftp.{col}" for col in ftp_columns] + \
                  [f"mqtt_connect.{col}" for col in mqtt_connect_columns] + \
                  [f"mqtt_subscribe.{col}" for col in mqtt_subscribe_columns] + \
                  [f"mqtt_publish.{col}" for col in mqtt_publish_columns]

        def _safe_dict(obj, *, name=None, uid=None):
            if obj is None: return {}
            if isinstance(obj, dict): return obj
            return {}

        # ==========================================
        # 1. 收集所有流并按时间排序
        # ==========================================
        all_flows = []
        for session in session_tuple_dict.values():
            if session.flow_list:
                all_flows.extend(session.flow_list)

        # 按 start_time 升序排列
        all_flows.sort(key=lambda f: f.start_time)

        with open(output_file, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()

            total_flows = 0
            ssl_flows = 0

            # 遍历排序后的 all_flows
            for flow in all_flows:
                if not flow.uid or str(flow.uid).strip() == "" or str(flow.uid).lower() in ['-', 'none', 'nan']:
                    continue
                total_flows += 1
                if flow.ssl_log and isinstance(flow.ssl_log, dict) and len(flow.ssl_log) > 0:
                    ssl_flows += 1

                conn_data = _safe_dict(flow.conn_log, name="conn_log", uid=flow.uid)
                flowmeter_data = _safe_dict(flow.flowmeter_log, name="flowmeter_log", uid=flow.uid)
                ssl_data = _safe_dict(flow.ssl_log, name="ssl_log", uid=flow.uid)
                dns_data = _safe_dict(flow.dns_log, name="dns_log", uid=flow.uid)

                if isinstance(flow.http_log, list):
                    http_data = reduce_events_aligned(flow.http_log, http_columns)
                else:
                    http_data = _safe_dict(flow.http_log, name="http_log", uid=flow.uid)

                if isinstance(flow.ftp_log, list):
                    ftp_data = reduce_events_aligned(flow.ftp_log, ftp_columns)
                else:
                    ftp_data = _safe_dict(flow.ftp_log, name="ftp_log", uid=flow.uid)

                # MQTT
                mqtt_log = flow.mqtt_log or {}
                if not isinstance(mqtt_log, dict): mqtt_log = {}

                def handle_mqtt_sublog(raw, columns):
                    if isinstance(raw, list): return reduce_events_aligned(raw, columns)
                    if isinstance(raw, dict): return {k: raw.get(k, "") for k in columns}
                    return {}

                mqtt_connect = handle_mqtt_sublog(mqtt_log.get("connect"), mqtt_connect_columns)
                mqtt_subscribe = handle_mqtt_sublog(mqtt_log.get("subscribe"), mqtt_subscribe_columns)
                mqtt_publish = handle_mqtt_sublog(mqtt_log.get("publish"), mqtt_publish_columns)

                if flow.uid == target_uid:
                    print(f">>> 🔍 Found Target UID: {target_uid}")

                x509_logs = flow.x509_logs if isinstance(flow.x509_logs, list) else []
                x509_flat = {}
                for i in range(max_x509_cert_chain_len):
                    cert_data = x509_logs[i] if i < len(x509_logs) else {}
                    x509_flat.update({f"x509.cert{i}.{k}": cert_data.get(k, "") for k in x509_columns})

                flow_record = {
                    "uid": flow.uid,
                    "label": flow.get_label(),
                    "is_malicious": flow.is_malicious(),
                    **{f"conn.{k}": conn_data.get(k, "") for k in conn_columns},
                    **{f"flowmeter.{k}": flowmeter_data.get(k, "") for k in flowmeter_columns},
                    **{f"dns.{k}": dns_data.get(k, "") for k in dns_columns},
                    **{f"ssl.{k}": ssl_data.get(k, "") for k in ssl_columns},
                    **x509_flat,
                    **{f"http.{k}": http_data.get(k, "") for k in http_columns},
                    **{f"ftp.{k}": ftp_data.get(k, "") for k in ftp_columns},
                    **{f"mqtt_connect.{k}": mqtt_connect.get(k, "") for k in mqtt_connect_columns},
                    **{f"mqtt_subscribe.{k}": mqtt_subscribe.get(k, "") for k in mqtt_subscribe_columns},
                    **{f"mqtt_publish.{k}": mqtt_publish.get(k, "") for k in mqtt_publish_columns}
                }

                try:
                    writer.writerow(flow_record)
                except Exception:
                    print(f"\n>>> ❌ Write CSV Error at uid={flow.uid}")
                    traceback.print_exc()

            print(f"\n>>> Flow CSV Done: Total={total_flows}, SSL={ssl_flows}")
            print(f"<<< dataset file {filename} successfully created !")

    def create_session_csv(self, path, filename="session_list.csv"):
        print(f"\nCreating session dataset for {filename} (Time Ascending)")

        header = [
            'session_index', 'is_malicious', 'ssl_version', 'cipher_suite_server',
            'cert_key_alg', 'cert_sig_alg', 'cert_key_type', 'max_duration',
            'avg_duration', 'percent_of_std_duration', 'number_of_flows',
            'ssl_flow_ratio', 'avg_size', 'recv_sent_size_ratio', 'avg_pkts',
            'recv_sent_pkts_ratio', 'packet_loss', 'percent_of_established_state',
            'avg_time_diff', 'std_time_diff', 'max_time_diff', 'ssl_tls_ratio',
            'resumed', 'self_signed_ratio', 'avg_key_length', 'avg_cert_valid_day',
            'std_cert_valid_day', 'percent_of_valid_cert', 'avg_valid_cert_percent',
            'number_of_cert_serial', 'number_of_domains_in_cert', 'avg_cert_path',
            'x509_ssl_ratio', 'SNI_ssl_ratio', 'is_SNIs_in_SNA_dns',
            'is_CNs_in_SNA_dns', 'subject_CN_is_IP', 'subject_is_com',
            'is_O_in_subject', 'is_CO_in_subject', 'is_ST_in_subject',
            'is_L_in_subject', 'subject_only_CN', 'issuer_is_com',
            'is_O_in_issuer', 'is_CO_in_issuer', 'is_ST_in_issuer',
            'is_L_in_issuer', 'issuer_only_CN', 'avg_TTL',
            'avg_domain_name_length', 'std_domain_name_length', 'avg_IPs_in_DNS',
            'flow_uid_list'
        ]

        output_file = os.path.join(path, filename)

        # ==========================================
        # 2. 收集 Session 并按其最早 Flow 时间排序
        # ==========================================
        all_sessions = []
        for key, session in self.session_tuple.items():
            start_t = 0.0
            if session.flow_list and len(session.flow_list) > 0:
                start_t = session.flow_list[0].start_time
            all_sessions.append((key, session, start_t))

        # 按照 start_t 排序
        all_sessions.sort(key=lambda x: x[2])

        with open(output_file, 'w+', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)

            for key, session, _ in all_sessions:
                is_malicious = session.is_malicious()
                # 清洗 flow_uid_list，去除 None 和 无效值
                raw_uid_list = [flow.uid for flow in session.flow_list]
                flow_uid_list = [
                    u for u in raw_uid_list
                    if u and str(u).strip() != "" and str(u).lower() not in ['-', 'none', 'nan']
                ]

                # 如果一个会话里面全是坏的流（虽然不太可能），则不写入该会话
                if not flow_uid_list:
                    continue

                session_feature = [
                    str(key),
                    is_malicious,
                    str(session.ssl_version()),
                    str(session.cipher_suite_server()),
                    str(session.cert_key_alg()),
                    str(session.cert_sig_alg()),
                    str(session.cert_key_type()),
                    str(session.max_duration()),
                    str(session.avg_duration()),
                    str(session.percent_of_std_duration()),
                    str(session.number_of_flows()),
                    str(session.ssl_flow_ratio()),
                    str(session.avg_size()),
                    str(session.recv_sent_size_ratio()),
                    str(session.avg_pkts()),
                    str(session.recv_sent_pkts_ratio()),
                    str(session.packet_loss()),
                    str(session.percent_of_established_state()),
                    str(session.avg_time_diff()),
                    str(session.std_time_diff()),
                    str(session.max_time_diff()),
                    str(session.ssl_tls_ratio()),
                    str(session.resumed()),
                    str(session.self_signed_ratio()),
                    str(session.avg_key_length()),
                    str(session.avg_cert_valid_day()),
                    str(session.std_cert_valid_day()),
                    str(session.percent_of_valid_cert()),
                    str(session.avg_valid_cert_percent()),
                    str(session.number_of_cert_serial()),
                    str(session.number_of_domains_in_cert()),
                    str(session.avg_cert_path()),
                    str(session.x509_ssl_ratio()),
                    str(session.SNI_ssl_ratio()),
                    str(session.is_SNIs_in_SNA_dns()),
                    str(session.is_CNs_in_SNA_dns()),
                    str(session.subject_CN_is_IP()),
                    str(session.subject_is_com()),
                    str(session.is_O_in_subject()),
                    str(session.is_CO_in_subject()),
                    str(session.is_ST_in_subject()),
                    str(session.is_L_in_subject()),
                    str(session.subject_only_CN()),
                    str(session.issuer_is_com()),
                    str(session.is_O_in_issuer()),
                    str(session.is_CO_in_issuer()),
                    str(session.is_ST_in_issuer()),
                    str(session.is_L_in_issuer()),
                    str(session.issuer_only_CN()),
                    str(session.avg_TTL()),
                    str(session.avg_domain_name_length()),
                    str(session.std_domain_name_length()),
                    str(session.avg_IPs_in_DNS()),
                    str(flow_uid_list)
                ]
                writer.writerow(session_feature)

        print(f"<<< dataset file {filename} successfully created !")