# -*- coding: utf-8 -*-
import numpy as np
import socket
import time
import datetime
import traceback


class FlowTuple:
    def __init__(self, conn_uid, conn_log, ssl_log=None, x509_logs=None, dns_log=None, flowmeter_log=None,
                 http_log=None, ftp_log=None, mqtt_log=None):
        self.uid = conn_uid

        # 五元组 (srcIP, srcPort, dstIP, dstPort, proto)
        self.src_ip = conn_log.get("id.orig_h", None)
        self.src_port = conn_log.get("id.orig_p", None)
        self.dst_ip = conn_log.get("id.resp_h", None)
        self.dst_port = conn_log.get("id.resp_p", None)
        self.proto = conn_log.get("proto", None)

        # 开始时间戳
        try:
            self.start_time = float(conn_log.get("ts", 0.0))
        except Exception:
            self.start_time = 0.0

        # 流时间戳
        try:
            self.duration = float(conn_log.get("duration", 0.0))
        except Exception:
            self.duration = 0.0

        # 标签
        self._label = conn_log.get("label", "Background")
        self._is_malicious = FlowTuple.is_malicious_label(self._label)

        # 保存原始日志
        self.conn_log = conn_log
        self.ssl_log = ssl_log if ssl_log is not None else {}
        # 初始化为空列表，防止 NoneType 错误
        self.x509_logs = x509_logs if x509_logs is not None else []
        self.dns_log = dns_log if dns_log is not None else {}
        self.flowmeter_log = flowmeter_log if flowmeter_log is not None else {}
        self.http_log = http_log if http_log is not None else {}
        self.ftp_log = ftp_log if ftp_log is not None else {}
        self.mqtt_log = mqtt_log if mqtt_log is not None else {}

    def get_uid(self):
        return self.uid

    def get_timestamp(self):
        return self.start_time

    def get_duration(self):
        return self.duration

    @staticmethod
    def is_malicious_label(label_str):
        if label_str is None:
            return True
        label_str = str(label_str).lower().strip()
        if label_str.isdigit():
            return int(label_str) != 0
        normal_keywords = ['benign', 'normal', 'legitimate', 'clean', 'safe']
        for keyword in normal_keywords:
            if keyword in label_str:
                return False
        background_keywords = ['background', 'unknown', 'unlabeled']
        for keyword in background_keywords:
            if keyword in label_str:
                return False
        return True

    def is_malicious(self):
        if self._is_malicious is None:
            return -1
        return int(self._is_malicious)

    def get_label(self):
        return self._label


class SessionTuple:
    def __init__(self, tuple_index):
        self.tuple_index = tuple_index

        # label
        self._is_malicious = False

        self.flow_list = []  # 存储 FlowTuple 序列

        # [Fix] 增加 flow_map 以便后续根据 UID 快速查找并更新 SSL/DNS/FlowMeter 日志
        self.flow_map = {}

        # Connection Features
        self.conn_log = []
        self.number_of_ssl_flows = 0
        self.number_of_not_ssl_flows = 0
        self.resp_bytes_list = []
        self.orig_bytes_list = []
        self.conn_state_dict = dict()
        self.duration_list = []
        self.resp_pkts_list = []
        self.orig_pkts_list = []
        self._packet_loss = 0

        # SSL features
        self.ssl_version_dict = dict()
        self.ssl_cipher_dict = dict()
        self.cert_path_length = []
        self.ssl_with_SNI = 0
        self._self_signed_cert = 0
        self._resumed = 0

        # X509 features
        self.number_of_x509 = 0
        self.cert_key_dict = dict()
        self.cert_key_length_list = []
        self.cert_serial_set = set()
        self.cert_valid_days = []
        self.invalid_cert_number = 0
        self.san_domain_list = []
        self.cert_validity_percent = []
        self._is_CNs_in_SAN = []
        self._is_SNIs_in_SAN_dns = []
        self._subject_CN_is_IP = []
        self.key_alg = set()
        self.sig_alg = set()
        self.key_type = set()

        self._subject_is_com = []
        self._is_O_in_subject = []
        self._is_CO_in_subject = []
        self._is_ST_in_subject = []
        self._is_L_in_subject = []
        self._subject_only_CN = []

        self._issuer_is_com = []
        self._is_O_in_issuer = []
        self._is_CO_in_issuer = []
        self._is_ST_in_issuer = []
        self._is_L_in_issuer = []
        self._issuer_only_CN = []

        # DNS features
        self.dns_uid_set = set()
        self.TTL_list = []
        self.number_of_IPs_in_DNS = []
        self.domain_name_length = []

    def get_label(self):
        return self._label

    # [Fix] 辅助函数：创建并添加 FlowTuple
    def _create_and_add_flow(self, conn_log):
        uid = None
        if isinstance(conn_log, dict):
            uid = conn_log.get('uid')
        elif isinstance(conn_log, str):
            # 兼容旧的文本日志格式
            parts = conn_log.split('\t')
            if len(parts) > 1: uid = parts[1]

        if not uid or str(uid).strip() == "" or str(uid).lower() in ['-', 'none', 'nan']:
            return None
        flow = FlowTuple(uid, conn_log)
        self.flow_list.append(flow)

        # 建立索引，供后续日志匹配使用
        if uid:
            self.flow_map[uid] = flow
        return flow

    def add_ssl_flow(self, conn_log):
        if isinstance(conn_log, dict):
            label = conn_log.get('label', 'Background')
        else:
            conn_split = conn_log.split('\t')
            label = conn_split[-1].strip() if conn_split else ''

        self._is_malicious = self._is_malicious or FlowTuple.is_malicious_label(label)

        self.conn_log.append(conn_log)
        self.number_of_ssl_flows += 1

        # [Fix] 关键修复：创建 FlowTuple 并存入列表
        self._create_and_add_flow(conn_log)

        self.compute_conn_features(conn_log)

    def add_not_ssl_flow(self, conn_log):
        if isinstance(conn_log, dict):
            label = conn_log.get('label', '')
        else:
            conn_split = conn_log.split('\t')
            label = conn_split[-1].strip() if conn_split else ''

        self._is_malicious = FlowTuple.is_malicious_label(label)

        self.conn_log.append(conn_log)
        self.number_of_not_ssl_flows += 1

        # [Fix] 关键修复：创建 FlowTuple 并存入列表
        self._create_and_add_flow(conn_log)

        self.compute_conn_features(conn_log)

    def add_ssl_log(self, ssl_log, debug_uid=None):
        if debug_uid:
            print(f">>> [add_ssl_log] Processing SSL log, UID: {debug_uid}")

        # [Fix] 关联 SSL 日志到具体的 FlowTuple
        uid = ssl_log.get('uid')
        if uid and uid in self.flow_map:
            self.flow_map[uid].ssl_log = ssl_log
            if debug_uid:
                print(f">>> [add_ssl_log] Linked SSL log to FlowTuple: {uid}")

        self.compute_ssl_features(ssl_log, debug_uid=debug_uid)

    def add_x509_log(self, x509_log):
        # X509通常用于Session统计，Flow级别关联比较复杂（需通过SSL指纹），此处保留Session统计
        self.number_of_x509 += 1
        self.compute_x509_features(x509_log)

    def add_dns_log(self, conn_log, dns_log):
        # [Fix] 关联 DNS 日志到具体的 FlowTuple
        uid = dns_log.get('uid')
        if uid and uid in self.flow_map:
            self.flow_map[uid].dns_log = dns_log

        self.compute_dns_features(conn_log, dns_log)

    # [Fix] 允许外部(如 analyze_log.py) 设置 FlowMeter 数据
    def set_flowmeter_log(self, uid, flowmeter_data):
        if uid and uid in self.flow_map:
            self.flow_map[uid].flowmeter_log = flowmeter_data

    # ================= 下面是统计特征计算部分 (保持原样) =================

    def compute_conn_features(self, conn_log):
        if isinstance(conn_log, dict):
            duration = conn_log.get('duration', '0')
            orig_bytes = conn_log.get('orig_bytes', '0')
            resp_bytes = conn_log.get('resp_bytes', '0')
            conn_state = str(conn_log.get('conn_state', '-'))
            missed_bytes = conn_log.get('missed_bytes', '0')
            orig_pkts = conn_log.get('orig_pkts', '0')
            resp_pkts = conn_log.get('resp_pkts', '0')
        else:
            conn_split = conn_log.split('\t')
            duration = conn_split[7] if len(conn_split) > 7 else '0'
            orig_bytes = conn_split[9] if len(conn_split) > 9 else '0'
            resp_bytes = conn_split[10] if len(conn_split) > 10 else '0'
            conn_state = conn_split[11] if len(conn_split) > 11 else '-'
            missed_bytes = conn_split[12] if len(conn_split) > 12 else '0'
            orig_pkts = conn_split[13] if len(conn_split) > 13 else '0'
            resp_pkts = conn_split[14] if len(conn_split) > 14 else '0'

        try:
            duration = float(duration)
            self.duration_list.append(duration)
        except:
            pass

        try:
            orig_bytes_number = int(orig_bytes)
            self.orig_bytes_list.append(orig_bytes_number)
        except:
            pass

        try:
            resp_bytes_number = int(resp_bytes)
            self.resp_bytes_list.append(resp_bytes_number)
        except:
            pass

        if conn_state in self.conn_state_dict:
            self.conn_state_dict[conn_state] += 1
        else:
            self.conn_state_dict[conn_state] = 1

        try:
            missed_bytes_number = int(missed_bytes)
            self._packet_loss += missed_bytes_number
        except:
            pass

        try:
            orig_pkts_number = int(orig_pkts)
            self.orig_pkts_list.append(orig_pkts_number)
        except:
            pass

        try:
            resp_pkts_number = int(resp_pkts)
            self.resp_pkts_list.append(resp_pkts_number)
        except:
            pass

    def compute_ssl_features(self, ssl_log, debug_uid=None):
        try:
            resumed = ssl_log.get('resumed', False)
            if resumed in [True, 'true', 'T', 't', 'True', 'yes', '1']:
                self._resumed += 1

            version = ssl_log.get('version', '')
            if version:
                version_upper = version.upper()
                if version_upper in self.ssl_version_dict:
                    self.ssl_version_dict[version_upper] += 1
                else:
                    self.ssl_version_dict[version_upper] = 1

            cipher = ssl_log.get('cipher', '')
            if cipher:
                if cipher in self.ssl_cipher_dict:
                    self.ssl_cipher_dict[cipher] += 1
                else:
                    self.ssl_cipher_dict[cipher] = 1

            cert_chain_uids = ssl_log.get('cert_chain_fps', [])
            if cert_chain_uids and len(cert_chain_uids) > 0:
                self.cert_path_length.append(len(cert_chain_uids))

            server_name = ssl_log.get('server_name', '')
            if server_name and server_name != '-':
                self.ssl_with_SNI += 1

        except Exception as e:
            print(f">>>   ❌ compute_ssl_features error: {e}")
            traceback.print_exc()

    def compute_x509_features(self, x509_log):
        try:
            key_alg = x509_log.get('certificate.key_alg', '-')
            if key_alg and key_alg != '-' and key_alg != 'None':
                self.key_alg.add(str(key_alg))
        except Exception as e:
            pass

        try:
            sig_alg = x509_log.get('certificate.sig_alg', '-')
            if sig_alg and sig_alg != '-' and sig_alg != 'None':
                self.sig_alg.add(str(sig_alg))
        except Exception as e:
            pass

        try:
            key_type = x509_log.get('certificate.key_type', '-')
            if key_type and key_type != '-' and key_type != 'None':
                self.key_type.add(str(key_type))
        except Exception as e:
            pass

        try:
            current_time = float(x509_log['ts'])
            before_time = float(x509_log['certificate.not_valid_before'])
            after_time = float(x509_log['certificate.not_valid_after'])
            if current_time > after_time or current_time < before_time:
                self.invalid_cert_number += 1
            else:
                date1 = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(int(before_time)))
                date2 = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(int(after_time)))
                date1 = time.strptime(date1, "%Y-%m-%d-%H-%M-%S")
                date2 = time.strptime(date2, "%Y-%m-%d-%H-%M-%S")
                d1 = datetime.datetime(date1[0], date1[1], date1[2])
                d2 = datetime.datetime(date2[0], date2[1], date2[2])
                valid_days = (d2 - d1).days
                if valid_days >= 0:
                    self.cert_valid_days.append(valid_days)

                norm_after = after_time - before_time
                current_time_norm = current_time - before_time
                if norm_after > 0:
                    self.cert_validity_percent.append(current_time_norm / norm_after)
        except:
            pass

        cert_serial = x509_log.get('certificate.serial')
        if cert_serial and cert_serial not in self.cert_serial_set:
            self.cert_serial_set.add(cert_serial)

            try:
                length = int(x509_log.get('certificate.key_length', 0))
                self.cert_key_length_list.append(length)
            except:
                pass

            domains = x509_log.get('san.dns', [])
            if isinstance(domains, str) and domains != '-':
                domains = domains.split(',') if domains else []
            elif not isinstance(domains, list):
                domains = []

            CN = None
            subject = x509_log.get('certificate.subject', '')
            if subject and subject != '-':
                subject_parts = subject.split(',')
                for part in subject_parts:
                    part = part.strip()
                    if part.startswith("CN="):
                        CN = part[3:]
                        break

            if CN and domains:
                if CN in domains:
                    self._is_CNs_in_SAN.append(1)
                else:
                    self._is_CNs_in_SAN.append(0)

                try:
                    socket.inet_aton(CN)
                    self._subject_CN_is_IP.append(1)
                except socket.error:
                    self._subject_CN_is_IP.append(0)
            else:
                self._is_CNs_in_SAN.append(0)
                self._subject_CN_is_IP.append(0)

            for domain in domains:
                self.san_domain_list.append(domain)

        subject = x509_log.get('certificate.subject', '').split(',')
        CN = 0
        for key in subject:
            if 'CN=' in key:
                CN += 1
                if '.com' in key:
                    self._subject_is_com.append(1)
                else:
                    self._subject_is_com.append(0)

            if 'O=' in key:
                self._is_O_in_subject.append(1)
            else:
                self._is_O_in_subject.append(0)

            if 'CO=' in key:
                self._is_CO_in_subject.append(1)
            else:
                self._is_CO_in_subject.append(0)

            if 'ST=' in key:
                self._is_ST_in_subject.append(1)
            else:
                self._is_ST_in_subject.append(0)

            if 'L=' in key:
                self._is_L_in_subject.append(1)
            else:
                self._is_L_in_subject.append(0)

        if len(subject) > 0 and CN == len(subject):
            self._subject_only_CN.append(1)
        else:
            self._subject_only_CN.append(0)

        issuer = x509_log.get('certificate.issuer', '').split(',')
        CN = 0
        for key in issuer:
            if 'CN=' in key:
                CN += 1
                if '.com' in key:
                    self._issuer_is_com.append(1)
                else:
                    self._issuer_is_com.append(0)

            if 'O=' in key:
                self._is_O_in_issuer.append(1)
            else:
                self._is_O_in_issuer.append(0)

            if 'CO=' in key:
                self._is_CO_in_issuer.append(1)
            else:
                self._is_CO_in_issuer.append(0)

            if 'ST=' in key:
                self._is_ST_in_issuer.append(1)
            else:
                self._is_ST_in_issuer.append(0)

            if 'L=' in key:
                self._is_L_in_issuer.append(1)
            else:
                self._is_L_in_issuer.append(0)

        if len(issuer) > 0 and CN == len(issuer):
            self._issuer_only_CN.append(1)
        else:
            self._issuer_only_CN.append(0)

    def compute_dns_features(self, conn_log, dns_log):
        if not dns_log:
            return

        domain = dns_log.get('query', '')
        self.domain_name_length.append(len(domain))

        answers = dns_log.get('answers', [])
        if isinstance(answers, str):
            dns_ans_list = answers.split(',')
        elif isinstance(answers, list):
            dns_ans_list = answers
        else:
            dns_ans_list = []

        self.number_of_IPs_in_DNS.append(len(dns_ans_list))

        ttls = dns_log.get('TTLs', [])
        if isinstance(ttls, str):
            TTLs = ttls.split(',')
        elif isinstance(ttls, list):
            TTLs = ttls
        else:
            TTLs = []

        try:
            dstIP = conn_log.get('id.resp_h', '')
            if dstIP in dns_ans_list:
                pos = dns_ans_list.index(dstIP)
                if pos < len(TTLs):
                    TTL = float(TTLs[pos])
                    self.TTL_list.append(TTL)
        except Exception:
            pass

    def flow_inter_arrival(self):
        flow_time_list = []
        for conn in self.conn_log:
            # Handle both dict and text format if needed
            if isinstance(conn, dict):
                ts = conn.get('ts', 0)
            else:
                ts = conn.split('\t')[0]  # Assuming ts is first
            flow_time_list.append(float(ts))
        flow_time_list.sort()

        pre_flow = flow_time_list[:-1]
        next_flow = flow_time_list[1:]
        time_diff_list = [
            next_flow[i] - pre_flow[i] for i in range(len(pre_flow))
        ]

        return time_diff_list

    def std_duration(self):
        if self.duration_list:
            return np.std(self.duration_list)
        else:
            return -1.0

    def avg_sent_size(self):
        if self.orig_bytes_list:
            return np.mean(self.orig_bytes_list)
        else:
            return 0

    def avg_recv_size(self):
        if self.resp_bytes_list:
            return np.mean(self.resp_bytes_list)
        else:
            return 0

    def avg_pkts_sent(self):
        if self.orig_pkts_list:
            return np.mean(self.orig_pkts_list)
        else:
            return 0

    def avg_pkts_recv(self):
        if self.resp_pkts_list:
            return np.mean(self.resp_pkts_list)
        else:
            return 0

    def is_SNI_in_cert(self, ssl_log, x509_log):
        SNI = ssl_log.get('server_name')
        if SNI and SNI != '-':
            if x509_log.get('san.dns'):
                san_dns_list = x509_log['san.dns']
                if SNI in san_dns_list:
                    self._is_SNIs_in_SAN_dns.append(1)
                else:
                    self._is_SNIs_in_SAN_dns.append(0)
            else:
                x509_log['san.dns'] = "-"

    def max_duration(self):
        if self.duration_list:
            return max(self.duration_list)
        else:
            return 0.0

    def avg_duration(self):
        if self.duration_list:
            return np.mean(self.duration_list)
        else:
            return 0.0

    def percent_of_std_duration(self):
        std_dur = self.std_duration()
        avg_dur = self.avg_duration()
        upper_dur = avg_dur + abs(std_dur)
        lower_dur = avg_dur - abs(std_dur)
        count = 0
        if std_dur != -1.0:
            for d in self.duration_list:
                if d >= lower_dur and d <= upper_dur:
                    count += 1
            if self.duration_list:
                return float(count / len(self.duration_list))
        return -1.0

    def number_of_flows(self):
        return self.number_of_ssl_flows + self.number_of_not_ssl_flows

    def ssl_flow_ratio(self):
        flow_number = self.number_of_flows()
        if flow_number > 0:
            return float(self.number_of_ssl_flows / flow_number)
        else:
            return -1.0

    def avg_size(self):
        return self.avg_sent_size() + self.avg_recv_size()

    def recv_sent_size_ratio(self):
        if self.avg_sent_size() > 0:
            return float(self.avg_recv_size() / self.avg_sent_size())
        else:
            return -1.0

    def avg_pkts(self):
        return self.avg_pkts_sent() + self.avg_pkts_recv()

    def recv_sent_pkts_ratio(self):
        if self.avg_pkts_sent():
            return float(self.avg_pkts_recv() / self.avg_pkts_sent())
        else:
            return -1.0

    def packet_loss(self):
        return self._packet_loss

    def percent_of_established_state(self):
        est_state = 0
        total_length_state = 0
        for key in self.conn_state_dict:
            total_length_state += self.conn_state_dict[key]
        if total_length_state > 0:
            est_state += self.conn_state_dict.get('SF', 0)
            est_state += self.conn_state_dict.get('S1', 0)
            est_state += self.conn_state_dict.get('S2', 0)
            est_state += self.conn_state_dict.get('S3', 0)
            est_state += self.conn_state_dict.get('RSTO', 0)
            est_state += self.conn_state_dict.get('RSTR', 0)
            return float(est_state / total_length_state)
        return -1.0

    def avg_time_diff(self):
        time_diff = self.flow_inter_arrival()
        if time_diff:
            return np.mean(time_diff)
        else:
            return 0.0

    def std_time_diff(self):
        time_diff = self.flow_inter_arrival()
        if time_diff:
            return np.std(time_diff)
        else:
            return -1.0

    def max_time_diff(self):
        time_diff = self.flow_inter_arrival()
        if time_diff:
            return max(time_diff)
        else:
            return 0.0

    def ssl_tls_ratio(self):
        tls = 0
        ssl = 0
        if self.ssl_version_dict:
            for key in self.ssl_version_dict:
                if 'TLS' in key: tls += 1
                if 'SSL' in key: ssl += 1
            if tls > 0: return float(ssl / tls)
        return -1.0

    def ssl_version(self):
        if self.ssl_version_dict:
            ssl_version = list(self.ssl_version_dict.keys())
            ssl_version.sort()
            return ssl_version
        else:
            return None

    def cipher_suite_server(self):
        if self.ssl_cipher_dict:
            cipher_suite = list(self.ssl_cipher_dict.keys())
            cipher_suite.sort()
            return cipher_suite
        else:
            return None

    def resumed(self):
        return self._resumed

    def self_signed_ratio(self):
        if self.number_of_ssl_flows:
            return float(self._self_signed_cert / self.number_of_ssl_flows)
        return -1.0

    def avg_key_length(self):
        if self.cert_key_length_list:
            return np.mean(self.cert_key_length_list)
        else:
            return -1.0

    def avg_cert_valid_day(self):
        if self.cert_valid_days:
            return np.mean(self.cert_valid_days)
        else:
            return 0.0

    def std_cert_valid_day(self):
        if self.cert_valid_days:
            return np.std(self.cert_valid_days)
        else:
            return -1.0

    def percent_of_valid_cert(self):
        valid_cert = len(self.cert_validity_percent)
        total = valid_cert + self.invalid_cert_number
        if total > 0:
            return float(valid_cert / total)
        else:
            return -1.0

    def avg_valid_cert_percent(self):
        if self.cert_validity_percent:
            return np.mean(self.cert_validity_percent)
        else:
            return 0.0

    def number_of_cert_serial(self):
        return len(self.cert_serial_set)

    def number_of_domains_in_cert(self):
        domain_set = set(self.san_domain_list)
        return len(domain_set)

    def avg_cert_path(self):
        if self.cert_path_length:
            return np.mean(self.cert_path_length)
        else:
            return -1.0

    def x509_ssl_ratio(self):
        if self.number_of_ssl_flows:
            return float(self.number_of_x509 / self.number_of_ssl_flows)
        else:
            return -1.0

    def SNI_ssl_ratio(self):
        if self.number_of_ssl_flows:
            return float(self.ssl_with_SNI / self.number_of_ssl_flows)
        else:
            return -1.0

    def is_SNIs_in_SNA_dns(self):
        if self._is_SNIs_in_SAN_dns:
            if 0 in self._is_SNIs_in_SAN_dns: return 0
            return 1
        return -1

    def is_CNs_in_SNA_dns(self):
        if self._is_CNs_in_SAN:
            if 0 in self._is_CNs_in_SAN: return 0
            return 1
        return -1

    def subject_CN_is_IP(self):
        if self._subject_CN_is_IP:
            return np.mean(self._subject_CN_is_IP)
        else:
            return 0

    def cert_key_alg(self):
        if self.key_alg:
            key_alg = list(self.key_alg)
            key_alg.sort()
            return key_alg
        else:
            return None

    def cert_sig_alg(self):
        if self.sig_alg:
            sig_alg = list(self.sig_alg)
            sig_alg.sort()
            return sig_alg
        else:
            return None

    def cert_key_type(self):
        if self.key_type:
            key_type = list(self.key_type)
            key_type.sort()
            return key_type
        else:
            return None

    def subject_is_com(self):
        if self._subject_is_com:
            return np.mean(self._subject_is_com)
        else:
            return 0

    def is_O_in_subject(self):
        if self._is_O_in_subject:
            return np.mean(self._is_O_in_subject)
        else:
            return 0

    def is_CO_in_subject(self):
        if self._is_CO_in_subject:
            return np.mean(self._is_CO_in_subject)
        else:
            return 0

    def is_ST_in_subject(self):
        if self._is_ST_in_subject:
            return np.mean(self._is_ST_in_subject)
        else:
            return 0

    def is_L_in_subject(self):
        if self._is_L_in_subject:
            return np.mean(self._is_L_in_subject)
        else:
            return 0

    def subject_only_CN(self):
        if self._subject_only_CN:
            return np.mean(self._subject_only_CN)
        else:
            return 0

    def issuer_is_com(self):
        if self._issuer_is_com:
            return np.mean(self._issuer_is_com)
        else:
            return 0

    def is_O_in_issuer(self):
        if self._is_O_in_issuer:
            return np.mean(self._is_O_in_issuer)
        else:
            return 0

    def is_CO_in_issuer(self):
        if self._is_CO_in_issuer:
            return np.mean(self._is_CO_in_issuer)
        else:
            return 0

    def is_ST_in_issuer(self):
        if self._is_ST_in_issuer:
            return np.mean(self._is_ST_in_issuer)
        else:
            return 0

    def is_L_in_issuer(self):
        if self._is_L_in_issuer:
            return np.mean(self._is_L_in_issuer)
        else:
            return 0

    def issuer_only_CN(self):
        if self._issuer_only_CN:
            return np.mean(self._issuer_only_CN)
        else:
            return 0

    def avg_TTL(self):
        if self.TTL_list:
            return np.mean(self.TTL_list)
        else:
            return 0.0

    def avg_domain_name_length(self):
        if self.domain_name_length:
            return np.mean(self.domain_name_length)
        else:
            return 0.0

    def std_domain_name_length(self):
        if self.domain_name_length:
            return np.std(self.domain_name_length)
        else:
            return -1.0

    def avg_IPs_in_DNS(self):
        if self.number_of_IPs_in_DNS:
            return np.mean(self.number_of_IPs_in_DNS)
        else:
            return 0.0

    def is_malicious(self):
        if self._is_malicious is None: return -1
        return int(self._is_malicious)

    def get_number_of_ssl_flows(self):
        return self.number_of_ssl_flows