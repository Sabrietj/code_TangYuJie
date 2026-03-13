import os, sys
import argparse
from flow_node_builder import FlowNodeBuilder
from session_graph_builder import SessionGraphBuilder
import logging

# 添加../utils目录到Python搜索路径
utils_path = os.path.join(os.path.dirname(__file__), '..', 'utils')
sys.path.insert(0, utils_path)

import config_manager as ConfigManager
from logging_config import setup_preset_logging
# 使用统一的日志配置
logger = setup_preset_logging(log_level=logging.INFO)

def main():
    parser = argparse.ArgumentParser(description="Construct multi-flow session graphs from dataset filepaths defined in config.cfg.")
    
    try:
        dataset_dir = ConfigManager.read_plot_data_path_config()
        
        # 从配置文件读取线程数
        thread_count = ConfigManager.read_thread_count_config()
        logger.info(f"配置了内核线程数 = {thread_count}")

        session_label_id_map = ConfigManager.read_session_label_id_map()
        logger.info(f"配置了 session label string-to-id mapping: {session_label_id_map}")
        
        # 1. 先处理flow数据
        logger.info("开始处理 Flow 数据文件 ...")
        merged_flow_path = os.path.join(dataset_dir, "all_embedded_flow.csv")
        if not os.path.exists(merged_flow_path):
            raise FileNotFoundError(f"合并后的 Flow 数据文件不存在: {merged_flow_path}")
        else:
            logger.info(f"合并后的 Flow 数据文件 {merged_flow_path} 已经存在!")
        
        enabled_flow_node_views = ConfigManager.read_enabled_flow_node_views_config()
        logger.info(f"Enabled flow node views = {enabled_flow_node_views}")
        max_packet_sequence_length = ConfigManager.read_max_packet_sequence_length()
        logger.info(f"max_packet_sequence_length = {max_packet_sequence_length}，长度不足的packet sequences会做padding，超长的予以截断")
        text_encoder_name, max_text_length = ConfigManager.read_text_encoder_config()        
        logger.info(f"text_encoder_name = {text_encoder_name}, max_text_length = {max_text_length}")
        flow_node_builder = FlowNodeBuilder(flow_csv_path = merged_flow_path,
                                            session_label_id_map = session_label_id_map,
                                            max_packet_sequence_length = max_packet_sequence_length,
                                            text_encoder_name = text_encoder_name,
                                            max_text_length = max_text_length,
                                            thread_count = thread_count,
                                            enabled_views = enabled_flow_node_views)

        # 2. 构建 session graph
        logger.info("开始处理 Session 数据文件 ...")        
        merged_session_path = os.path.join(dataset_dir, "all_split_session.csv")
        if not os.path.exists(merged_session_path):
            raise FileNotFoundError(f"合并后的 Session 数据文件不存在: {merged_session_path}")
        else: 
            logger.info(f"合并后的 Session 数据文件 {merged_session_path} 已经存在!")
        
        dump_file = os.path.join(dataset_dir, "all_session_graph")
        logger.info(f"正在构建统一的 session graph 到输出文件路径 {dump_file}.bin ...")
        concurrent_flow_iat_threshold = ConfigManager.read_concurrent_flow_iat_threshold()
        sequential_flow_iat_threshold = ConfigManager.read_sequential_flow_iat_threshold()    
        logger.info(f"使用配置参数: concurrent_flow_iat_threshold={concurrent_flow_iat_threshold}, sequential_flow_iat_threshold={sequential_flow_iat_threshold}")
        builder = SessionGraphBuilder(flow_node_builder, merged_session_path, dump_file, 
                                      concurrent_flow_iat_threshold= concurrent_flow_iat_threshold,
                                      sequential_flow_iat_threshold=sequential_flow_iat_threshold,
                                      session_label_id_map = session_label_id_map, 
                                      thread_count = thread_count)
        
        logger.info(f"[SUCCESS] 成功构建 unified session graph! 输出文件: {dump_file}.bin")
        
    except Exception as e:
        import traceback
        logger.error(f"构建 graph 失败: {str(e)}")
        logger.error("详细错误信息:")
        # 使用 traceback.format_exc() 获取完整的堆栈跟踪信息
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
