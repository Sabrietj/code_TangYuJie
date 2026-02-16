# -*- coding: UTF-8 -*-
import sys
import time
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import shutil
import traceback

# 添加../utils目录到Python搜索路径
utils_path = os.path.join(os.path.dirname(__file__), '..', 'utils')
sys.path.insert(0, utils_path)

from logging_config import setup_preset_logging
import logging
# 使用统一的日志配置
logger = setup_preset_logging(log_level=logging.INFO)

import config_manager as ConfigManager
from print_manager import __PrintManager__
from analyze_log import LogAnalyzer

def process_directory(logAnalyzer, dir_path, plot_data_base_path, dir_name=None):
    """处理单个目录的任务函数"""
    try:
        conn_log_path = os.path.join(dir_path, "conn_label.log")
        if os.path.exists(conn_log_path):
            __PrintManager__.single_folder_header(dir_path)
            
            # 处理特征评估
            logAnalyzer.evaluate_features(dir_path)
            __PrintManager__.succ_single_folder_header()
            
            # 确定输出目录
            if dir_name is None:
                # 根目录的情况
                output_dir = plot_data_base_path
                output_name = "."
            else:
                # 子目录的情况
                output_dir = os.path.join(plot_data_base_path, dir_name)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                output_name = os.path.basename(dir_path)
            
            # 创建输出数据
            logAnalyzer.create_plot_data(output_dir, output_name)
            return True
        return False
    except Exception as e:
        print(f"Error processing directory {dir_path}: {str(e)}")
        traceback.print_exc()
        return False

def find_conn_log_directories(base_path):
    """递归查找包含conn_label.log的目录"""
    conn_dirs = []
    
    # 检查根目录
    if os.path.exists(os.path.join(base_path, "conn_label.log")):
        conn_dirs.append((base_path, None))  # (目录路径, 父目录名)
    
    # 遍历一级子目录
    for dir_name in os.listdir(base_path):
        dir_path = os.path.join(base_path, dir_name)
        if os.path.isdir(dir_path):
            # 检查一级子目录
            if os.path.exists(os.path.join(dir_path, "conn_label.log")):
                conn_dirs.append((dir_path, dir_name))
            else:
                # 检查二级子目录
                for sub_dir in os.listdir(dir_path):
                    sub_path = os.path.join(dir_path, sub_dir)
                    if os.path.isdir(sub_path) and os.path.exists(os.path.join(sub_path, "conn_label.log")):
                        conn_dirs.append((sub_path, dir_name))
    
    return conn_dirs

if __name__ == "__main__":
    t0 = time.time()
    __PrintManager__.welcome_header()

    # The argument of this program should be name of the resulting plot data file.
    # If there is no argument, default name for plot data is: 'dataset-YYYYMMDD-HHMMSS.csv'
    local_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    result_file = local_time
    if len(sys.argv) == 2:
        result_file = sys.argv[1]

    # Get path to multi dataset from config file.
    dataset_path = ConfigManager.read_dataset_path_config()
    plot_data_path = ConfigManager.read_plot_data_path_config()
    if not os.path.exists(plot_data_path):
        os.makedirs(plot_data_path)
    else:
        if os.listdir(plot_data_path):
            print(f"plot_data_path '{plot_data_path}' 已存在且非空，开始执行清空操作…")
            for filename in os.listdir(plot_data_path):
                file_path = os.path.join(plot_data_path, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"清理 {file_path} 时发生错误: {e}")
                    traceback.print_exc()
                    raise

            print(f"plot_data_path '{plot_data_path}' 已清空")

    if dataset_path == -1:
        raise ValueError

    # 读取线程数配置（默认为4个线程）
    try:
        thread_count = ConfigManager.read_thread_count_config()
    except:
        thread_count = 4  # 默认值
    
    print(f"Using {thread_count} threads for processing")

    # 查找所有包含conn_label.log的目录
    conn_dirs = find_conn_log_directories(dataset_path)
    __PrintManager__.dataset_folder_header([d[0] for d in conn_dirs], len(conn_dirs))
    
    # 创建线程池处理所有目录
    results = []
    with ThreadPoolExecutor(max_workers=thread_count) as executor:
        # 为每个目录创建一个LogAnalyzer实例（避免线程间冲突）
        analyzers = [LogAnalyzer() for _ in range(len(conn_dirs))]
        
        # 提交所有任务
        future_to_dir = {}
        for i, (dir_path, dir_name) in enumerate(conn_dirs):
            analyzer = analyzers[i]
            future = executor.submit(process_directory, analyzer, dir_path, plot_data_path, dir_name)
            future_to_dir[future] = (dir_path, dir_name)
        
        # 等待所有任务完成并收集结果
        for future in as_completed(future_to_dir):
            dir_path, dir_name = future_to_dir[future]
            try:
                result = future.result()
                results.append((dir_path, result))
                if result:
                    print(f"\n✓ Successfully processed: {dir_path}")
                else:
                    print(f"\n✗ No conn_label.log found, or encounter error when processing the directory: {dir_path}")
            except Exception as e:
                print(f"✗ Error processing {dir_path}: {str(e)}")
                traceback.print_exc()
                results.append((dir_path, False))
    
    # 统计处理结果
    success_count = sum(1 for _, result in results if result)
    print(f"\nProcessing completed: {success_count}/{len(conn_dirs)} directories successful")
    
    t1 = time.time()
    print("\n<<< Total approximate running time: %f min." % ((t1 - t0) / 60.0))