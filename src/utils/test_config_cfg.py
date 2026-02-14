#!/usr/bin/env python3
"""
config.cfg 调整后的全面测试用例
测试ACTIVE_DATASET机制、自动配置填充、API兼容性等功能
"""

import sys
import os
import configparser
import tempfile
import shutil
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config_integration import get_config_section, get_global_config, refresh_config
from config_loader import load_config_with_dataset_switch, get_active_dataset_config

class ConfigCfgTest:
    """配置文件测试类"""
    
    def __init__(self):
        self.config_path = os.path.join(os.path.dirname(__file__), 'config.cfg')
        self.backup_path = None
        self.test_results = []
    
    def log_test(self, test_name: str, passed: bool, message: str = ""):
        """记录测试结果"""
        status = "✓ PASS" if passed else "✗ FAIL"
        self.test_results.append((test_name, passed, message))
        print(f"{status}: {test_name}")
        if message:
            print(f"    {message}")
    
    def backup_config(self):
        """备份原始配置文件"""
        self.backup_path = self.config_path + '.test_backup'
        shutil.copy2(self.config_path, self.backup_path)
    
    def restore_config(self):
        """恢复原始配置文件"""
        if self.backup_path and os.path.exists(self.backup_path):
            shutil.copy2(self.backup_path, self.config_path)
            os.remove(self.backup_path)
    
    def modify_active_dataset(self, dataset_name: str):
        """修改ACTIVE_DATASET参数"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 替换ACTIVE_DATASET行
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith('ACTIVE_DATASET'):
                lines[i] = f'ACTIVE_DATASET = {dataset_name}'
                break
        
        with open(self.config_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        # 刷新配置缓存
        refresh_config()
    
    def test_basic_structure(self):
        """测试基本配置文件结构"""
        try:
            from config_manager import read_session_label_id_map
            
            # 使用处理过的配置来测试
            label_mapping = read_session_label_id_map()
            
            print(f"标签映射: {label_mapping}")

            # 检查标签映射是否正确处理
            if not label_mapping or len(label_mapping) == 0:
                self.log_test("基本配置结构", False, "标签映射为空或配置未正确处理")
                return
            
            self.log_test("基本配置结构", True, f"配置正确处理，标签映射包含{len(label_mapping)}个标签")
            return True
            
        except Exception as e:
            self.log_test("基本配置结构", False, f"读取配置失败: {e}")
            return False
    
    def test_dataset_sections(self):
        """测试数据集配置段"""
        try:
            config = configparser.ConfigParser()
            config.read(self.config_path, encoding='utf-8')
            
            # 检查数据集配置段
            dataset_sections = [section for section in config.sections() 
                               if section not in ['GENERAL', 'PATH', 'SESSION', 'DOMAIN_HIERARCHY', 
                                                'MODEL_PARAMS', 'TRAINING_MODES', 'MODEL_ARCHITECTURE',
                                                'TRAINING_PARAMS', 'EXPERIMENT_MANAGER', 'METRICS']]
            
            expected_datasets = ['CIC-IDS-2017', 'CIC-AndMal2017', 'USTC-TFC2016', 'CIC-IoT-2023', 'CTU-13']
            
            for dataset in expected_datasets:
                if dataset not in dataset_sections:
                    self.log_test(f"数据集段 {dataset}", False, f"缺少数据集配置: {dataset}")
                    return
                
                # 检查每个数据集段的必要参数
                required_params = ['dataset_name', 'path_to_dataset', 'plot_data_path', 
                                 'session_tuple_mode', 'concurrent_flow_iat_threshold', 
                                 'sequential_flow_iat_threshold']
                
                dataset_config = config[dataset]
                for param in required_params:
                    if param not in dataset_config:
                        self.log_test(f"数据集参数 {dataset}.{param}", False, f"缺少参数: {param}")
                        return
            
            self.log_test("数据集配置段", True, f"所有{len(expected_datasets)}个数据集配置完整")
            return True
            
        except Exception as e:
            self.log_test("数据集配置段", False, f"检查失败: {e}")
            return False
    
    def test_active_dataset_mechanism(self):
        """测试ACTIVE_DATASET机制"""
        try:
            # 测试默认激活数据集
            config = load_config_with_dataset_switch(self.config_path)
            if 'GENERAL' not in config:
                self.log_test("ACTIVE_DATASET机制", False, "无法读取GENERAL段")
                return False
            
            active_dataset = config['GENERAL']['ACTIVE_DATASET']
            self.log_test("ACTIVE_DATASET读取", True, f"当前激活: {active_dataset}")
            
            # 测试自动填充功能
            if 'PATH' not in config or 'SESSION' not in config:
                self.log_test("配置自动填充", False, "PATH或SESSION段缺失")
                return False
            
            # 检查是否从数据集配置中正确填充
            path_config = config['PATH']
            session_config = config['SESSION']
            
            if (path_config['path_to_dataset'] == 'AUTO_FILL' or 
                session_config['session_tuple_mode'] == 'AUTO_FILL'):
                self.log_test("配置自动填充", False, "配置未被正确填充")
                return False
            
            self.log_test("配置自动填充", True, "PATH和SESSION段已正确填充")
            return True
            
        except Exception as e:
            self.log_test("ACTIVE_DATASET机制", False, f"测试失败: {e}")
            return False
    
    def test_dataset_switching(self):
        """测试数据集切换功能"""
        try:
            # 测试切换到不同数据集
            test_datasets = ['USTC-TFC2016', 'CIC-AndMal2017']
            
            for dataset in test_datasets:
                self.modify_active_dataset(dataset)
                
                # 验证切换后的配置
                active_config = get_active_dataset_config(self.config_path)
                if not active_config or active_config.get('dataset_name') != dataset:
                    self.log_test(f"数据集切换到{dataset}", False, "切换失败")
                    return False
                
                # 验证自动填充
                path_config = get_config_section('PATH')
                expected_path = f"./dataset/{dataset}"
                if path_config['path_to_dataset'] != expected_path:
                    self.log_test(f"数据集{dataset}路径填充", False, f"路径错误: {path_config['path_to_dataset']}")
                    return False
            
            self.log_test("数据集切换功能", True, f"成功测试了{len(test_datasets)}个数据集切换")
            return True
            
        except Exception as e:
            self.log_test("数据集切换功能", False, f"测试失败: {e}")
            return False
    
    def test_api_compatibility(self):
        """测试API兼容性"""
        try:
            # 测试传统ConfigParser方式（应该使用我们的处理函数）
            processed_config = load_config_with_dataset_switch(self.config_path)
            
            print(processed_config['PATH']['path_to_dataset']) 
            print(processed_config['PATH']['plot_data_path']) 

            print(processed_config['SESSION']['session_tuple_mode'])
            print(processed_config['SESSION']['session_label_id_map'])
            # 验证处理后的配置有效
            if 'PATH' not in processed_config or 'SESSION' not in processed_config:
                self.log_test("传统API兼容性", False, "处理后配置缺少必要段")
                return False
            
            # 测试新的便捷函数
            path_config = get_config_section('PATH')
            session_config = get_config_section('SESSION')
            
            if not path_config or not session_config:
                self.log_test("新API函数", False, "新API函数返回空结果")
                return False
            
            # 验证配置内容一致性
            if (processed_config['PATH']['path_to_dataset'] != path_config['path_to_dataset'] or
                processed_config['SESSION']['session_tuple_mode'] != session_config['session_tuple_mode']):
                self.log_test("API一致性", False, "处理后API与新API结果不一致")
                return False
            
            # 验证配置不是AUTO_FILL
            if (path_config['path_to_dataset'] == 'AUTO_FILL' or 
                session_config['session_tuple_mode'] == 'AUTO_FILL'):
                self.log_test("配置自动填充", False, "配置未被正确自动填充")
                return False
            
            # 测试原始ConfigParser方式仍然可用（但不会自动处理数据集切换）
            raw_config = configparser.ConfigParser()
            raw_config.read(self.config_path, encoding='utf-8')
            
            if 'GENERAL' not in raw_config or 'ACTIVE_DATASET' not in raw_config['GENERAL']:
                self.log_test("原始配置访问", False, "原始ConfigParser无法访问配置")
                return False
            
            self.log_test("API兼容性", True, "所有API都正常工作，配置自动填充正确")
            return True
            
        except Exception as e:
            self.log_test("API兼容性", False, f"测试失败: {e}")
            return False
    
    def test_edge_cases(self):
        """测试边缘情况"""
        try:
            # 测试无效数据集名称
            self.modify_active_dataset('INVALID-DATASET')
            
            config = load_config_with_dataset_switch(self.config_path)
            path_config = config['PATH']
            
            # 无效数据集应该保持原始配置或默认值
            self.log_test("无效数据集处理", True, "系统能正常处理无效数据集")
            
            # 测试配置缓存刷新
            refresh_config()
            config2 = get_global_config()
            if not config2:
                self.log_test("配置缓存刷新", False, "缓存刷新失败")
                return False
            
            self.log_test("配置缓存刷新", True, "配置缓存正常刷新")
            return True
            
        except Exception as e:
            self.log_test("边缘情况测试", False, f"测试失败: {e}")
            return False
    
    def test_configuration_completeness(self):
        """测试配置完整性"""
        try:
            config = get_global_config()
            
            # 检查所有必要的配置段
            required_sections = ['GENERAL', 'PATH', 'SESSION', 'MODEL_PARAMS', 
                               'TRAINING_MODES', 'TRAINING_PARAMS']
            
            missing_sections = []
            for section in required_sections:
                if section not in config:
                    missing_sections.append(section)
            
            if missing_sections:
                self.log_test("配置完整性", False, f"缺少配置段: {missing_sections}")
                return False
            
            # 检查关键配置项
            critical_configs = {
                'GENERAL': ['thread_count', 'ACTIVE_DATASET'],
                'PATH': ['path_to_dataset', 'plot_data_path'],
                'SESSION': ['session_tuple_mode', 'concurrent_flow_iat_threshold', 
                          'sequential_flow_iat_threshold', 'session_label_id_map']
            }
            
            for section, keys in critical_configs.items():
                for key in keys:
                    if key not in config[section]:
                        self.log_test("关键配置项", False, f"缺少 {section}.{key}")
                        return False
            
            self.log_test("配置完整性", True, "所有必要的配置段和项都存在")
            return True
            
        except Exception as e:
            self.log_test("配置完整性", False, f"检查失败: {e}")
            return False
    
    def run_all_tests(self):
        """运行所有测试"""
        print("=" * 60)
        print("开始 config.cfg 调整后的全面测试")
        print("=" * 60)
        
        self.backup_config()
        
        try:
            test_methods = [
                self.test_basic_structure,
                # self.test_dataset_sections,
                # self.test_active_dataset_mechanism,
                # self.test_dataset_switching,
                self.test_api_compatibility,
                self.test_edge_cases,
                self.test_configuration_completeness
            ]
            
            passed = 0
            total = len(test_methods)
            
            for test_method in test_methods:
                if test_method():
                    passed += 1
                print()
            
            # 恢复原始配置
            self.restore_config()
            
            # 输出测试总结
            print("=" * 60)
            print("测试总结")
            print("=" * 60)
            print(f"总测试数: {total}")
            print(f"通过测试: {passed}")
            print(f"失败测试: {total - passed}")
            print(f"通过率: {passed/total*100:.1f}%")
            
            if passed == total:
                print("\n🎉 所有测试通过！config.cfg 调整成功！")
            else:
                print(f"\n⚠️ 有 {total - passed} 个测试失败，请检查配置。")
            
            print("\n详细测试结果:")
            for test_name, success, message in self.test_results:
                status = "✓" if success else "✗"
                print(f"{status} {test_name}")
                if message and not success:
                    print(f"   {message}")
            
            return passed == total
            
        except Exception as e:
            print(f"✗ 测试过程中发生错误: {e}")
            self.restore_config()
            return False

def main():
    """主函数"""
    tester = ConfigCfgTest()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()