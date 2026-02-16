#!/usr/bin/env python3
"""
多视图流量分类模型 SHAP 分析示例
支持 flow_bert_multiview, ssl, ssl_mlm, ssl_seq2stat 模型

使用方法:
python multiview_shap_example.py --model_type ssl --batch_data_path batch_data.pkl
"""

import argparse
import pickle
import logging
from pathlib import Path
from typing import Dict, Any

# 导入SHAP分析框架
from universal_analyzer import UniversalSHAPAnalyzer

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(model_type: str) -> Dict[str, Any]:
    """加载模型特定配置"""
    config_path = Path(__file__).parent / "multiview_config.yaml"
    
    # 这里应该解析YAML文件，简化起见返回基础配置
    base_config = {
        'shap': {
            'enabled': True,
            'num_background_samples': 50,
            'max_evals': 500,
            'save_plots': True,
            'save_detailed_results': True,
            'enable_data_validation': True,
            'enable_score_rebalancing': True,
            'enabled_strategies': ['numeric', 'sequence', 'text', 'categorical']
        },
        'output': {
            'base_dir': './shap_results',
            'create_timestamp_dirs': True
        }
    }
    
    # 根据模型类型添加特定配置
    model_configs = {
        'flow_bert_multiview': {
            'decompose_combined_text': True,
            'exclude_features': ['idx', 'uid', 'sequence_mask', 'combined_text']
        },
        'ssl': {
            'decompose_combined_text': True,
            'exclude_features': ['idx', 'uid', 'sequence_mask', 'ssl_mask', 'combined_text']
        },
        'ssl_mlm': {
            'decompose_combined_text': True,
            'exclude_features': ['idx', 'uid', 'sequence_mask', 'mlm_mask', 'sequence_mlm_mask', 'combined_text']
        },
        'ssl_seq2stat': {
            'decompose_combined_text': True,
            'exclude_features': ['idx', 'uid', 'sequence_mask', 'seq2stat_mask', 'combined_text']
        }
    }
    
    if model_type in model_configs:
        base_config.update(model_configs[model_type])
    
    return base_config

def load_batch_data(data_path: str) -> Dict[str, Any]:
    """加载批次数据"""
    try:
        with open(data_path, 'rb') as f:
            batch_data = pickle.load(f)
        logger.info(f"成功加载批次数据，包含 {len(batch_data)} 个字段")
        return batch_data
    except Exception as e:
        logger.error(f"加载批次数据失败: {e}")
        raise

def create_mock_model(model_type: str):
    """创建模拟模型用于演示"""
    # 这里应该加载实际训练好的模型
    # 为了演示，创建一个模拟对象
    class MockModel:
        def __init__(self, model_type):
            self.model_type = model_type
            self.device = 'cpu'
        
        def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
            # 模拟前向传播
            batch_size = batch.get('numeric_features', batch.get('uid', [0])).__class__.__len__([0])
            return {
                'classification_logits': [[0.3, 0.7] for _ in range(batch_size)],
                'multiview_embeddings': [[0.1, 0.2, 0.3] for _ in range(batch_size)]
            }
        
        def eval(self):
            pass
    
    return MockModel(model_type)

def run_shap_analysis(model_type: str, batch_data_path: str):
    """运行SHAP分析"""
    logger.info(f"开始为模型 {model_type} 运行SHAP分析...")
    
    # 1. 加载配置
    config = load_config(model_type)
    logger.info("配置加载完成")
    
    # 2. 加载批次数据
    batch_data = load_batch_data(batch_data_path)
    
    # 3. 加载模型
    model = create_mock_model(model_type)
    logger.info(f"模型 {model_type} 加载完成")
    
    # 4. 创建SHAP分析器
    analyzer = UniversalSHAPAnalyzer(config)
    logger.info("SHAP分析器初始化完成")
    
    # 5. 运行分析
    try:
        results = analyzer.analyze(model, batch_data, model_type)
        logger.info("SHAP分析完成")
        
        # 6. 保存结果
        output_path = analyzer.save_results(results, f"{model_type}_shap_analysis")
        logger.info(f"分析结果已保存到: {output_path}")
        
        # 7. 打印摘要
        if 'analysis_summary' in results:
            summary = results['analysis_summary']
            print(f"\n=== {model_type} SHAP分析摘要 ===")
            print(f"总特征数: {summary.get('total_features_analyzed', 0)}")
            print(f"成功分析: {summary.get('successful_analyses', 0)}")
            print(f"失败分析: {summary.get('failed_analyses', 0)}")
            print(f"发现特征类型: {', '.join(summary.get('feature_types_found', []))}")
        
        # 8. 打印重要特征（如果有的话）
        if 'aggregated_importance' in results and 'sorted_importance' in results['aggregated_importance']:
            sorted_importance = results['aggregated_importance']['sorted_importance']
            print(f"\n=== 前10个重要特征 ===")
            for i, (feature_name, importance_data) in enumerate(list(sorted_importance.items())[:10]):
                score = importance_data.get('score', 0)
                feature_type = importance_data.get('type', 'unknown')
                print(f"{i+1:2d}. {feature_name:30s} [分数: {score:.4f}, 类型: {feature_type}]")
        
        return results
        
    except Exception as e:
        logger.error(f"SHAP分析失败: {e}")
        raise

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='多视图流量分类模型 SHAP 分析')
    parser.add_argument(
        '--model_type',
        type=str,
        required=True,
        choices=['flow_bert_multiview', 'ssl', 'ssl_mlm', 'ssl_seq2stat'],
        help='模型类型'
    )
    parser.add_argument(
        '--batch_data_path',
        type=str,
        required=True,
        help='批次数据文件路径'
    )
    parser.add_argument(
        '--config_path',
        type=str,
        help='配置文件路径（可选）'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./shap_results',
        help='输出目录'
    )
    
    args = parser.parse_args()
    
    # 运行SHAP分析
    results = run_shap_analysis(args.model_type, args.batch_data_path)
    
    print("\n✅ SHAP分析完成！")
    print(f"📊 结果已保存到: {args.output_dir}")

if __name__ == "__main__":
    main()