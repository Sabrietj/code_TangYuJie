"""
增强版SHAP组件 - 集成五大特征类别分析架构
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import logging
from typing import Dict, List, Any, Optional

from .five_tier_analyzer import FiveTierSHAPAnalyzer
from .multi_model_adapter import MultiModelSHAPAdapter

logger = logging.getLogger(__name__)

class EnhancedShapAnalyzer:
    """
    增强版SHAP分析器，集成五大特征类别分析架构
    解决原有的维度不匹配、特征分类混乱、跨模型兼容性问题
    """
    
    def __init__(self, model, enable_five_tier_analysis: bool = True):
        """
        初始化增强版SHAP分析器
        
        Args:
            model: 要分析的模型实例
            enable_five_tier_analysis: 是否启用五大特征类别分析
        """
        self.model = model
        self.cfg = model.cfg
        self.enable_five_tier_analysis = enable_five_tier_analysis
        
        # 缓冲区设置
        self.buffer = []
        self.sample_limit = 100  # 限制分析样本数，防止OOM
        self.collected_count = 0
        self.enabled = True
        
        # 初始化五大特征类别分析器
        if self.enable_five_tier_analysis:
            adapter = MultiModelSHAPAdapter()
            model_type = adapter.auto_detect_model_type(model)
            
            self.five_tier_analyzer = FiveTierSHAPAnalyzer(
                model=model, 
                model_type=model_type,
                config=self._get_five_tier_config()
            )
            logger.info(f"增强版SHAP分析器初始化完成，模型类型: {model_type}")
        else:
            self.five_tier_analyzer = None
            logger.info("增强版SHAP分析器使用传统模式")
    
    def _get_five_tier_config(self) -> Dict[str, Any]:
        """获取五大特征类别分析配置"""
        return {
            'shap': {
                'enable_level1_analysis': True,   # 启用大类别分析（饼图）
                'enable_level2_analysis': True,   # 启用具体特征分析（柱状图）
                'focus_numeric_features': True,   # 重点分析数值特征
                'dynamic_dimension_calculation': True,  # 动态计算特征维度
                'enable_data_validation': True,
                'save_plots': True,
                'background_samples': 50,
                'eval_samples': 20
            },
            'logging': {
                'save_dir': './shap_results',
                'name': 'enhanced_five_tier_analysis',
                'version': 'default'
            }
        }
    
    def reset(self):
        """每个epoch开始时重置缓冲区"""
        self.buffer = []
        self.collected_count = 0
    
    def collect_batch(self, batch: Dict[str, Any]):
        """收集测试阶段的Batch数据 (CPU缓存)"""
        if not self.enabled or self.collected_count >= self.sample_limit:
            return
        
        # 移动到CPU并分离计算图，只保留数据
        batch_cpu = {}
        batch_size = 0
        
        with torch.no_grad():
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch_cpu[k] = v.detach().cpu()
                    if batch_size == 0: 
                        batch_size = v.shape[0]
                elif isinstance(v, list):
                    batch_cpu[k] = v
                else:
                    batch_cpu[k] = v  # 元数据等
        
        self.buffer.append(batch_cpu)
        self.collected_count += batch_size
    
    def finalize(self):
        """测试结束时执行SHAP分析"""
        # 只在主进程执行
        if hasattr(self.model, 'trainer') and not self.model.trainer.is_global_zero:
            return
        
        if not self.buffer:
            logger.warning("[EnhancedShapAnalyzer] 未收集到样本，跳过分析")
            return
        
        logger.info("=" * 80)
        logger.info(f"🔍 [EnhancedShapAnalyzer] 开始执行增强版特征归因分析 (样本数: {self.collected_count})...")
        
        try:
            if self.enable_five_tier_analysis and self.five_tier_analyzer is not None:
                self._run_five_tier_analysis()
            else:
                self._run_legacy_analysis()
        except Exception as e:
            logger.error(f"❌ [EnhancedShapAnalyzer] 分析失败: {e}", exc_info=True)
        
        logger.info("=" * 80)
    
    def _run_five_tier_analysis(self):
        """执行五大特征类别分析"""
        logger.info("🎯 启用五大特征类别分析架构...")
        
        # ======================================================================
        # 🔴 终极修复: 退出 Inference Mode + 开启 Grad
        # PyTorch Lightning 的 test 阶段默认处于 inference_mode (比 no_grad 更强)
        # 必须显式退出 inference_mode 才能构建计算图
        # ======================================================================
        with torch.inference_mode(False):  # 1. 退出推理模式
            with torch.enable_grad():      # 2. 开启梯度计算
                
                # 3. 再次确保模型参数允许求导 (双重保险)
                for param in self.model.parameters():
                    param.requires_grad = True
                
                try:
                    # 1. 合并缓冲区数据
                    combined_batch = self._merge_buffer()
                    
                    # 验证计算图状态
                    logger.info(f"[EnhancedShapAnalyzer] 计算图状态检查:")
                    logger.info(f"  is_grad_enabled: {torch.is_grad_enabled()}")
                    logger.info(f"  is_inference_mode: {torch.is_inference_mode_enabled()}")
                    
                    # 2. 执行五大特征类别分析
                    results = self.five_tier_analyzer.analyze(combined_batch)
                    
                    # 3. 保存结果
                    output_path = self.five_tier_analyzer.save_results(
                        results, 
                        experiment_name='enhanced_five_tier_analysis'
                    )
                    
                    # 4. 输出摘要
                    self._print_five_tier_summary(results)
                    
                    logger.info(f"✅ [EnhancedShapAnalyzer] 五大特征类别分析完成，结果保存至: {output_path}")
                    
                except Exception as e:
                    logger.error(f"五大特征类别分析过程中发生错误: {e}", exc_info=True)
                    raise
    
    def _run_legacy_analysis(self):
        """执行传统SHAP分析（兼容性）"""
        logger.info("🔄 使用传统SHAP分析模式...")
        
        # 这里可以集成原来的ShapComponent逻辑
        # 为了向后兼容，保留原有分析方式
        from .shap_component import ShapAnalyzer
        
        # 创建传统分析器实例
        legacy_analyzer = ShapAnalyzer(self.model)
        legacy_analyzer.buffer = self.buffer
        legacy_analyzer.collected_count = self.collected_count
        
        # 执行传统分析
        legacy_analyzer.finalize()
    
    def _merge_buffer(self) -> Dict[str, Any]:
        """合并缓冲区数据"""
        if not self.buffer:
            return {}
        
        combined = {}
        keys = self.buffer[0].keys()
        
        for k in keys:
            val = self.buffer[0][k]
            if isinstance(val, torch.Tensor):
                combined[k] = torch.cat([b[k] for b in self.buffer], dim=0)
            elif isinstance(val, list):
                combined[k] = [item for b in self.buffer for item in b[k]]
        
        return combined
    
    def _print_five_tier_summary(self, results: Dict[str, Any]):
        """打印五大特征类别分析摘要"""
        summary = results['analysis_summary']
        five_tier = results['five_tier_analysis']
        
        logger.info("\n" + "=" * 60)
        logger.info("📊 五大特征类别SHAP分析报告")
        logger.info("=" * 60)
        
        logger.info(f"模型类型: {summary['model_type']}")
        logger.info(f"维度兼容性: {summary['dimension_compatibility']}")
        logger.info(f"分析时间: {summary['analysis_timestamp']}")
        logger.info(f"SHAP方法: {summary['shap_method']}")
        
        logger.info("\n🎯 Level 1: 大类别重要性分布")
        level1_importance = five_tier['level1_category_importance']
        for category, importance in sorted(level1_importance.items(), key=lambda x: x[1], reverse=True):
            if importance > 0.1:  # 只显示大于0.1%的类别
                logger.info(f"  {category.replace('_', ' ').title()}: {importance:.2f}%")
        
        logger.info("\n📈 Level 2: Top 10 数值特征")
        level2_importance = five_tier['level2_numeric_importance']
        for i, (feature, importance) in enumerate(list(level2_importance.items())[:10], 1):
            logger.info(f"  {i:2d}. {feature}: {importance:.4f}")
        
        logger.info("\n🔧 特征分类统计")
        feature_classification = five_tier['feature_classification']
        for category, features in feature_classification.items():
            logger.info(f"  {category}: {len(features)} 个特征")
        
        logger.info("=" * 60)
    
    def get_analysis_capabilities(self) -> Dict[str, Any]:
        """获取分析能力信息"""
        capabilities = {
            'supports_five_tier_analysis': self.enable_five_tier_analysis,
            'model_type': None,
            'feature_classification': None,
            'dimension_validation': None,
            'supported_visualizations': []
        }
        
        if self.enable_five_tier_analysis and self.five_tier_analyzer is not None:
            adapter = MultiModelSHAPAdapter()
            capabilities['model_type'] = adapter.auto_detect_model_type(self.model)
            
            # 获取特征分类能力
            if hasattr(self.five_tier_analyzer.feature_classifier, 'get_feature_hierarchy_info'):
                hierarchy_info = self.five_tier_analyzer.feature_classifier.get_feature_hierarchy_info()
                capabilities['feature_classification'] = hierarchy_info
            
            # 获取维度验证能力
            if hasattr(self.five_tier_analyzer.dimension_extractor, 'calculate_all_dimensions'):
                dimensions = self.five_tier_analyzer.dimension_extractor.calculate_all_dimensions()
                capabilities['dimension_validation'] = {
                    'supported_dimensions': list(dimensions.keys()),
                    'total_input_dims': dimensions.get('total_input_dims', 0)
                }
            
            # 获取可视化支持
            if self.five_tier_analyzer.config['shap']['enable_level1_analysis']:
                capabilities['supported_visualizations'].append('pie_chart')
            if self.five_tier_analyzer.config['shap']['enable_level2_analysis']:
                capabilities['supported_visualizations'].append('bar_chart')
        
        return capabilities
    
    def enable_five_tier_mode(self, enable: bool = True):
        """动态启用/禁用五大特征类别分析模式"""
        self.enable_five_tier_analysis = enable
        
        if enable and self.five_tier_analyzer is None:
            adapter = MultiModelSHAPAdapter()
            model_type = adapter.auto_detect_model_type(self.model)
            
            self.five_tier_analyzer = FiveTierSHAPAnalyzer(
                model=self.model, 
                model_type=model_type,
                config=self._get_five_tier_config()
            )
            logger.info(f"五大特征类别分析模式已启用，模型类型: {model_type}")
        elif not enable:
            logger.info("五大特征类别分析模式已禁用，将使用传统分析模式")
    
    def update_config(self, new_config: Dict[str, Any]):
        """更新分析配置"""
        if self.five_tier_analyzer is not None:
            # 合并配置
            current_config = self.five_tier_analyzer.config
            for key, value in new_config.items():
                if key in current_config:
                    if isinstance(current_config[key], dict) and isinstance(value, dict):
                        current_config[key].update(value)
                    else:
                        current_config[key] = value
                else:
                    current_config[key] = value
            
            logger.info("增强版SHAP分析器配置已更新")
    
    def get_recent_results(self) -> Optional[Dict[str, Any]]:
        """获取最近的分析结果"""
        return getattr(self, 'analysis_results', None)