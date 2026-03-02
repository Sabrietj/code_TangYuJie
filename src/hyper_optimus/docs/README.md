# 消融实验框架

统一的消融实验执行框架，支持多视图网络流量分类模型的系统性消融研究。

## 🚀 功能特性

- **统一配置管理**: 基于YAML的实验配置，支持特征消融、融合消融、损失消融
- **智能配置转换**: 动态参数覆盖，支持参数删除和动态覆盖，支持外部配置映射文件
- **W&B集成**: 完整的实验跟踪，支持每个epoch的实时指标采集和SHAP分析结果上传
- **多种执行模式**: 支持串行/并行执行、标准实验模式和消融实验模式
- **结果生成器**: 独立的结果生成和解析工具，支持从已有日志生成final_results.json
- **完善的CLI工具**: 统一命令行接口，支持生成、上传、运行等多种操作
- **自动化报告**: 自动生成实验报告和结果汇总
- **错误恢复**: 完善的错误处理和恢复机制
- **变体标识系统**: 智能的消融变体ID生成和解析
- **模块化设计**: 清晰的模块分离，便于扩展和维护

## 📁 目录结构

```
src/hyper_optimus/
├── configs/                 # 配置文件目录
├── docs/                   # 文档目录  
├── experiment/              # 实验执行核心
│   ├── __main__.py        # 模块主入口
│   ├── cli_tools.py        # 统一CLI工具接口
│   ├── experiment_executor.py  # 实验执行器
│   ├── config_converter.py     # 配置转换器
│   ├── wandb_integration.py    # W&B集成
│   ├── result_generator.py     # 结果生成器
│   ├── variant_identifier.py    # 变体标识系统
│   └── run_ablation_exp.py   # 主执行脚本
└── shap_analysis/          # SHAP分析模块
    ├── universal_analyzer.py    # 通用SHAP分析器
    ├── analysis_strategies.py  # 分析策略
    ├── feature_classifier.py   # 特征分类器
    └── ...
```
### 完整数据流程

  输入配置文件 (exp_config.yaml)
         ↓
   配置验证和转换
         ↓  
   实验变体解析 (VariantIdentifier)
         ↓
   训练命令构建 (ConfigConverter)
         ↓
   模型训练执行 (ExperimentExecutor)
         ↓
   实时指标上传 (WandBIntegration) ←─┐
         ↓                          │
   日志解析 (ResultGenerator)        │
         ↓                          │
   SHAP分析 (UniversalSHAPAnalyzer) ─┘
         ↓
   结果汇总和报告生成
         ↓
   最终输出 (JSON + Markdown + W&B)


## 🛠️ 安装依赖

```bash
pip install pyyaml wandb torch pytorch-lightning transformers hydra omegaconf psutil GPUtil tensorboard
```

### 可选依赖

```bash
# GPU监控
pip install GPUtil

# TensorBoard日志解析
pip install tensorboard
```

## 📖 使用方法

### Python 虚拟化环境

cd /data/qinyf/code-multiview-network-traffic-classification-model/
source myvenv/bin/activate

### 运行实验

```bash
# 仅验证配置，不执行实验(指定配置文件exp_config.yaml)
python src/hyper_optimus/experiment/run_ablation_exp.py --config src/hyper_optimus/configs/exp_config.yaml --dry-run

# 并行执行（2个实验同时运行） 默认单个实验运行
python src/hyper_optimus/experiment/run_ablation_exp.py --config src/hyper_optimus/configs/exp_config.yaml --parallel 2

# 运行实验并指定输出目录，默认输出到根目录 ./ablation_results
python src/hyper_optimus/experiment/run_ablation_exp.py --config src/hyper_optimus/configs/exp_config.yaml --output-dir ./my_results

# 检查依赖包
python src/hyper_optimus/experiment/run_ablation_exp.py --check-deps

# 自定义W&B项目
python src/hyper_optimus/experiment/run_ablation_exp.py --config src/hyper_optimus/configs/exp_config.yaml --wandb-project my-project --wandb-entity my-entity

# 设置日志级别和日志文件
python src/hyper_optimus/experiment/run_ablation_exp.py --config src/hyper_optimus/configs/exp_config.yaml --log-level DEBUG --log-file experiment.log

# 启用批量W&B上传
python src/hyper_optimus/experiment/run_ablation_exp.py --config src/hyper_optimus/configs/exp_config.yaml --batch-wandb-upload

# 启用延迟报告生成
python src/hyper_optimus/experiment/run_ablation_exp.py --config src/hyper_optimus/configs/exp_config.yaml --enable-delayed-report
```

### 实验运行结果处理
# 手动生成最终结果文件(final_results.json， experiment_report.md)
# 手动上传W&B

```bash
# 生成最新实验的结果文件(ablation_results)
python -m src.hyper_optimus.experiment generate --latest

# 生成指定目录的结果文件
python -m src.hyper_optimus.experiment generate --results-dir ablation_results/suite_20251127_123456

# 仅验证目录结构
python -m src.hyper_optimus.experiment generate --results-dir suite_XXX --validate-only

# 上传最新实验结果到W&B
python -m src.hyper_optimus.experiment upload --latest

# 上传指定目录结果
python -m src.hyper_optimus.experiment upload --results-dir ablation_results/suite_20251127_123456

# Dry-run模式验证文件格式
python -m src.hyper_optimus.experiment upload --results-file final_results.json --dry-run

# 指定W&B项目
python -m src.hyper_optimus.experiment run --config src/hyper_optimus/configs/exp_config.yaml --wandb-project my-project --wandb-entity my-entity

# 查看帮助
python -m src.hyper_optimus.experiment --help
```


## ⚙️ 配置文件格式

### 基础配置结构

###  基于hydra 的 模型配置文件体系 

# 模型主函数增加装饰器,通过hydra.main 装饰器传入配置文件路径和名称
@hydra.main(config_path="./config", config_name="flow_bert_multiview_config", version_base=None)

# 实验配置文件
通过命令行参数，传递需要覆盖修改的配置参数，以满足自动超参搜索，消融实验的需要。
但运行这些实验的基础配置参数，还是使用模型自己定义的配置文件，一般实在模型的config文件夹下。

- 实验用的配置文件的格式：

```yaml
# 通用配置，会覆盖模型配置文件中的同名参数
data:
  data_path: "/path/to/dataset.csv"
  batch_size: 1024
  num_workers: 8

training:
  max_epochs: 50
  patience: 8

# 实验配置
experiment:
  model_name: flow_bert_multiview        # 模型名称, 到对应的模型目录下(models),找到对应的config.yaml,数据处理脚本,训练脚本(train.py)
  ablation_strategy: "full"               # 消融策略: standard(仅基线), full(全部使能实验), selective(选择性)
  enable:                                 # 实验配置, 实验类型使能开关
    feature_ablation: true                # 实验配置, 是否开启特征消融 
    fusion_ablation: true                 # 实验配置, 是否开启融合消融
    loss_ablation:   true                 # 实验配置, 是否开启损失消融
```

### 消融实验变体

#### 特征消融 (feature_ablation)

```yaml
ablation_variants:
  FT1
    name: "流基线模型"
    description: "数值特征配置"
    type: "feature_ablation"
    section: "data"
    config:
      enabled_features:
        sequence_features: false
        domain_name_embedding_features: false
        text_features: false
    baseline: true
```

#### 融合消融 (fusion_ablation)

```yaml
  FU1
    name: "拼接"
    description: "多视图拼接融合"
    type: "fusion_ablation"
    section: "fusion"
    method: "concat"  # concat, cross_attention, weighted_sum
```

#### 损失消融 (loss_ablation)

```yaml
  LS1
    name: "预测损失"
    description: "仅预测损失"
    type: "loss_ablation"
    model_name: "flow_bert_multiview_ssl"
```

## 🔄 配置转换机制

### 外部配置映射文件

框架使用 `config_mapping.yaml` 文件定义配置映射关系（实验配置->模型配置）：

```yaml
# 支持模型特定的映射
data:
  data_path:
    flow_bert_multiview: "data.flow_data_path"
    flow_autoencoder: "data.data_path"
    default: "data.data_path"

training:
  max_epochs:
    default: "training.max_epochs"
  
  patience:
    flow_bert_multiview: "training.early_stopping.patience"
    default: "training.patience"

multiview:
  sequence_features:
    default: "data.sequence_features"
  text_features:
    default: "data.text_features"
```

### 参数覆盖路径()
## 注意： 如果新增加模型，尽量确保模型配置文件中，参数定义的稳定，要不然会导致映射关系不稳定，需要手动维护映射关系。
```python
# 支持多种配置路径映射
'enabled_features.text_features' -> 'data.text_features.enabled'
'enabled_features.sequence_features' -> 'data.sequence_features.enabled'
'fusion.method' -> 'model.fusion.method'
'training.max_epochs' -> 'training.max_epochs' (flow_bert_multiview)
'training.max_epochs' -> 'training.early_stopping.patience' (flow_autoencoder)
```


## 📊 W&B集成

### 实时指标采集

每个训练epoch的指标都会实时上传到W&B：

- **训练指标**: `train/loss`, `train/accuracy`, `train/f1_score`, `train/precision`, `train/recall`, `train/learning_rate`
- **验证指标**: `val/loss`, `val/accuracy`, `val/f1_score`, `val/precision`, `val/recall`, `val/auc`
- **测试指标**: `test/loss`, `test/accuracy`, `test/f1_score`, `test_precision`, `test_recall`, `test_roc_auc`
- **系统资源**: `system/cpu_percent`, `system/memory_percent`, `system/gpu_utilization`
- **SHAP分析**: `shap/feature_importance`, `shap/summary_plot`
- **实验上下文**: `experiment_name`, `model_type`, `ablation_config`, `variant_id`

### 实验组织

```
W&B Project: multiview-ablation-studies
├── suite_20251127_143022_FT1_baseline_143025/
├── suite_20251127_143022_FT2_text_features_143045/
└── suite_20251127_143022_FU1_concat_fusion_143108/
```

### 标签系统

- `feature_ablation`: 特征消融实验
- `fusion_ablation`: 融合消融实验
- `loss_ablation`: 损失消融实验
- `baseline`: 基线实验
- `{model_name}`: 模型类型
- `{variant_id}`: 变体标识符

### SHAP结果上传

框架自动处理SHAP分析结果并上传到W&B：

```python
# 自动上传SHAP图像
wandb_run.log({
    "shap/summary_plot": wandb.Image("shap_results/summary_plot.png"),
    "shap/feature_importance": wandb.Image("shap_results/feature_importance.png")
})

# 自动上传SHAP数据
with open("shap_results/shap_values.json", 'r') as f:
    shap_data = json.load(f)
    wandb_run.log({"shap/values": shap_data})
```

## 📈 输出结果

### 实验输出目录

```
ablation_results/
├── suite_20251127_143022/
│   ├── FT1_baseline/
│   │   ├── .hydra/
│   │   ├── training.log
│   │   ├── shap_results/
│   │   │   ├── summary_plot.png
│   │   │   └── shap_values.json
│   │   └── tensorboard/
│   ├── FU1_concat_fusion/
│   ├── LS1_prediction_loss/
│   ├── experiment_config.yaml
│   ├── experiment_report.md   # 实验报告,包含实验ID,状态,最终准确率,最终F1分数,训练时间
│   ├── intermediate_results.json
│   └── final_results.json     # 最终结果,epoch_results, test_results, shap_results
```

### 自动生成报告

实验完成后自动生成Markdown报告：

```markdown
# 消融实验报告

**生成时间**: 2025-11-27 14:30:22
**实验总数**: 3
**成功实验**: 3
**失败实验**: 0

## 实验结果汇总

| 实验ID             | 状态          | 最终准确率    | 最终F1分数    | 训练时间  |
|-------------------|--------------|--------------|--------------|----------|
| FT1               | ✅ completed | 0.8234       | 0.8012       | 45.2s    |
| FT2               | ✅ completed | 0.8567       | 0.8423       | 52.1s    |
| FU1               | ✅ completed | 0.8345       | 0.8198       | 48.7s    |
```

### 解析的指标类型

- **训练指标**: 训练损失、准确率、精确率、召回率、F1分数
- **验证指标**: 验证损失、准确率、精确率、召回率、F1分数、AUC
- **测试指标**: 测试准确率、F1分数、精确率、召回率、ROC-AUC
- **系统指标**: CPU使用率、内存使用、GPU利用率、训练时长
- **业务指标**: 吞吐量、延迟、推理时间、模型大小
- **SHAP分析**: 特征重要性、可视化图表、解释性数据


```

## 🔧 故障排除

### 常见问题

#### 1. 配置验证失败

```
Configuration validation failed:
  - Missing required field: experiment.model_name
  - Type mismatch at data.batch_size: expected int, got str
```

**解决方案**: 
- 检查配置文件格式，确保必需字段存在且类型正确
- 验证 `config_mapping.yaml` 中的映射关系是否正确
- 使用 `--dry-run` 参数仅验证配置不执行实验

#### 2. 依赖包缺失

```
Missing required packages: wandb, pytorch_lightning
```

**解决方案**: 
```bash
pip install wandb pytorch_lightning GPUtil tensorboard
```

#### 3. W&B连接失败

```
W&B integration error: Failed to initialize wandb run
```

**解决方案**: 
```bash
wandb login
# 或设置环境变量
export WANDB_API_KEY="your-api-key"
```

#### 4. 训练脚本未找到

```
ModelNotFoundError: No training script found for model: unknown_model
```

**解决方案**: 
- 确保模型名称在`experiment_executor.py`的`model_script_mapping`中存在
- 检查训练脚本路径是否正确且文件存在

#### 5. SHAP分析失败

```
SHAP分析目录不存在: /path/to/output/shap_results
```

**解决方案**: 
- 确保模型配置中启用了SHAP分析
- 检查输出权限和磁盘空间
- 查看训练日志中的SHAP相关错误信息

#### 6. 结果解析失败

```
解析实验结果失败: 无法解析training.log
```

**解决方案**: 
- 检查训练日志文件是否存在且可读
- 验证日志格式是否符合框架期望的格式
- 使用 `ResultGenerator` 类的验证功能

#### 7. 并行执行冲突

```
并行执行时资源冲突: GPU内存不足
```

**解决方案**: 
- 减少并行作业数量
- 检查系统资源（GPU内存、磁盘空间）
- 使用串行模式 `--parallel 1`

### 调试模式

启用详细日志输出：

```bash
# 传统方式
python src/hyper_optimus/experiment/run_ablation_exp.py --config exp_config.yaml --debug --log-file debug.log

# CLI方式
python -m src.hyper_optimus.experiment run --config exp_config.yaml --log-level DEBUG
```

### 配置验证

仅验证配置不执行实验：

```bash
# 验证配置文件
python src/hyper_optimus/experiment/run_ablation_exp.py --config exp_config.yaml --dry-run

# 验证实验目录
python -m src.hyper_optimus.experiment generate --results-dir suite_XXX --validate-only

# 验证结果文件格式
python -m src.hyper_optimus.experiment upload --results-file final_results.json --dry-run
```

### 日志分析

```python
from src.hyper_optimus.experiment import ExperimentExecutor

executor = ExperimentExecutor(workspace_root='/path/to/project')
parsed_data = executor._parse_log_files('/path/to/experiment/output')

print(f"解析到 {len(parsed_data['epoch_metrics'])} 个epoch指标")
print(f"测试结果: {parsed_data['test_results']}")
print(f"训练时长: {parsed_data['duration']}秒")
```

## 📚 扩展开发

### 添加新模型支持

1. 在`experiment_executor.py`中添加模型映射：

```python
self.model_script_mapping = {
    'your_new_model': 'path/to/your/model/train.py',
    # ... 其他模型
}
```

2. 在`config_mapping.yaml`中添加配置映射：

```yaml
data:
  data_path:
    your_new_model: "data.custom_data_path"
  batch_size:
    your_new_model: "data.batch_size"

training:
  max_epochs:
    your_new_model: "training.max_epochs"
  learning_rate:
    your_new_model: "optimizer.lr"

# 其他配置段...
```

3. 在`config_converter.py`中更新映射逻辑（如果需要特殊处理）

### 自定义指标采集

扩展日志解析模式：

```python
# 在 ExperimentExecutor._parse_epoch_patterns() 中添加新模式
def _parse_epoch_patterns(self, content: str, epoch_dict: dict):
    epoch_patterns = {
        # 现有模式...
        'custom_format': r'Epoch\s+(\d+).*custom_metric[=\s]\s*([\d.]+)',
    }
    # 解析逻辑...
```

### 自定义W&B集成

继承W&B集成类：

```python
from src.hyper_optimus.experiment import WandBIntegration

class CustomWandBIntegration(WandBIntegration):
    def init_experiment_run(self, experiment_name, experiment_config, variant_id, ablation_variant):
        # 自定义初始化逻辑
        run = super().init_experiment_run(experiment_name, experiment_config, variant_id, ablation_variant)
        
        # 添加自定义标签或配置
        run.tags.extend(['custom_tag'])
        
        return run
```

### 自定义配置转换

继承配置转换器：

```python
from src.hyper_optimus.experiment import AblationConfigConverter

class CustomConfigConverter(AblationConfigConverter):
    def convert_custom_ablation(self, ablation_config, model_name):
        """自定义消融类型转换"""
        override_config = {}
        # 自定义转换逻辑
        return override_config
    
    def convert_ablation_config(self, ablation_variant, model_name):
        override_config = super().convert_ablation_config(ablation_variant, model_name)
        
        if ablation_variant.get('type') == 'custom_ablation':
            custom_config = self.convert_custom_ablation(ablation_variant.get('config', {}), model_name)
            override_config.update(custom_config)
        
        return override_config
```

### 添加新的消融类型

1. 在`config_mapping.yaml`中添加新的配置段

2. 在`variant_identifier.py`中添加新的类型标识：

```python
class VariantIdentifier:
    def __init__(self):
        self.type_prefixes = {
            # 现有类型...
            'custom_ablation': 'CU'
        }
```

3. 更新配置转换器支持新类型

## 🎯 最佳实践

### 1. 实验设计

```yaml
# 推荐的实验配置结构
experiment:
  model_name: flow_bert_multiview
  ablation_strategy: ablation  # 或 'standard'
  enable:
    feature_ablation: true
    fusion_ablation: true
    loss_ablation: false

# 明确定义基线实验
ablation_variants:
  BASE:
    name: "基线模型"
    type: "feature_ablation"
    baseline: true
    config:
      enabled_features:
        sequence_features: true
        text_features: true
```

### 2. 资源管理

```python
# 合理设置并行数量
executor = BatchExperimentExecutor(
    workspace_root='/path/to/project',
    max_parallel_jobs=min(4, psutil.cpu_count() // 2)  # 根据CPU资源动态调整
)
```


