# test_fix.py
import os

def check_structure():
    """检查项目结构"""
    required_files = [
        'src/models/flow_bert_multiview/config/flow_bert_multiview_config.yaml',
        'src/models/flow_bert_multiview/data/flow_bert_multiview_dataset.py', 
        'src/models/flow_bert_multiview/models/flow_bert_multiview.py',
        'src/models/flow_bert_multiview/train.py'
    ]
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file} 存在")
        else:
            print(f"✗ {file} 缺失")

if __name__ == "__main__":
    check_structure()