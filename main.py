#!/usr/bin/env python3

# -*- coding:utf-8 -*-
# Author：Bill Liu
# Create：2025-11-01
# Update：2025-11-01

"""
YOLOv11 RealSense D455 人体检测应用程序 - 主入口文件
"""

import os
from app.instance_segmentation_app import InstanceSegmentationApp

def main():
    """主函数"""
    print("=" * 60)
    print("YOLOv11 RealSense D455 人体检测应用程序 - 无边界框版本")
    print("=" * 60)
    
    # 优先使用小模型
    model_paths = [
        # "./weights/yolo11n-seg.pt",  # Nano版本
        # "./weights/yolo11s-seg.pt",  # Small版本
        # "./weights/yolo11m-seg.pt",  # Medium版本
        # "./weights/yolo11l-seg.pt",  # Large版本
        "./weights/yolo11x-seg.pt",  # XLarge版本
    ]
    
    # 自动检测可用的模型
    selected_model = None
    for model_path in model_paths:
        if os.path.exists(model_path):
            selected_model = model_path
            print(f"找到模型: {model_path}")
            break
    
    if selected_model is None:
        print("警告: 未找到YOLOv11分割模型文件")
        print("请将模型文件放在以下位置之一:")
        for model_path in model_paths:
            print(f"  - {model_path}")
        return
    
    # 创建并运行应用程序
    app = InstanceSegmentationApp(selected_model)
    app.run()

if __name__ == "__main__":
    main()
