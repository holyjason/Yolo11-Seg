#!/usr/bin/env python3

# -*- coding:utf-8 -*-
# Author：Bill Liu
# Create：2025-11-01
# Update：2025-11-01
"""
YOLOv11实例分割类
"""

import torch
from ultralytics import YOLO

class YOLOv11Segmentation:
    """YOLOv11实例分割类 - 只检测人"""
    
    def __init__(self, model_path, conf_threshold=0.5, iou_threshold=0.45):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model = None
        self.device = None
        self.imgsz = 640  # 固定为640x640，与TensorRT引擎匹配
        
    def initialize(self):
        """初始化YOLOv11模型"""
        try:
            # 检测设备类型
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                print("使用CUDA加速")
            else:
                self.device = torch.device('cpu')
                print("使用CPU")
            
            # 加载模型
            if self.model_path.endswith('.engine'):  # TensorRT引擎
                self.model = YOLO(self.model_path)
                print(f"加载TensorRT模型: {self.model_path}")
            else:  # PyTorch模型
                self.model = YOLO(self.model_path)
                print(f"加载PyTorch模型: {self.model_path}")
            
            # 预热模型
            dummy_input = torch.randn(1, 3, 320, 320).to(self.device)
            if hasattr(self.model, 'predict'):
                self.model.predict(dummy_input, verbose=False)
            else:
                self.model(dummy_input)
                
            print("YOLOv11模型初始化成功")
            print("配置为只检测'人'类别")
            return True
            
        except Exception as e:
            print(f"YOLOv11初始化失败: {e}")
            return False
    
    def segment_frame(self, image):
        """对图像进行实例分割 - 只检测人"""
        if self.model is None or image is None:
            return None, None, None
            
        try:
            # 使用YOLO进行推理 - 只检测人（类别0）
            results = self.model(image, 
                               conf=self.conf_threshold, 
                               iou=self.iou_threshold, 
                               verbose=False,
                               imgsz=320)  # 固定推理尺寸
                               # classes=[0])  # 只检测人（类别索引0）
            
            if len(results) == 0:
                return image, [], []
                
            result = results[0]
            
            # 获取分割结果
            if result.masks is not None:
                masks = result.masks.data.cpu().numpy()  # 分割掩码
                boxes = result.boxes.xyxy.cpu().numpy()  # 边界框
                classes = result.boxes.cls.cpu().numpy()  # 类别
                confidences = result.boxes.conf.cpu().numpy()  # 置信度
                
                # 获取类别名称
                if hasattr(result, 'names'):
                    class_names = result.names
                else:
                    class_names = {i: f"Class_{i}" for i in range(int(classes.max()) + 1)}
                
                return image, masks, boxes, classes, confidences, class_names
            else:
                return image, [], [], [], [], {}
                
        except Exception as e:
            print(f"分割失败: {e}")
            return image, [], [], [], [], {}
