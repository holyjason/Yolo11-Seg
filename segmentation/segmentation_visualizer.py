#!/usr/bin/env python3

# -*- coding:utf-8 -*-
# Author：Bill Liu
# Create：2025-11-01
# Update：2025-11-01
"""
分割结果可视化类
"""

import cv2
import numpy as np

class SegmentationVisualizer:
    """分割结果可视化类 - 无边界框版本"""
    
    def __init__(self):
        # 定义颜色调色板
        self.colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (128, 0, 0), (0, 128, 0), (0, 0, 128),
            (128, 128, 0), (128, 0, 128), (0, 128, 128)
        ]
        
    def draw_segmentation(self, image, masks, boxes, classes, confidences, class_names, 
                         depth_frame=None, fps_info=None):
        """在图像上绘制分割结果，包含深度信息，但不绘制边界框"""
        if image is None:
            return image
            
        result_image = image.copy()
        
        # 绘制帧率信息
        if fps_info:
            # 创建半透明背景
            overlay = result_image.copy()
            cv2.rectangle(overlay, (10, 10), (300, 110), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, result_image, 0.3, 0, result_image)
            
            # 绘制帧率文本
            y_offset = 30
            line_height = 20
            for key, value in fps_info.items():
                text = f"{key}: {value}"
                cv2.putText(result_image, text, (15, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                y_offset += line_height
        
        # 绘制每个检测到的实例
        for i, (mask, box, cls, conf) in enumerate(zip(masks, boxes, classes, confidences)):
            color = self.colors[i % len(self.colors)]
            
            # 获取边界框坐标（用于计算中心点）
            x1, y1, x2, y2 = map(int, box)
            
            # 计算边界框中心点
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # 获取中心点深度
            depth_value = 0.0
            if depth_frame is not None:
                depth_value = depth_frame.get_distance(center_x, center_y)
                if depth_value > 0:
                    # 在中心点绘制深度标记
                    cv2.circle(result_image, (center_x, center_y), 8, color, -1)
                    cv2.putText(result_image, f"{depth_value:.2f}m", 
                               (center_x + 10, center_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # 绘制分割掩码
            if mask is not None:
                # 将掩码调整为图像大小
                mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
                
                # 创建彩色掩码
                colored_mask = np.zeros_like(image)
                colored_mask[mask_resized > 0.5] = color
                
                # 将掩码叠加到原图上
                result_image = cv2.addWeighted(result_image, 0.7, colored_mask, 0.3, 0)
            
            # 添加标签（不绘制边界框，只显示标签）
            class_name = class_names.get(int(cls), f"Class_{int(cls)}")
            if depth_value > 0:
                label = f"{class_name} {conf:.2f} ({depth_value:.2f}m)"
            else:
                label = f"{class_name} {conf:.2f}"
            
            # 计算标签位置（放在检测到的物体上方）
            label_x = max(10, center_x - 50)
            label_y = max(30, center_y - 20)
            
            # 计算标签背景大小
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            # 绘制标签背景
            cv2.rectangle(result_image, 
                         (label_x, label_y - label_height - 5), 
                         (label_x + label_width, label_y), 
                         color, -1)
            
            # 绘制标签文本
            cv2.putText(result_image, label, 
                       (label_x, label_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return result_image
    
    def create_depth_colormap(self, depth_frame):
        """创建深度图的彩色可视化"""
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), 
            cv2.COLORMAP_JET
        )
        
        # 在深度图上添加距离刻度
        height, width = depth_colormap.shape[:2]
        cv2.putText(depth_colormap, "Depth Map", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 添加距离刻度
        for i, dist in enumerate([1, 2, 3, 4, 5]):
            y_pos = int(height * 0.1 * (i + 1))
            cv2.line(depth_colormap, (width - 50, y_pos), (width - 30, y_pos), (255, 255, 255), 2)
            cv2.putText(depth_colormap, f"{dist}m", (width - 45, y_pos + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        return depth_colormap
