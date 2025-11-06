#!/usr/bin/env python3

# -*- coding:utf-8 -*-
# Author：Bill Liu
# Create：2025-11-01
# Update：2025-11-01
"""
实例分割主应用程序类
"""

import time
import threading
import cv2
import numpy as np
from queue import Queue

from camera.realsense_d455 import RealSenseD455
from segmentation.yolov11_segmentation import YOLOv11Segmentation
from segmentation.segmentation_visualizer import SegmentationVisualizer
from utils.fps_counter import FPSCounter

class InstanceSegmentationApp:
    """实例分割主应用程序 - 无边界框版本"""
    
    def __init__(self, model_path):
        # 使用D455支持的配置
        self.camera = RealSenseD455(width=848, height=480, fps=30)
        self.segmentor = YOLOv11Segmentation(model_path)
        self.visualizer = SegmentationVisualizer()
        self.running = False
        self.frame_queue = Queue(maxsize=1)
        self.result_queue = Queue(maxsize=1)
        
        # 帧率计数器
        self.camera_fps = FPSCounter()
        self.processing_fps = FPSCounter()
        
        # 显示模式
        self.show_depth = True
        self.show_distance = True
        
    def initialize(self):
        """初始化应用程序"""
        print("初始化RealSense D455相机...")
        if not self.camera.initialize():
            return False
            
        print("初始化YOLOv11分割模型...")
        if not self.segmentor.initialize():
            return False
            
        print("应用程序初始化成功")
        print("模式: 只检测人，无边界框")
        return True
    
    def camera_thread(self):
        """相机采集线程"""
        while self.running:
            color_frame, depth_frame = self.camera.get_frames()
            if color_frame is not None:
                self.camera_fps.update()
                
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except:
                        pass
                self.frame_queue.put((color_frame, depth_frame))
            time.sleep(0.001)
    
    def processing_thread(self):
        """处理线程"""
        while self.running:
            try:
                color_frame, depth_frame = self.frame_queue.get(timeout=1.0)
                self.processing_fps.update()
                
                # 进行实例分割 - 只检测人
                result_image, masks, boxes, classes, confidences, class_names = \
                    self.segmentor.segment_frame(color_frame)
                
                # 可视化结果
                if result_image is not None:
                    fps_info = {
                        "Camera FPS": f"{self.camera_fps.get_fps():.1f}",
                        "Processing FPS": f"{self.processing_fps.get_fps():.1f}",
                        "Persons": len(boxes)  # 修改为显示人数
                    }
                    
                    segmented_image = self.visualizer.draw_segmentation(
                        result_image, masks, boxes, classes, confidences, class_names, depth_frame, fps_info
                    )
                    
                    if self.result_queue.full():
                        try:
                            self.result_queue.get_nowait()
                        except:
                            pass
                    self.result_queue.put((segmented_image, depth_frame))
                    
            except:
                continue
    
    def run(self):
        """运行主应用程序"""
        if not self.initialize():
            print("应用程序初始化失败")
            return
        
        self.running = True
        
        # 启动相机和处理线程
        camera_thread = threading.Thread(target=self.camera_thread)
        processing_thread = threading.Thread(target=self.processing_thread)
        
        camera_thread.daemon = True
        processing_thread.daemon = True
        
        camera_thread.start()
        processing_thread.start()
        
        print("应用程序开始运行，按以下键操作:")
        print("  'q' - 退出")
        print("  's' - 保存当前帧")
        print("  'd' - 切换显示深度图")
        print("  't' - 切换距离显示")
        print("  'c' - 切换显示模式（分割/深度/并排）")
        
        show_mode = 0  # 0: 分割结果, 1: 深度图, 2: 并排显示
        save_count = 0
        
        try:
            while self.running:
                try:
                    segmented_image, depth_frame = self.result_queue.get(timeout=1.0)
                    
                    # 根据显示模式准备图像
                    if show_mode == 0:  # 只显示分割结果
                        display_image = segmented_image
                    elif show_mode == 1:  # 只显示深度图
                        display_image = self.visualizer.create_depth_colormap(depth_frame)
                    else:  # 并排显示
                        depth_colormap = self.visualizer.create_depth_colormap(depth_frame)
                        # 调整尺寸以匹配
                        if segmented_image.shape != depth_colormap.shape:
                            depth_colormap = cv2.resize(depth_colormap, 
                                                       (segmented_image.shape[1], segmented_image.shape[0]))
                        display_image = np.hstack((segmented_image, depth_colormap))
                    
                    # 添加模式指示器
                    mode_text = ["Segmentation", "Depth Map", "Side by Side"][show_mode]
                    cv2.putText(display_image, f"Mode: {mode_text}", 
                               (10, display_image.shape[0] - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    
                    # 添加检测模式指示器
                    cv2.putText(display_image, "Detection: Person Only (No BBox)", 
                               (display_image.shape[1] - 300, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    cv2.imshow('YOLOv11 RealSense Person Detection (No BBox)', display_image)
                    
                except:
                    # 显示等待信息
                    wait_image = np.zeros((480, 848, 3), dtype=np.uint8)
                    cv2.putText(wait_image, "等待相机数据...", (300, 240), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.imshow('YOLOv11 RealSense Person Detection (No BBox)', wait_image)
                
                # 处理键盘输入
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # 保存当前帧
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"person_detection_no_bbox_{timestamp}_{save_count}.jpg"
                    cv2.imwrite(filename, segmented_image)
                    print(f"保存图像: {filename}")
                    save_count += 1
                elif key == ord('d'):
                    self.show_depth = not self.show_depth
                    print(f"显示深度图: {'开启' if self.show_depth else '关闭'}")
                elif key == ord('t'):
                    self.show_distance = not self.show_distance
                    print(f"显示距离: {'开启' if self.show_distance else '关闭'}")
                elif key == ord('c'):
                    show_mode = (show_mode + 1) % 3
                    mode_names = ["分割结果", "深度图", "并排显示"]
                    print(f"显示模式: {mode_names[show_mode]}")
                    
        except KeyboardInterrupt:
            print("程序被用户中断")
        finally:
            self.running = False
            self.camera.stop()
            cv2.destroyAllWindows()
            print("应用程序已关闭")
