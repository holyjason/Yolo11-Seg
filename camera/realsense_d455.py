#!/usr/bin/env python3

# -*- coding:utf-8 -*-
# Author：Bill Liu
# Create：2025-11-01
# Update：2025-11-01
"""
RealSense D455相机控制类
"""

import cv2
import numpy as np
import pyrealsense2 as rs

class RealSenseD455:
    """RealSense D455相机控制类 - 修复版本"""
    
    def __init__(self, width=848, height=480, fps=30):  # 使用D455支持的常见配置
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = None
        self.config = None
        self.align = None
        self.running = False
        
    def initialize(self):
        """初始化RealSense相机"""
        try:
            # 检查设备连接
            ctx = rs.context()
            devices = ctx.query_devices()
            if len(devices) == 0:
                print("未检测到RealSense设备")
                return False
                
            print(f"检测到RealSense设备: {devices[0].get_info(rs.camera_info.name)}")
            
            # 创建管道和配置
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            
            # 启用彩色和深度流 - 使用D455支持的配置
            # D455通常支持848x480 @ 30fps 或 1280x720 @ 30fps
            print(f"尝试配置: 彩色流 {self.width}x{self.height} @ {self.fps}fps")
            self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
            print(f"尝试配置: 深度流 {self.width}x{self.height} @ {self.fps}fps")
            self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
            
            # 创建对齐对象（将深度图对齐到彩色图）
            self.align = rs.align(rs.stream.color)
            
            # 尝试启动管道
            print("启动RealSense管道...")
            profile = self.pipeline.start(self.config)
            
            # 获取深度传感器并设置参数
            depth_sensor = profile.get_device().first_depth_sensor()
            if depth_sensor.supports(rs.option.depth_units):
                depth_sensor.set_option(rs.option.depth_units, 0.001)  # 设置深度单位为米
            
            print("RealSense D455初始化成功")
            print(f"分辨率: {self.width}x{self.height}, FPS: {self.fps}")
            return True
            
        except Exception as e:
            print(f"RealSense初始化失败: {e}")
            # 尝试备用配置
            return self.try_alternative_configs()
    
    def try_alternative_configs(self):
        """尝试备用配置"""
        alternative_configs = [
            (640, 480, 30),   # 640x480 @ 30fps
            (1280, 720, 30),  # 1280x720 @ 30fps
            (848, 480, 15),   # 848x480 @ 15fps
            (640, 480, 15),   # 640x480 @ 15fps
        ]
        
        for width, height, fps in alternative_configs:
            try:
                print(f"尝试备用配置: {width}x{height} @ {fps}fps")
                
                # 创建新的配置
                self.config = rs.config()
                self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
                self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
                
                # 启动管道
                profile = self.pipeline.start(self.config)
                
                # 更新参数
                self.width = width
                self.height = height
                self.fps = fps
                
                print(f"备用配置成功: {width}x{height} @ {fps}fps")
                return True
                
            except Exception as e:
                print(f"备用配置 {width}x{height} @ {fps}fps 失败: {e}")
                continue
        
        print("所有配置尝试均失败")
        return False
    
    def get_frames(self):
        """获取对齐的彩色和深度帧"""
        if not self.pipeline:
            return None, None
            
        try:
            # 等待帧
            frames = self.pipeline.wait_for_frames()
            
            # 对齐深度帧到彩色帧
            aligned_frames = self.align.process(frames)
            
            # 获取对齐后的帧
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                return None, None
                
            # 转换为numpy数组
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            
            return color_image, depth_frame  # 返回深度帧对象而不是numpy数组
            
        except Exception as e:
            print(f"获取帧失败: {e}")
            return None, None
    
    def get_depth_at_point(self, depth_frame, x, y):
        """获取指定点的深度值（米）"""
        if 0 <= x < self.width and 0 <= y < self.height:
            try:
                return depth_frame.get_distance(x, y)
            except:
                return 0.0
        return 0.0
    
    def get_depth_map(self, depth_frame):
        """获取深度图的可视化"""
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), 
            cv2.COLORMAP_JET
        )
        return depth_colormap
    
    def stop(self):
        """停止相机"""
        self.running = False
        if self.pipeline:
            self.pipeline.stop()
