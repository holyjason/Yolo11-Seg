#!/usr/bin/env python3

# -*- coding:utf-8 -*-
# Author：Bill Liu
# Create：2025-11-01
# Update：2025-11-01
"""
帧率计数器类
"""

import time

class FPSCounter:
    """帧率计数器类"""
    
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.timestamps = []
        
    def update(self):
        """更新帧率计数"""
        self.timestamps.append(time.time())
        if len(self.timestamps) > self.window_size:
            self.timestamps.pop(0)
    
    def get_fps(self):
        """计算当前帧率"""
        if len(self.timestamps) < 2:
            return 0.0
            
        time_span = self.timestamps[-1] - self.timestamps[0]
        if time_span > 0:
            return len(self.timestamps) / time_span
        return 0.0
