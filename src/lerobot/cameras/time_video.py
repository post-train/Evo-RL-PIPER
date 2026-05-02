#!/usr/bin/env python3
# -*-coding:utf8-*-
"""
实时显示三个摄像头的画面 (纯 OpenCV 版本)
video0, video2: 旋转 180 度
video4: 不旋转
"""

import cv2
import numpy as np

def main():
    # 相机配置字典：{设备路径: 是否需要旋转180度}
    camera_configs = {
        "/dev/video0": True,
        "/dev/video2": True,
        "/dev/video4": True,
    }
    
    caps = {}
    print("正在连接相机...")
    
    for cam_id, rotate in camera_configs.items():
        # 提取设备号，例如 /dev/video0 提取出 0
        try:
            idx = int(cam_id.replace("/dev/video", ""))
        except ValueError:
            print(f"  无效的设备路径: {cam_id}")
            continue
            
        # 关键修复：强制使用 V4L2 后端 (cv2.CAP_V4L2)
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        
        # 强制设置分辨率，避免底层报错
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # 推荐：如果你需要高帧率，可以取消下面这行的注释
        # cap.set(cv2.CAP_PROP_FPS, 30)
        
        if cap.isOpened():
            caps[cam_id] = (cap, rotate)
            print(f"  {cam_id} 已连接 (旋转: {'180度' if rotate else '无'})")
        else:
            print(f"  {cam_id} 连接失败！")
            cap.release()
            
    if not caps:
        print("没有可用的相机，退出程序")
        return
        
    print("\n按 'q' 键退出")
    print("-" * 40)
    
    try:
        while True:
            for cam_id, (cap, rotate) in caps.items():
                ret, frame = cap.read()
                
                if ret:
                    # 如果该相机配置为需要旋转
                    if rotate:
                        frame = cv2.rotate(frame, cv2.ROTATE_180)
                    cv2.imshow(f"Camera: {cam_id}", frame)
                else:
                    # 如果偶发读取失败，显示带有提示的黑屏，防止程序崩溃退出
                    black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(black_frame, "No Signal", (220, 240), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow(f"Camera: {cam_id}", black_frame)
                    
            # 监听键盘按键，按下 'q' 退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        print("\n正在断开相机连接...")
        for cam_id, (cap, _) in caps.items():
            try:
                cap.release()
                print(f"  {cam_id} 已断开")
            except Exception as e:
                print(f"  {cam_id} 断开时发生错误: {e}")
                
        cv2.destroyAllWindows()
        print("程序已退出")

if __name__ == "__main__":
    main()