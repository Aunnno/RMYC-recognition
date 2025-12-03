from ultralytics import YOLO
import cv2
import os
import time
import types  # 用于判断返回值是否为 generator

class recognizer:
    def __init__(self) -> None:
        self.photos_path = "photos" # 图片路径
        self.model_path = "model" # 模型路径
        self.cap_running = False
        self.model_running = False
        self.conf = 0.7 # 置信度阈值
        self.iou = 0.7 # 非极大值抑制
        self._fps = 0.0 # 平滑帧率值，用于显示
        # 仅用于显示缩放尺寸（不改变采集分辨率）
        self.display_size = (320, 240)
        if not os.path.exists(self.photos_path): # 照片路径检测
            os.makedirs(self.photos_path)
        if not os.path.exists(self.model_path): # 模型路径检测
            os.makedirs(self.model_path)

    def camera_init(self) -> bool: # 摄像头初始化
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("未读取到摄像头")
                return False
            # 摄像头参数设置
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480) # 高480
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,640) # 宽640
            self.cap.set(cv2.CAP_PROP_FPS,30.0) # 帧率30
            self.cap_running = True
            print("摄像头打开成功")
            return True
        except Exception as e:
            print(f"摄像头打开出错:{e}")
            return False

    def model_init(self) -> bool: # 模型初始化
        """
        要求输入模型文件名（位于 model_path 下），如：yolov8n.onnx 或 best.pt
        返回：初始化是否成功
        """
        try:
            print("输入模型名（位于 model/ 目录下）：")
            name = input().strip()
            if name == "":
                print("模型名为空")
                return False
            path = os.path.join(self.model_path, name)  # 修复：使用 model_path
            if not os.path.isfile(path):
                print("未找到该文件:", path)
                return False
            # 加载模型
            self.model = YOLO(path)
            if self.model is None:
                print("模型打开失败")
                return False
            self.model_running = True
            print("模型加载成功")
            return True
        except Exception as e:
            print(f"打开模型时出现错误:{e}")
            return False

    def cap_to_display(self) -> None: # 从摄像头获取图像，标注后显示到屏幕
        print("从摄像头到屏幕（按 c 结束）")
        if not self.model_running:
            if not self.model_init():
                return
        if not self.cap_running:
            if not self.camera_init():
                return
        prev_time = time.time()
        try:
            while True:
                ret,frame = self.cap.read()
                if not ret:
                    print("未读取到帧")
                    return
                # 推理（stream=True 返回 generator）
                results = self.model(frame,conf=self.conf,iou=self.iou,stream=True)
                # 兼容 generator 与 可下标结果的处理
                if isinstance(results, types.GeneratorType):
                    res = next(results)  # 获取第一个结果
                else:
                    res = results[0]
                out_frame = res.plot()

                # 缩放用于显示（不改变原始采集分辨率）
                try:
                    out_frame = cv2.resize(out_frame, self.display_size)
                except Exception:
                    # 若 resize 失败则保持原图
                    pass

                # 计算并平滑 FPS
                cur_time = time.time()
                dt = cur_time - prev_time if cur_time - prev_time > 0 else 1e-6
                inst_fps = 1.0 / dt
                if self._fps <= 0.0:
                    self._fps = inst_fps
                else:
                    alpha = 0.9  # 平滑因子（越接近1越平滑）
                    self._fps = alpha * self._fps + (1 - alpha) * inst_fps
                prev_time = cur_time

                # 在图像左上角绘制 FPS（适配缩放后尺寸）
                fps_text = f"FPS: {self._fps:.1f}"
                cv2.putText(out_frame, fps_text, (10,22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

                cv2.imshow("camera",out_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('c'):
                    return
        except Exception as e:
            print(f"实时检测错误:{e}")

    def photo_to_display(self) -> None: # 获取照片，标注后显示到屏幕
        print("从照片到屏幕（按 p 结束）")
        print("输入照片名（位于 photos/ 目录下）：")
        name = input().strip()
        if name == "":
            print("照片名为空")
            return
        path = os.path.join(self.photos_path,name)
        if not os.path.isfile(path):
            print("未找到该文件:", path)
            return
        if not self.model_running:
            if not self.model_init():
                return
        prev_time = time.time()
        try:
            while True:
                results = self.model(path,conf=self.conf,iou=self.iou)
                # 兼容返回类型
                if isinstance(results, types.GeneratorType):
                    res = next(results)
                else:
                    res = results[0]
                out_frame = res.plot()

                # 缩放用于显示（不改变原始图片文件）
                try:
                    out_frame = cv2.resize(out_frame, self.display_size)
                except Exception:
                    pass

                # 计算并平滑 FPS（对于基于单张图片连续推理也显示速度）
                cur_time = time.time()
                dt = cur_time - prev_time if cur_time - prev_time > 0 else 1e-6
                inst_fps = 1.0 / dt
                if self._fps <= 0.0:
                    self._fps = inst_fps
                else:
                    alpha = 0.9
                    self._fps = alpha * self._fps + (1 - alpha) * inst_fps
                prev_time = cur_time

                fps_text = f"FPS: {self._fps:.1f}"
                cv2.putText(out_frame, fps_text, (10,22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

                cv2.imshow("photo",out_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('p'):
                    return
        except Exception as e:
            print(f"图片检测错误:{e}")

    # def cap_to_robot(self): # 从摄像头获取图像并返回中心点给车子（保留为注释示例）
    #     if not self.model_running:
    #         if not self.model_init():
    #             return
    #     if not self.cap_running:
    #         if not self.camera_init():
    #             return
    #     try:
    #         while True:
    #             ret,frame = self.cap.read()
    #             if not ret:
    #                 print("未读取到帧")
    #                 return
    #             results = self.model(frame)
    #             boxes = results[0].boxes
    #             for i,box in enumerate(boxes):
    #                 x1,y1,x2,y2=box.xyxy[0].cpu().numpy()
    #                 x=(x1+x2)/2
    #                 y=(y1+y2)/2
    #                 # 这里要写一个传数据的函数但是暂时没有
    #             key = cv2.waitKey(1) & 0xFF
    #             if key == ord('c'):
    #                 return

    def clean_up(self) -> None: # 手动调用清理的接口
        if self.cap_running and getattr(self, "cap", None) is not None:
            self.cap_running = False
            try:
                self.cap.release()
            except Exception:
                pass
        self.model_running = False
        cv2.destroyAllWindows()

    def __del__(self) -> None:
        self.clean_up() # 程序退出时确保清理

if __name__ == "__main__":
    # 简单交互式测试入口
    r = recognizer()
    try:
        while True:
            print("\n选择操作：")
            print("1: 摄像头实时检测（按 c 结束）")
            print("2: 图片检测（按 p 结束）")
            print("q: 退出")
            choice = input("输入选项: ").strip().lower()
            if choice == "1":
                r.cap_to_display()
            elif choice == "2":
                r.photo_to_display()
            elif choice == "q":
                break
            else:
                print("无效选项")
    finally:
        r.clean_up()