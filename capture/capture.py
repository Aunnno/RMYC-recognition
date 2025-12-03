import cv2
import os
from datetime import datetime
import time
class CameraCapture:
    def __init__(self):
        self.camera_index = 0
        self.save_dir = "captured"
        self.is_running = False
        self.is_recording = False
        self.video_writer = None
        self.is_shooting = False
        self.delta_t = 0.2
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
    def camera_init(self): #摄像头初始化
        try:
            self.cap=cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                print("无法打开摄像头")
                return False
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            print("摄像头初始化成功")
            self.is_running = True
            return True
        except Exception as e:
            print(f"初始化失败:{e}")
            return False
    def capture_photo(self): #拍照
        try: 
            ret, frame = self.cap.read()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"photo_{timestamp}.jpg"
            filepath = os.path.join(self.save_dir,filename)
            if not ret:
                print("无法读取")
                return False
            success = cv2.imwrite(filepath,frame)
            if success:
                print(f"照片已保存")
                return True
            else:
                print(f"保存失败")
                return False
        except Exception as e:
            print(f"拍照时出现错误:{e}")
            return False
    def start_record(self,frame_width,frame_height,fps): #开始录像
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"video_{timestamp}.avi"
            filepath = os.path.join(self.save_dir,filename)
            fourcc = cv2.VideoWriter.fourcc(*'XVID')
            self.video_writer = cv2.VideoWriter(filepath,fourcc,fps,(frame_width,frame_height))
            if self.video_writer.isOpened():
                print("录制中")
                self.is_recording = True
                return True
            else:
                print("录制失败")
                return False
        except Exception as e:
            print(f"录制时出现错误{e}")
            return False
    def stop_record(self): #结束录像
        if self.video_writer is not None and self.is_recording:
            self.video_writer.release()
            self.is_recording = False
            self.video_writer = None
            print("录制结束")
        else:
            print("未发现录制进程")

    def write_frame(self,frame): #写入帧
        if self.video_writer is not None and self.is_recording:
            self.video_writer.write(frame)
    def preview(self): #预览
        print("c拍照,v开始/停止连拍,r开始/停止录制,q退出")
        if not self.camera_init():
            return
        try:
            frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            while self.is_running:
                ret,frame=self.cap.read()
                if not ret:
                    print("读取帧失败")
                    break
                if self.is_recording:
                    self.write_frame(frame)
                cv2.imshow('camera',frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('c'):
                    self.capture_photo()
                elif key == ord('r'):
                    if self.is_recording:
                        self.stop_record()
                    else:
                        self.start_record(frame_width,frame_height,30.0)
                elif key == ord('v'):
                    while True:
                        if self.is_shooting:
                            self.is_shooting = False
                            break
                        else:
                            self.capture_photo
                            time.sleep(self.delta_t)
                elif key == ord('q'):
                    print("退出程序")
                    break
        except KeyboardInterrupt:
            print("用户自行退出")
        except Exception as e:
            print(f"录制时出现错误:{e}")
        finally:
            self.clean_up()
    def clean_up(self): #清理
        if self.is_recording:
            self.stop_record()
        self.is_running = False
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
def main():
    camera = CameraCapture()
    camera.preview()
if __name__ == "__main__":
    main()