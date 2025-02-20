import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import time
from episode_controller.episodeServer import EpisodeAPP
import threading
import cv2

class GeneratePoints:
    def __init__(self, ):
        self.episode_app = EpisodeAPP('localhost', 3001)
        self.episode_app1 = EpisodeAPP('localhost', 3002)
        self.motors_degrees = None
        self.in_free_mode = True
        self.prev_degrees = []  # 用于保存最近的角度变化记录
        self.last_saved_degrees = None  # 保存上一次存储的角度值
        
    def get_degrees(self):
        # 循环获取角度
        while self.in_free_mode:
            self.motors_degrees = self.episode_app.get_motor_degrees()
            time.sleep(0.1)  # 减少读取频率，避免过高的资源消耗

    def is_motor_stable(self, threshold=0.5, window=5):
        """
        判断机械臂是否稳定
        :param threshold: 稳定判断的角度变化阈值
        :param window: 判断稳定所需的连续帧数量
        :return: 是否稳定（True/False）
        """
        if self.motors_degrees is None:
            return False
        
        self.prev_degrees.append(self.motors_degrees)
        if len(self.prev_degrees) > window:
            self.prev_degrees.pop(0)
        
        if len(self.prev_degrees) < window:
            return False  # 数据不足时无法判断
        
        # 计算每一帧的变化量
        deltas = [np.linalg.norm(np.array(self.prev_degrees[i]) - np.array(self.prev_degrees[i-1])) 
                  for i in range(1, len(self.prev_degrees))]
        
        # 如果所有变化量都小于阈值，则认为稳定
        return all(delta < threshold for delta in deltas)

    def prepare(self):
        self.episode_app.set_free_mode(1)
        thread = threading.Thread(target=self.get_degrees, daemon=True)  # 创建守护线程
        thread.start()
        degrees_list = []
        
        while True:
            image = np.zeros((400, 800, 3), dtype=np.uint8)
            # cv2.putText(image, f"motors_degrees: {self.motors_degrees}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("image", image)
            key = cv2.waitKey(1)
            
            if self.is_motor_stable(threshold=0.5, window=50):  # 机械臂稳定时保存角度
                # 检查是否与上一次保存的角度相似
                if (self.last_saved_degrees is None or 
                    np.linalg.norm(np.array(self.motors_degrees) - np.array(self.last_saved_degrees)) > 20):
                    degrees_list.append(self.motors_degrees)
                    self.last_saved_degrees = self.motors_degrees  # 更新最后保存的角度
                    print(f'Stable motors degrees saved: {self.motors_degrees}')
            
            if key == ord('q'):
                cv2.destroyAllWindows()
                self.episode_app.set_free_mode(0)
                self.in_free_mode = False
                print(len(degrees_list))
                # save the data
                np.save("./motors_degrees.npy", degrees_list)
                break

    def replicate_track(self):
        # 读取角度数据
        degrees_list = np.load("./motors_degrees.npy")
        for degree in degrees_list:
            tt = self.episode_app.angle_mode(degree)
            self.episode_app1.angle_mode(degree)
            time.sleep(tt)
if __name__ == "__main__":
    G = GeneratePoints()
    # G.prepare()
    G.replicate_track()
