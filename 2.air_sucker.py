# realsense and Robot calibration
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pyrealsense2 as rs
import numpy as np
import cv2
import cv2.aruco as aruco
import time
import socket, pickle,time
import random
from episode_controller.episodeServer import EpisodeAPP

class Calibration:
    def __init__(self):
        # initialize 
        # 吸嘴长度
        self.sucker_length = 60
        self.EpRobot = EpisodeAPP()
        
        # move to calibration position
        self.EpRobot.move_robot_to_xyz_fixed_oritentaion(320,0,100)

        # initialize realsense
        # Create a context object. This object owns the handles to all connected realsense devices
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30) # 分别是宽、高、数据格式、帧率
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self.align = rs.align(rs.stream.color)
        
        # Start streaming
        self.pipeline.start(config)

        # dictionary used in ArUco markers
        self.dictionary = aruco.getPredefinedDictionary(aruco.DICT_5X5_50)
        # create parameters object
        self.parameters = aruco.DetectorParameters()

    def get_aruco_center(self, calib = True):
        # Create a pipeline object. This object configures the streaming camera and owns it's handle
        frames = self.pipeline.wait_for_frames()
        frames = self.align.process(frames)
        # get depth frame
        depth = frames.get_depth_frame()
        
        # display color frame
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())

        # 获取intelrealsense参数
        intr = color_frame.profile.as_video_stream_profile().intrinsics
        # 内参矩阵，转ndarray方便后续opencv直接使用
        intr_matrix = np.array([
            [intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]
        ]) 
        intr_coeffs = np.array(intr.coeffs)
        # 输入rgb图, aruco的dictionary, 相机内参, 相机的畸变参数
        corners, ids, rejected_img_points = aruco.detectMarkers(color_image, self.dictionary, parameters=self.parameters)

        rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(corners, 0.05, intr_matrix, intr_coeffs)
        
        center = None
        # if markers are detected
        if ids is not None:
            # draw borders around markers
            aruco.drawDetectedMarkers(color_image, corners)
            # draw axis around markers, parameters: image, camera internal parameters, distortion parameters, rotation vector, translation vector, length of axis line
            cv2.drawFrameAxes(color_image, intr_matrix, intr_coeffs, rvec, tvec, 0.05) 
            # print ids and corners of detected markers
            for i, corner in zip(ids, corners):
                # get aruco center coordinate
                # if calib:
                #     x = (corner[0][0][0] + corner[0][3][0]) / 2
                #     y = (corner[0][0][1] + corner[0][3][1]) / 2
                # else:
                x = (corner[0][0][0] + corner[0][2][0]) / 2
                y = (corner[0][0][1] + corner[0][2][1]) / 2
                    
                cv2.circle(color_image, (int(x), int(y)), 3, (0, 0, 255), -1)

                # get middle pixel distance
                dist_to_center = depth.get_distance(int(x), int(y))
                # realsense提供的方法，将像素坐标转换为相机坐标系下的坐标
                x_cam, y_cam, z_cam = rs.rs2_deproject_pixel_to_point(intr, [x, y], dist_to_center)
                display_txt = "x: {:.3f}, y: {:.3f}, z: {:.3f}".format(x_cam, y_cam, z_cam)
                cv2.putText(color_image, display_txt, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                center = [x_cam, y_cam, z_cam]
                
                # just need one marker
                break

        # depth frame
        depth_img = np.asanyarray(depth.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=0.14), cv2.COLORMAP_JET)

        # stack color frame and depth frame
        images = np.hstack((color_image, depth_colormap))
        return images, center
        
    def run_calibration(self):
        # Robot move to different position
        # get the position of the aruco marker
        # calculate the transform matrix
        # save the transform matrix
        # test the transform matrix

        # 打开吸嘴
        self.EpRobot.gripper(1)
        print("#################please put the aruco marker on the Robot end effector")
        time.sleep(5)

        # define move points, x, y, z, r
        default_cali_points = [(350, 0, 10),
                                (300, 50, 10),
                                (250, 0, 10),
                                (300, -50, 10),
                                (400, 0, 10),
                                (300, 100, 10),
                                (250, 0, 10),
                                (300, -100, 10),
                                (430, 0, 10),
                                (300, 150, 10),
                                (250, 0, 10),
                                (300, -150, 10),
                                (430, 0, 10),
                                (300, 200, 10),
                                (250, 0, 10),
                                (299, -200, 10),
                                (350, 0, 30),
                                (300, 50, 30),
                                (250, 0, 30),
                                (300, -50, 30),
                                (400, 0, 30),
                                (300, 100, 30),
                                (250, 0, 30),
                                (300, -100, 30),
                                (430, 0, 30),
                                (300, 150, 30),
                                (250, 0, 30),
                                (300, -150, 30),
                                (430, 0, 30),
                                (300, 200, 30),
                                (250, 0, 30),
                                (299, -200, 30),
                                (350, 0, 50),
                                (300, 50, 50),
                                (250, 0, 50),
                                (300, -50, 50),
                                (400, 0, 50),
                                (300, 100, 50),
                                (250, 0, 50),
                                (300, -100, 50),
                                (430, 0, 50),
                                (300, 150, 50),
                                (250, 0, 50),
                                (300, -150, 50),
                                (430, 0, 50),
                                (300, 200, 50),
                                (250, 0, 50),
                                (299, -200, 50),
                                (350, 0, 70),
                                (300, 50, 70),
                                (250, 0, 70),
                                (300, -50, 70),
                                (400, 0, 70),
                                (300, 100, 70),
                                (250, 0, 70),
                                (300, -100, 70),
                                (430, 0, 70),
                                (300, 150, 70),
                                (250, 0, 70),
                                (300, -150, 70),
                                (430, 0, 70),
                                (300, 200, 70),
                                (250, 0, 70),
                                (299, -200, 70),
                                (350, 0, 90),
                                (300, 50, 90),
                                (250, 0, 90),
                                (300, -50, 90),
                                (400, 0, 90),
                                (300, 100, 90),
                                (250, 0, 90),
                                (300, -100, 90),
                                (430, 0, 90),
                                (300, 150, 90),
                                (250, 0, 90),
                                (300, -150, 90),
                                (430, 0, 90),
                                (300, 200, 90),
                                (250, 0, 90),
                                (299, -200, 90),
                                (350, 0, 110),
                                (300, 50, 110),
                                (250, 0, 110),
                                (300, -50, 110),
                                (400, 0, 110),
                                (300, 100, 110),
                                (250, 0, 110),
                                (300, -100, 110),
                                (430, 0, 110),
                                (300, 150, 110),
                                (250, 0, 110),
                                (300, -150, 110),
                                (430, 0, 110),
                                (300, 200, 110),
                                (250, 0, 110),
                                (299, -200, 110)]
                           
        np_cali_points = np.array(default_cali_points)
        arm_cord = np.column_stack(
            (np_cali_points[:, 0:3], np.ones(np_cali_points.shape[0]).T)).T 
        centers = np.ones(arm_cord.shape)
        
        img_to_arm_file = "./save_parms/image_to_arm.npy"
        arm_to_img_file = "./save_parms/arm_to_image.npy"

        if os.path.exists(img_to_arm_file) and os.path.exists(arm_to_img_file):
            image_to_arm = np.load(img_to_arm_file)
            arm_to_image = np.load(arm_to_img_file)
            print("load image to arm and arm to image transform matrix")
        else:
            print("need to calibrate the camera and Robot")
            for index, point in enumerate(default_cali_points):
                print("#################Robot move to point {}, x: {}, y: {}, z: {}".format(index, point[0], point[1], point[2]))
                # self.device.speed(100, 100 )
                # move to the point
                # 需要注意吸嘴长度加上
                self.EpRobot.move_robot_to_xyz_fixed_oritentaion(point[0], point[1], point[2] + self.sucker_length )
                # add x offset
                arm_cord.T[index][0] = arm_cord.T[index][0] + 50 # +50 因为 aruco marker 的中心点距离end effector 50mm

                # get the position of the aruco marker
                images, center = self.get_aruco_center()
                if center is not None:
                    # save the center
                    centers[0:3, index] = center                
                    # display the image
                    cv2.imshow("image", images)
                    cv2.waitKey(1)
                else:
                    print("no aruco marker detected")
                    continue

                time.sleep(1)

        # calculate the transform matrix
        image_to_arm = np.dot(arm_cord, np.linalg.pinv(centers))
        arm_to_image = np.linalg.pinv(image_to_arm)
        print("Finished calibration!")

        print("Image to arm transform:\n", image_to_arm)
        print("Arm to Image transform:\n", arm_to_image)
        # write to file
        np.save(img_to_arm_file, image_to_arm)
        np.save(arm_to_img_file, arm_to_image)

        print("Sanity Test:")

        print("-------------------")
        print("Image_to_Arm")
        print("-------------------")
        for ind, pt in enumerate(centers.T):
            print("Expected:", arm_cord.T[ind][0:3])
            print("Result:", np.dot(image_to_arm, np.array(pt))[0:3])

        print("-------------------")
        print("Arm_to_Image")
        print("-------------------")
        for ind, pt in enumerate(arm_cord.T):
            print("Expected:", centers.T[ind][0:3])
            pt[3] = 1
            print("Result:", np.dot(arm_to_image, np.array(pt))[0:3])

    def run_recog(self):
        if os.path.exists("./save_parms/image_to_arm.npy"):
            image_to_arm = np.load("./save_parms/image_to_arm.npy")
        self.EpRobot.gripper(0)
        time.sleep(3)
        while True:
            images, center = self.get_aruco_center( calib = False)
            if center is not None:
                cv2.imwrite("save.jpg", images)
                cv2.imshow("image", images)
                cv2.waitKey(1)
                img_pos = np.ones(4)
                img_pos[0:3] = center
                arm_pos = np.dot(image_to_arm, np.array(img_pos))
                print(arm_pos)
                # if (np.sqrt(arm_pos[0]*arm_pos[0] + arm_pos[1]*arm_pos[1]) > 300):
                #     print("Can not reach!!!!!!!!!!!!!!!")
                #     time.sleep(3)
                #     continue
                # self.EpRobot.speed(100, 100)
                
                self.EpRobot.move_robot_to_xyz_fixed_oritentaion(arm_pos[0], arm_pos[1], arm_pos[2]+ self.sucker_length + 100)
                self.EpRobot.move_robot_to_xyz_fixed_oritentaion(arm_pos[0], arm_pos[1], arm_pos[2]+ self.sucker_length -8)
                self.EpRobot.gripper(1)
                self.EpRobot.move_robot_to_xyz_fixed_oritentaion(arm_pos[0], arm_pos[1], arm_pos[2]+ self.sucker_length + 100)
                # self.device.speed(50, 50)
                # self.EpRobot.move_robot_to_xyz_fixed_oritentaion(arm_pos[0], arm_pos[1], arm_pos[2] + self.sucker_length)
                # self.device.speed(100, 100)
                # x range: 140 - 300,  
                


                self.EpRobot.move_robot_to_xyz_fixed_oritentaion(random.randint(250, 380), random.randint(-110, 210), 100 + self.sucker_length)
               
                self.EpRobot.gripper(0)
                time.sleep(1)
                print("another one")
                
if __name__ == "__main__":
    cali = Calibration()
    if not os.path.exists("./save_parms/image_to_arm.npy") or not os.path.exists("./save_parms/arm_to_image.npy"):
        cali.run_calibration()
    cali.run_recog()