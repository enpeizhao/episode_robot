""" Demo to show prediction results.
    Author: chenxi-wang
"""

import os
import sys
import numpy as np
import open3d as o3d
import argparse
import importlib
import scipy.io as scio
from PIL import Image
import cv2
import time

import torch
from graspnetAPI import GraspGroup
# https://graspnetapi.readthedocs.io/en/latest/about.html

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from graspnet import GraspNet, pred_decode
from graspnet_dataset import GraspNetDataset
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image

from episode_controller.episodeServer import EpisodeAPP
from scipy.spatial.transform import Rotation 

class GraspNetDemo:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--checkpoint_path',  default="./checkpoint-rs.tar", help='Model checkpoint path')
        parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
        parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
        parser.add_argument('--collision_thresh', type=float, default=0.01, help='Collision Threshold in collision detection [default: 0.01]')
        parser.add_argument('--voxel_size', type=float, default=0.01, help='Voxel Size to process point clouds before collision detection [default: 0.01]')
        self.cfgs = parser.parse_args()

        
        self.EpRobot = EpisodeAPP('localhost',3001)

        # load camera2end T
        R_camera2end = np.load('./episode_params/R_camera2end.npy')
        t_camera2end = np.load('./episode_params/t_camera2end.npy')
        T_camera2end = np.eye(4)
        T_camera2end[:3, :3] = R_camera2end
        T_camera2end[:3, 3] = t_camera2end.reshape(3)
        self.T_camera2end = T_camera2end

        # end2base T
        self.T_end2base = self.EpRobot.calculate_ee2base_T([260,0,400],[90,0,180])

        # T_servo2end
        theta = 0
        alpha = 0
        d = 120 
        a = 0
        self.T_servo2end = np.array(
            [
                [np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
                [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
                [0, np.sin(alpha), np.cos(alpha), d],
                [0, 0, 0, 1]
            ]
        )
        
        # move to initial position
        self.EpRobot.move_robot_to_xyz_fixed_oritentaion(260,0,400)
        # servo to initial position
        self.EpRobot.servo_gripper(0)

    def get_net(self):
        # Init the model
        net = GraspNet(input_feature_dim=0, num_view=self.cfgs.num_view, num_angle=12, num_depth=4,
                cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.to(device)
        # Load checkpoint
        checkpoint = torch.load(self.cfgs.checkpoint_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        print("-> loaded checkpoint %s (epoch: %d)"%(self.cfgs.checkpoint_path, start_epoch))
        # set model to eval mode
        net.eval()
        return net



    def get_and_process_data(self,camera,bbox):
        # load data
        color = np.load('./VLM_related/realsense_captured/color.npy')
        depth = np.load('./VLM_related/realsense_captured/depth.npy')
        # bgr to rgb
        color = color[:, :, [2, 1, 0]]
        # as float32
        color = color.astype(np.float32) / 255.0
        depth = depth.astype(np.float32)

        # workspace_mask = np.array(Image.open('doc/example_data/workspace_mask.png'))
        l,t,r,b = bbox
        workspace_mask =  np.zeros((720,1280), dtype=bool)
        if l < 280:
            l = 280
        if r > (280+720):
            r = 280+720
        workspace_mask[t:b,l:r] = True

        # generate cloud
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

        # get valid points
        mask = (workspace_mask & (depth > 0) )  
        cloud_masked = cloud[mask]
        color_masked = color[mask]

        # sample points
        if len(cloud_masked) >= self.cfgs.num_point:
            idxs = np.random.choice(len(cloud_masked), self.cfgs.num_point, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.cfgs.num_point-len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxs]
        color_sampled = color_masked[idxs]

        # convert data
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
        cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
        end_points = dict()
        cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cloud_sampled = cloud_sampled.to(device)
        end_points['point_clouds'] = cloud_sampled
        end_points['cloud_colors'] = color_sampled

        return end_points, cloud

    def get_grasps(self, net, end_points):
        # Forward pass
        with torch.no_grad():
            end_points = net(end_points)
            grasp_preds = pred_decode(end_points)
        gg_array = grasp_preds[0].detach().cpu().numpy()
        gg = GraspGroup(gg_array)
        return gg

    def collision_detection(self, gg, cloud):
        mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=self.cfgs.voxel_size)
        collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=self.cfgs.collision_thresh)
        gg = gg[~collision_mask]
        return gg

    def filter_grasps_from_inliers(self, grasp_positions, cloud, distance_thresh=0.003):
        plane_model, inliers = cloud.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
        distance_matrix = np.linalg.norm(grasp_positions[:, np.newaxis] - np.asarray(cloud.points)[inliers], axis=2)
        is_close_to_inlier = np.any(distance_matrix < distance_thresh, axis=1)
        return np.where(~is_close_to_inlier)[0]
    
    def filterT(self):
        pass

    def vis_grasps(self,  gg, cloud):
        # extra degree and extra height
        extra_degree = 83
        extra_height = 0
        

        gg.nms()
        gg.sort_by_score()
        gg = gg[:1]
    

        rotation_matrix = gg.rotation_matrices
        translation = gg.translations
        widths = gg.widths

        if len (rotation_matrix) == 0:
            print("No grasps found.")
            return

        # display in the open3d window
        grippers = gg.to_open3d_geometry_list()
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([mesh_frame, cloud, *grippers])

     

         # 创建一个 4x4 的齐次变换矩阵
        T_cam = np.eye(4)

        # 将旋转矩阵插入到左上角 3x3 子矩阵
        T_cam[:3, :3] = rotation_matrix[0]
        # 交换第一列和第三列
        T_cam[:, [0, 2]] = T_cam[:, [2, 0]]
        # 取第一列（原来的第三列）的负值
        T_cam[:, 0] = -T_cam[:, 0]
        # 将平移向量插入到最后一列
        # T_cam = rotate_homogeneous_matrix_around_z(T_cam)
        T_cam[:3, 3] = translation[0] * 1000



        # 转换到工具末端坐标系
        T_gripper = self.T_camera2end @ T_cam
        print(f'相对于末端的坐标和旋转矩阵：\n{T_gripper}')

        T_base = self.T_end2base @ T_gripper
        print(f'相对于基座的坐标和旋转矩阵：\n{T_base}')

        
        # calculate the new end2base
        new_T_end2base = T_base @ np.linalg.inv(self.T_servo2end)

        P_base = new_T_end2base[:3, 3]
        print(f'相对于基座的坐标：{P_base}')

        # move to basic position
        rotation = Rotation.from_matrix(new_T_end2base[:3, :3])
        euler = rotation.as_euler('XYZ', degrees=True)
      

        print("Directly grasp")
    
        # move to the position above the object
        self.EpRobot.move_robot_to_xyz_oritentaion(P_base[0], P_base[1], P_base[2]+ extra_height +100, [euler[0],euler[1],euler[2]+extra_degree,'xyz'])
        # move to the object
        self.EpRobot.move_robot_to_xyz_oritentaion(P_base[0], P_base[1], P_base[2] + extra_height, [euler[0],euler[1],euler[2] + extra_degree,'xyz'])
        # grasp the object
        self.EpRobot.servo_gripper(90)
        # move to the position above the object
        self.EpRobot.move_robot_to_xyz_oritentaion(P_base[0], P_base[1], P_base[2]+ extra_height +100, [euler[0],euler[1],euler[2]+extra_degree,'xyz'])
        self.EpRobot.move_robot_to_xyz_fixed_oritentaion(140, -300, 340,)
        # release the object
        self.EpRobot.servo_gripper(0)
        
        # move back     
        self.EpRobot.move_robot_to_xyz_fixed_oritentaion(260,0,400)
            
      
    def demo(self):
        net = self.get_net()
        result_box_file = './VLM_related/exchange/result_box.npy'
        if os.path.exists(result_box_file):
            os.remove(result_box_file)

        while True:
            # 检查文件是否存在
            if os.path.exists(result_box_file):
                # get camera parameters
                intrinsic = np.load('./VLM_related/exchange/intr_matrix.npy')
                factor_depth = [[1000.]]
                camera = CameraInfo(1280, 720, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
                # 获取点云数据
                bbox = np.load(result_box_file)
                end_points, cloud = self.get_and_process_data(camera, bbox)
                # 获取抓取点
                gg = self.get_grasps(net, end_points)
                if self.cfgs.collision_thresh > 0:
                    self.vis_grasps( gg, cloud)
            
                # 删除文件
                os.remove(result_box_file)
            else:
                print("Wait...")
                time.sleep(1)


if __name__ == '__main__':
    demo = GraspNetDemo()
    demo.demo()
