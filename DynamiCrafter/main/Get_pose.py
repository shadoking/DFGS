import argparse, os, sys, datetime, glob

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from omegaconf import OmegaConf
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import logging as transf_logging
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from utils.utils import instantiate_from_config
from main.utils_train import get_trainer_callbacks, get_trainer_logger, get_trainer_strategy
from main.utils_train import set_logger, init_workspace, load_checkpoints
from plyfile import PlyData
from einops import rearrange, repeat
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import time
from read_write_model import read_images_binary
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def load_pose(pose_file):
    images = read_images_binary(pose_file)
    poses = []

    for _, data in images.items():
        qvec = data.qvec
        tvec = data.tvec

        pose = np.concatenate([qvec, tvec])
        poses.append(pose)

    poses_np = np.array(poses, dtype=np.float32)  # 转为 NumPy 数组
    poses_tensor = torch.from_numpy(poses_np)
    return poses_tensor.unsqueeze(0)  # [2, 7]


# def get_new_pose(original_pose, num_views=14):
#     """
#     根据两个初始相机的位姿生成一个椭圆轨道，并在轨道上生成新的视角和位姿。
#     轨道会仅包含两个初始视角之间的一段。
#
#     参数:
#         original_pose (torch.Tensor 或 np.ndarray): 包含两个相机位姿的张量 (1, 2, 7)。
#         num_views (int): 需要生成的新视角数量，默认为14。
#
#     返回:
#         list: 包含每个新视角的位姿 (R_new, t_new) 的列表。
#     """
#     # 确保 original_pose 是 NumPy 数组
#     if isinstance(original_pose, np.ndarray):
#         pose_array = original_pose
#     else:
#         pose_array = original_pose.detach().cpu().numpy()  # 确保是 NumPy 数据
#
#     print("original_pose shape:", pose_array.shape)  # (1, 2, 7)
#
#     # 提取第一相机的四元数和平移向量
#     q1 = pose_array[0, 0, :4]  # (4,) 四元数
#     t1 = pose_array[0, 0, 4:7].reshape(3, 1)  # (3,1) 平移向量
#
#     # 提取第二相机的四元数和平移向量
#     q2 = pose_array[0, 1, :4]  # (4,)
#     t2 = pose_array[0, 1, 4:7].reshape(3, 1)  # (3,1)
#
#     # 将四元数转换为旋转矩阵 (确保四元数格式为 [x, y, z, w])
#     R1 = R.from_quat(q1).as_matrix()  # (3,3)
#     R2 = R.from_quat(q2).as_matrix()  # (3,3)
#
#     # 计算相机中心
#     C1 = -R1.T @ t1  # (3,1)
#     C2 = -R2.T @ t2  # (3,1)
#
#     print("R1:\n", R1)
#     print("t1:\n", t1)
#     print("R2:\n", R2)
#     print("t2:\n", t2)
#
#     # 计算椭圆轨道的参数
#     center = (C1 + C2) / 2  # 椭圆的中心点，位于两个视角的中间
#     direction = C2 - C1  # 从第一个视角到第二个视角的方向
#     distance = np.linalg.norm(direction)  # 两个视角之间的距离
#     norm_direction = direction / distance  # 单位方向向量
#
#     # 计算两个视角在椭圆上的对应角度
#     angle1 = np.arctan2(C1[1] - center[1], C1[0] - center[0])  # 视角1的角度
#     angle2 = np.arctan2(C2[1] - center[1], C2[0] - center[0])  # 视角2的角度
#
#     # 确保角度按顺时针或逆时针顺序排列
#     if angle2 < angle1:
#         angle2 += 2 * np.pi  # 保证 angle2 大于 angle1
#
#     # 在两个视角之间均匀分布的角度
#     angles = np.linspace(angle1, angle2, num_views)  # 均匀采样角度
#
#     # 生成轨道上点的参数
#     ellipse_points = []
#     for angle in angles:
#         offset = np.array([np.cos(angle), np.sin(angle), 0])  # 椭圆偏移
#         point = center + offset * distance / 2  # 根据方向调整
#         ellipse_points.append(point)
#
#     ellipse_points = np.array(ellipse_points)  # (num_views, 3)
#
#     # 计算每个点的位姿
#     poses = []
#     for point in ellipse_points:
#         # 相机位置
#         C_new = point.reshape(3, 1)  # 转换为 (3,1)
#
#         # 相机朝向场景中心
#         forward = C_new - center
#         forward = forward / np.linalg.norm(forward)
#
#         # 计算旋转矩阵
#         up = np.array([0, 1, 0])  # 假设上方向为 Y 轴
#         right = np.cross(up, forward.squeeze())  # 计算右方向
#         right = right / np.linalg.norm(right)
#         up = np.cross(forward.squeeze(), right)  # 重新计算 up
#
#         R_new = np.vstack((right, up, -forward.squeeze())).T  # 计算新的旋转矩阵
#
#         # 计算新的平移向量
#         t_new = -R_new @ C_new
#
#         poses.append((R_new, t_new))
#
#     return poses, ellipse_points, center


import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

def get_new_pose(original_pose, num_views=14):
    """
    根据两个初始相机的位姿生成一个椭圆轨道，并在轨道上生成新的视角和位姿。

    参数:
        original_pose (torch.Tensor 或 np.ndarray): 包含两个相机位姿的张量 (1, 2, 7)。
        num_views (int): 需要生成的新视角数量，默认为14。

    返回:
        torch.Tensor: 包含每个新视角的位姿 (R_new, t_new) 的张量，形状为 (16, 7)。
    """
    # 确保 original_pose 是 NumPy 数组
    if isinstance(original_pose, np.ndarray):
        pose_array = original_pose
    else:
        pose_array = original_pose.detach().cpu().numpy()  # 确保是 NumPy 数据

    print("original_pose shape:", pose_array.shape)  # (1, 2, 7)

    # 提取第一相机的四元数和平移向量
    q1 = pose_array[0, 0, :4]  # (4,) 四元数
    t1 = pose_array[0, 0, 4:7].reshape(3, 1)  # (3,1) 平移向量

    # 提取第二相机的四元数和平移向量
    q2 = pose_array[0, 1, :4]  # (4,)
    t2 = pose_array[0, 1, 4:7].reshape(3, 1)  # (3,1)

    # 将四元数转换为旋转矩阵 (确保四元数格式为 [x, y, z, w])
    R1 = R.from_quat(q1).as_matrix()  # (3,3)
    R2 = R.from_quat(q2).as_matrix()  # (3,3)

    # 计算相机中心
    C1 = -R1.T @ t1  # (3,1)
    C2 = -R2.T @ t2  # (3,1)

    # 椭圆参数
    center = (C1 + C2) / 2  # 椭圆中心 (3,1)
    a = np.linalg.norm(C1 - C2) / 2  # 长轴
    b = a / 2  # 短轴

    # 生成椭圆轨道上的点
    angles = np.linspace(0, 2 * np.pi, num_views)  # 均匀采样角度
    ellipse_points = np.array([a * np.cos(angles), b * np.sin(angles), np.zeros(num_views)]).T  # (14,3)

    # 处理广播问题 (转换 center 为 (1,3) 以进行广播)
    center = center.reshape(1, 3)  # 确保形状匹配
    ellipse_points += center  # (14,3) + (1,3) 可广播

    # 计算每个点的位姿
    poses = []
    for point in ellipse_points:
        # 相机位置
        C_new = point.reshape(3, 1)  # 转换为 (3,1)

        # 相机朝向场景中心
        look_at = center.T  # 变成 (3,1)
        forward = look_at - C_new
        forward = forward / np.linalg.norm(forward)

        # 计算旋转矩阵
        up = np.array([0, 1, 0])  # 假设上方向为 Y 轴
        right = np.cross(up, forward.squeeze())  # 计算右方向
        right = right / np.linalg.norm(right)
        up = np.cross(forward.squeeze(), right)  # 重新计算 up

        R_new = np.vstack((right, up, -forward.squeeze())).T  # 计算新的旋转矩阵
        # 强制保证旋转矩阵是右手坐标系
        if np.linalg.det(R_new) < 0:
            R_new = -R_new

        # 计算新的平移向量
        t_new = -R_new @ C_new

        # 将旋转矩阵转换为四元数
        q_new = R.from_matrix(R_new).as_quat()  # (4,)

        # 将四元数和平移向量合并为一个 7 维向量
        pose_new = np.concatenate([q_new, t_new.squeeze()])  # (7,)

        poses.append(pose_new)

    # 将第一个和最后一个视角添加为原始位姿
    q1_t = np.concatenate([q1, t1.squeeze()])  # (7,)
    q2_t = np.concatenate([q2, t2.squeeze()])  # (7,)

    poses.insert(0, q1_t)  # 在最前面添加原始第一个相机位姿
    poses.append(q2_t)  # 在最后添加原始第二个相机位姿

    # 转换为 torch.Tensor 并返回
    poses_tensor = torch.tensor(poses, dtype=torch.float32)

    return poses_tensor


def visualize_orbit(ellipse_points, center, poses, point_cloud, point_colors, save_path):
    """
    可视化椭圆轨道、相机位置和朝向，并保存为图片。

    参数:
        ellipse_points (np.ndarray): 椭圆轨道上的点 (num_views, 3)。
        center (np.ndarray): 椭圆中心 (3,)。
        poses (list): 包含每个新视角的位姿 (R_new, t_new) 的列表。
        point_cloud (np.ndarray): 点云数据 (N, 3)，默认为 None。
        point_colors (np.ndarray): 点云颜色数据 (N, 3)，默认为 None。
        save_path (str): 图片保存路径，默认为 "orbit_visualization.png"。
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制点云（如果提供）
    if point_cloud is not None:
        if point_colors is not None:
            # 使用点云的颜色信息
            ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], c=point_colors, s=1, alpha=0.8, label="Point Cloud")
        else:
            # 如果没有颜色信息，使用默认的灰色
            ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], c='gray', s=1, alpha=0.5, label="Point Cloud")

    # 绘制椭圆轨道
    ax.plot(ellipse_points[:, 0], ellipse_points[:, 1], ellipse_points[:, 2], 'b-', label="Ellipse Orbit")

    # 绘制椭圆中心
    ax.scatter(center[0, 0], center[0, 1], center[0, 2], color='r', s=100, label="Center")

    # 绘制每个相机的位置和朝向
    for i, (R_new, t_new) in enumerate(poses):
        # 相机位置
        camera_pos = -R_new.T @ t_new
        ax.scatter(camera_pos[0], camera_pos[1], camera_pos[2], color='g', s=50)

        # 相机朝向
        forward = R_new[:, 2]  # 相机的 Z 轴是朝向
        ax.quiver(
            camera_pos[0], camera_pos[1], camera_pos[2],
            forward[0], forward[1], forward[2],
            length=1.0, color='orange', normalize=True
        )

    # 设置坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 设置图例
    ax.legend()

    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"可视化结果已保存为 {save_path}")

    # 关闭图形，避免显示
    plt.close()


# 示例使用
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose_file", "-p", type=str, default=False, help= "file direction for pose")

    pose_file_dir = parser.parse_args().pose_file
    original_pose = load_pose(pose_file_dir)

    print("original_pose", original_pose)
    print("original_pose_type", original_pose.type(torch.float32))
    print("original_pose shape", original_pose.size())

    # 假设两个相机的初始位姿
    # R1 = np.eye(3)  # 第一个相机的旋转矩阵
    # t1 = np.array([0, 0, -5])  # 第一个相机的平移向量
    # R2 = np.eye(3)  # 第二个相机的旋转矩阵
    # t2 = np.array([5, 0, -5])  # 第二个相机的平移向量

    # 获取轨道和视角
    # poses, ellipse_points, center = get_new_pose(original_pose, num_views=14)

    # # 打印结果
    # for i, (R_new, t_new) in enumerate(poses):
    #     print(f"View {i + 1}:")
    #     print("Rotation Matrix:\n", R_new)
    #     print("Translation Vector:\n", t_new)
    #     print()
    #
    # # 读取点云
    # point_cloud = o3d.io.read_point_cloud("data/prompts/points3D.ply")
    # if not point_cloud.has_points():
    #     print("点云为空，请检查文件是否正确。")
    # points = np.asarray(point_cloud.points)
    # colors = np.asarray(point_cloud.colors)
    # # 可视化椭圆轨道、相机位姿和点云，并保存为图片
    # save_path = 'DynamiCrafter/main/res.png'
    # # 可视化椭圆轨道、相机位姿和点云，并保存为图片
    # visualize_orbit(
    #     ellipse_points, center, poses,
    #     point_cloud=points, point_colors=colors,
    #     save_path=save_path
    # )
    new_poses = get_new_pose(original_pose)
    print("new_pose:\n", new_poses.shape)