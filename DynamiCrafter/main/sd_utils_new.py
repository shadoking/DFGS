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
from lvdm.modules.encoders.condition import PointNetEncoder
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

def get_nondefault_trainer_args(args):
    parser = argparse.ArgumentParser()
    default_trainer_args = parser.parse_args([])
    return sorted(k for k in vars(default_trainer_args) if getattr(args, k) != getattr(default_trainer_args, k))

def get_latent_z(model, videos):
    b, c, t, h, w = videos.shape
    x = rearrange(videos, 'b c t h w -> (b t) c h w')
    z = model.encode_first_stage(x)
    z = rearrange(z, '(b t) c h w -> b c t h w', b=b, t=t)
    return z

class DiffusionModule(nn.Module):
    def __init__(self, parser, config_dir, save_dir, prompt_dir, epochs=500, save_every_n_epoch=100, device="cuda"):
        super().__init__()
        self.save_dir = save_dir
        self.prompt_dir = prompt_dir
        self.device = device
        self.epochs = epochs
        self.save_every_n_epoch = save_every_n_epoch
        
        now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        
        ## Extends existing argparse by default Trainer attributes
        args, unknown = parser.parse_known_args()
        ## disable transformer warning
        transf_logging.set_verbosity_error()
        seed_everything(args.seed)
        ## yaml configs: "model" | "data" | "lightning"
        configs = [OmegaConf.load(cfg) for cfg in config_dir]
        cli = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*configs, cli)
        
        lightning_config = config.pop("lightning", OmegaConf.create())
        trainer_config = lightning_config.get("trainer", OmegaConf.create()) 
        workdir, ckptdir, cfgdir, loginfo = init_workspace(args.name, args.logdir, config, lightning_config, 0)
        logger = set_logger(logfile=os.path.join(loginfo, 'log_%s.txt'%(now)))
        
        ## 解析配置
        args, unknown = parser.parse_known_args()
        seed = args.seed if hasattr(args, "seed") else 42
        torch.manual_seed(seed)

        configs = [OmegaConf.load(cfg) for cfg in config_dir]
        cli = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*configs, cli)

        lightning_config = config.pop("lightning", OmegaConf.create())
        trainer_config = lightning_config.get("trainer", OmegaConf.create()) 
        
        logger.info("***** Configing Model *****")
        config.model.params.logdir = workdir
        model = instantiate_from_config(config.model)
        model = load_checkpoints(model, config.model)
        if model.rescale_betas_zero_snr:
            model.register_schedule(given_betas=model.given_betas, beta_schedule=model.beta_schedule, timesteps=model.timesteps,
                                linear_start=model.linear_start, linear_end=model.linear_end, cosine_s=model.cosine_s)

        ## update trainer config
        for k in get_nondefault_trainer_args(args):
            trainer_config[k] = getattr(args, k)
            
        ## setup learning rate
        base_lr = config.model.base_learning_rate
        model.learning_rate = base_lr
        
        self.model = model.to(device)
             
    def training_step(self):
        loss, loss_dict = self.shared_step()
        #self.model.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=False)
        return loss
    
    def shared_step(self, **kwargs):
        x, c, poses, fs = self.get_batch_input(return_fs=True)
        # kwargs.update({"fs": fs.long()})
        loss, loss_dict = self.model(x, c, poses, **kwargs)
        return loss, loss_dict
        
        
    def get_batch_input(self, return_fs=False, return_pose=True, fs=3):
        filename_list, data_list, prompt_list, pointcloud_list, pose_list = self.get_data()
        prompts = prompt_list[0] 
        videos = data_list[0]
        # filenames = filename_list[0]
        pointcloud = pointcloud_list[0]
        pose = pose_list[0] #(1, 2, 7)
        
        if isinstance(videos, list):
            videos = torch.stack(videos, dim=0).to("cuda")
        else:
            videos = videos.unsqueeze(0).to("cuda")
            
        fs = torch.tensor([fs], dtype=torch.long, device=self.device)
        
        img = videos[:,:,0]  #bchw
        img_emb = self.model.embedder(img)  ## blc
        img_emb = self.model.image_proj_model(img_emb)
        
        pc_emb = self.model.pc_embedder(pointcloud)
        
        ## TODO 根据首尾帧POSE以及帧数预测出全部
        # (16, 7)
        all_poses = self.get_new_pose(pose) # (16, 7)形状的所有位姿，前两个是原始位姿 pose是初始的两个位姿
        
        cond_emb = self.model.get_learned_conditioning(prompts)
        cond = {"c_crossattn": [torch.cat([cond_emb, img_emb, pc_emb], dim=1)]}
        # cond = {"c_crossattn": [torch.cat([cond_emb, img_emb], dim=1)]}
        if self.model.model.conditioning_key == 'hybrid':
            z = get_latent_z(self.model, videos) # b c t h w
            img_cat_cond = torch.zeros_like(z)
            img_cat_cond[:,:,0,:,:] = z[:,:,0,:,:]
            img_cat_cond[:,:,-1,:,:] = z[:,:,-1,:,:]
            
            cond["c_concat"] = [img_cat_cond] # b c 1 h w [1, 4, 16, 64, 64]

        out = [z, cond]
        
        if return_pose:
            out.append(pose)
            
        if return_fs:
            out.append(fs)
        
        return out

    # new_poses return (16, 7) 固定椭圆 没有限制在一个椭圆轨道之内
    # def get_new_pose(self, original_pose, num_views=14):
    #     """
    #     根据两个初始相机的位姿生成一个椭圆轨道，并在轨道上生成新的视角和位姿。
    #
    #     参数:
    #         original_pose (torch.Tensor 或 np.ndarray): 包含两个相机位姿的张量 (1, 2, 7)。
    #         num_views (int): 需要生成的新视角数量，默认为14。
    #
    #     返回:
    #         torch.Tensor: 包含每个新视角的位姿 (R_new, t_new) 的张量，形状为 (16, 7)。
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
    #     # 椭圆参数
    #     center = (C1 + C2) / 2  # 椭圆中心 (3,1)
    #     a = np.linalg.norm(C1 - C2) / 2  # 长轴
    #     b = a / 2  # 短轴
    #
    #     # 生成椭圆轨道上的点
    #     angles = np.linspace(0, 2 * np.pi, num_views)  # 均匀采样角度
    #     ellipse_points = np.array([a * np.cos(angles), b * np.sin(angles), np.zeros(num_views)]).T  # (14,3)
    #
    #     # 处理广播问题 (转换 center 为 (1,3) 以进行广播)
    #     center = center.reshape(1, 3)  # 确保形状匹配
    #     ellipse_points += center  # (14,3) + (1,3) 可广播
    #
    #     # 计算每个点的位姿
    #     poses = []
    #     for point in ellipse_points:
    #         # 相机位置
    #         C_new = point.reshape(3, 1)  # 转换为 (3,1)
    #
    #         # 相机朝向场景中心
    #         look_at = center.T  # 变成 (3,1)
    #         forward = look_at - C_new
    #         forward = forward / np.linalg.norm(forward)
    #
    #         # 计算旋转矩阵
    #         up = np.array([0, 1, 0])  # 假设上方向为 Y 轴
    #         right = np.cross(up, forward.squeeze())  # 计算右方向
    #         right = right / np.linalg.norm(right)
    #         up = np.cross(forward.squeeze(), right)  # 重新计算 up
    #
    #         R_new = np.vstack((right, up, -forward.squeeze())).T  # 计算新的旋转矩阵
    #         # 强制保证旋转矩阵是右手坐标系
    #         if np.linalg.det(R_new) < 0:
    #             R_new = -R_new
    #
    #         # 计算新的平移向量
    #         t_new = -R_new @ C_new
    #
    #         # 将旋转矩阵转换为四元数
    #         q_new = R.from_matrix(R_new).as_quat()  # (4,)
    #
    #         # 将四元数和平移向量合并为一个 7 维向量
    #         pose_new = np.concatenate([q_new, t_new.squeeze()])  # (7,)
    #
    #         poses.append(pose_new)
    #
    #     # 将第一个和最后一个视角添加为原始位姿
    #     q1_t = np.concatenate([q1, t1.squeeze()])  # (7,)
    #     q2_t = np.concatenate([q2, t2.squeeze()])  # (7,)
    #
    #     poses.insert(0, q1_t)  # 在最前面添加原始第一个相机位姿
    #     poses.append(q2_t)  # 在最后添加原始第二个相机位姿
    #
    #     # 转换为 torch.Tensor 并返回
    #     poses_tensor = torch.tensor(poses, dtype=torch.float32)
    #
    #     return poses_tensor

    def get_data(self, height=512, width=512, n_frames=16):
        os.makedirs(self.save_dir, exist_ok=True)
        
        assert os.path.exists(self.prompt_dir)
        filename_list, data_list, prompt_list, pointcloud_list, pose_list = self.load_data_prompts(self.prompt_dir, video_size=(height, width), video_frames=n_frames, interp=True)

        return filename_list, data_list, prompt_list, pointcloud_list, pose_list

    def load_data_prompts(self, data_dir, video_size=(256,256), video_frames=16, interp=False):
        transform = transforms.Compose([
            transforms.Resize(min(video_size)),
            transforms.CenterCrop(video_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
        ## load prompts
        prompt_file = self.get_filelist(data_dir, ['txt'])
        assert len(prompt_file) > 0, "Error: found No prompt file!"
        ###### default prompt
        default_idx = 0
        default_idx = min(default_idx, len(prompt_file)-1)
        if len(prompt_file) > 1:
            print(f"Warning: multiple prompt files exist. The one {os.path.split(prompt_file[default_idx])[1]} is used.")
        ## only use the first one (sorted by name) if multiple exist
        
        ## load video
        file_list = self.get_filelist(data_dir, ['jpg', 'png', 'jpeg', 'JPEG', 'PNG', 'JPG'])
        # assert len(file_list) == n_samples, "Error: data and prompts are NOT paired!"
        data_list = []
        filename_list = []
        pointcloud_list = []
        pose_list = []
        prompt_list = self.load_prompts(prompt_file[default_idx])
        n_samples = len(prompt_list)
        for idx in range(n_samples):
            if interp:
                image1 = Image.open(file_list[2*idx]).convert('RGB')
                image_tensor1 = transform(image1).unsqueeze(1) # [c,1,h,w]
                image2 = Image.open(file_list[2*idx+1]).convert('RGB')
                image_tensor2 = transform(image2).unsqueeze(1) # [c,1,h,w]
                frame_tensor1 = repeat(image_tensor1, 'c t h w -> c (repeat t) h w', repeat=video_frames//2)
                frame_tensor2 = repeat(image_tensor2, 'c t h w -> c (repeat t) h w', repeat=video_frames//2)
                frame_tensor = torch.cat([frame_tensor1, frame_tensor2], dim=1)
                _, filename = os.path.split(file_list[idx*2])
            else:
                image = Image.open(file_list[idx]).convert('RGB')
                image_tensor = transform(image).unsqueeze(1) # [c,1,h,w]
                frame_tensor = repeat(image_tensor, 'c t h w -> c (repeat t) h w', repeat=video_frames)
                _, filename = os.path.split(file_list[idx])

            data_list.append(frame_tensor)
            filename_list.append(filename)
        pointcloud_tensor = self.load_pointcloud(os.path.join(data_dir, "points3D.ply"))
        pose_tensor = self.load_pose(os.path.join(data_dir, "images.bin"))
        pointcloud_list.append(pointcloud_tensor)
        pose_list.append(pose_tensor)
        return filename_list, data_list, prompt_list, pointcloud_list, pose_list

    def load_pointcloud(self, ply_file, sample_ratio=1/8):
        plydata = PlyData.read(ply_file)
        vertices = plydata['vertex']
        positions = torch.tensor(np.vstack([vertices['x'], vertices['y'], vertices['z']]).T, dtype=torch.float32)                # [N,3]
        colors = torch.tensor(np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0, dtype=torch.float32)  # [N,3]
        total_points = positions.size(0)
        sample_size = int(total_points * sample_ratio)
        indices = torch.randperm(total_points)[:sample_size]
        sampled_positions = positions[indices]
        sampled_colors = colors[indices]
        
        pointcloud = torch.cat([sampled_positions, sampled_colors], dim=1) # [N, 6]

        return  pointcloud.unsqueeze(0) # [1, N, 6]
    
    def load_pose(self, pose_file):
        images = read_images_binary(pose_file)
        poses = []
    
        for _, data in images.items():
            qvec = data.qvec
            tvec = data.tvec
            
            pose = np.concatenate([qvec, tvec])
            poses.append(pose)
                
        poses_np = np.array(poses, dtype=np.float32)  # 转为 NumPy 数组
        poses_tensor = torch.from_numpy(poses_np)  
        return poses_tensor.unsqueeze(0).to(self.device) # [2, 7]
    
    def load_prompts(self, prompt_file):
        f = open(prompt_file, 'r')
        prompt_list = []
        for idx, line in enumerate(f.readlines()):
            l = line.strip()
            if len(l) != 0:
                prompt_list.append(l)
            f.close()
        return prompt_list
    
    def get_filelist(self, data_dir, postfixes):
        patterns = [os.path.join(data_dir, f"*.{postfix}") for postfix in postfixes]
        file_list = []
        for pattern in patterns:
            file_list.extend(glob.glob(pattern))
        file_list.sort()
        return file_list
    
    def configure_optimizers(self):
        lr = self.model.learning_rate
        params = list(self.model.model.parameters())
        if self.model.image_proj_model_trainable:
            params.extend(list(self.model.image_proj_model.parameters()))
        
        optimizer = torch.optim.AdamW(params, lr=lr) 
        return optimizer        
            
    def train(self):
        os.makedirs(self.save_dir, exist_ok=True)
        
        optimizer = self.configure_optimizers()
        scaler = torch.amp.GradScaler('cuda')
        # start_time = time.time() 
        best_loss = float("inf") 
        pbar = tqdm(range(1, self.epochs+1), desc="Training Progress", unit="epoch")
        for epoch in pbar:
            self.model.train()
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                loss = self.training_step()
                
            scaler.scale(loss).backward()
    
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
             # **计算剩余时间**
            # elapsed_time = time.time() - start_time
            # avg_time_per_epoch = elapsed_time / (epoch + 1)
            # remaining_time = avg_time_per_epoch * (self.epochs - epoch - 1)

            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(self.model.state_dict(), os.path.join(save_dir, "best_model.ckpt"))
            
            if epoch % self.save_every_n_epoch == 0:
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, f"model_{epoch}.ckpt"))

            # print(f"Epoch {epoch+1}/{self.epochs} | Loss: {loss.item():.4f} | Remaining Time: {remaining_time:.2f}s")
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "best_loss": f"{best_loss:.4f}"
            })
            
        #torch.save(self.model.state_dict(), os.path.join(save_dir, "last.ckpt"))
        print("Done!")
    
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", "-s", type=int, default=20230211, help="seed for seed_everything")
    parser.add_argument("--name", "-n", type=str, default="", help="experiment name, as saving folder")

    parser.add_argument("--base", "-b", nargs="*", metavar="base_config.yaml", help="paths to base configs. Loaded from left-to-right. "
                            "Parameters can be overwritten or added with command-line options of the form `--key value`.", default=list())
    
    parser.add_argument("--train", "-t", action='store_true', default=False, help='train')
    parser.add_argument("--val", "-v", action='store_true', default=False, help='val')
    parser.add_argument("--test", action='store_true', default=False, help='test')

    parser.add_argument("--logdir", "-l", type=str, default="logs", help="directory for logging dat shit")
    parser.add_argument("--auto_resume", action='store_true', default=False, help="resume from full-info checkpoint")
    parser.add_argument("--auto_resume_weight_only", action='store_true', default=False, help="resume from weight-only checkpoint")
    parser.add_argument("--debug", "-d", action='store_true', default=False, help="enable post-mortem debugging")
    
    name="training_512_v1.0"
    config_file="configs/" + name + "/config_interp.yaml"
    save_dir = "checkpoints"
    data_dir = "../data/prompts"

    diffusionModule = DiffusionModule(parser, [config_file], save_dir, data_dir, epochs=20, save_every_n_epoch=20)
    diffusionModule.train()