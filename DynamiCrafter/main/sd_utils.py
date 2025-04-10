import argparse, os, sys, datetime, glob

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from omegaconf import OmegaConf
import torch
import torch.nn as nn
import numpy as np
from transformers import logging as transf_logging
from pytorch_lightning import seed_everything
from utils.utils import instantiate_from_config
from lvdm.models.samplers.ddim import DDIMSampler
from main.utils_train import set_logger, init_workspace, load_checkpoints
from plyfile import PlyData
from einops import rearrange, repeat
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from read_write_model import read_images_binary
import torchvision




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

def save_results_seperate(prompt, samples, filename, fakedir, fps=10, loop=False):
    prompt = prompt[0] if isinstance(prompt, list) else prompt

    ## save video
    videos = [samples]
    savedirs = [fakedir]
    for idx, video in enumerate(videos):
        if video is None:
            continue
        # b,c,t,h,w
        video = video.detach().cpu()
        if loop: # remove the last frame
            video = video[:,:,:-1,...]
        video = torch.clamp(video.float(), -1., 1.)
        n = video.shape[0]
        for i in range(n):
            grid = video[i,...]
            grid = (grid + 1.0) / 2.0
            grid = (grid * 255).to(torch.uint8).permute(1, 2, 3, 0) #thwc
            path = os.path.join(savedirs[idx].replace('samples', 'samples_separate'), f'{filename.split(".")[0]}_sample{i}.mp4')
            torchvision.io.write_video(path, grid, fps=fps, video_codec='h264', options={'crf': '10'})
    print("Saved!")

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
        logger = set_logger(logfile=os.path.join(loginfo, 'log_%s.txt' % (now)))

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
            model.register_schedule(given_betas=model.given_betas, beta_schedule=model.beta_schedule,
                                    timesteps=model.timesteps,
                                    linear_start=model.linear_start, linear_end=model.linear_end,
                                    cosine_s=model.cosine_s)

        ## update trainer config
        for k in get_nondefault_trainer_args(args):
            trainer_config[k] = getattr(args, k)

        ## setup learning rate
        base_lr = config.model.base_learning_rate
        model.learning_rate = base_lr

        self.model = model.to(device)

    def training_step(self):
        loss, loss_dict = self.shared_step()
        # self.model.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=False)
        return loss, loss_dict
    
    def shared_step(self, **kwargs):
        x, c, fs = self.get_batch_input(return_fs=True)
        # kwargs.update({"fs": fs.long()})
        loss, loss_dict = self.model(x, c, **kwargs)
        return loss, loss_dict

    def get_batch_input(self, return_fs=False, fs=3, height=512, width=512, n_frames=16):
        os.makedirs(self.save_dir, exist_ok=True)

        assert os.path.exists(self.prompt_dir)
        filename_list, data_list, prompt_list, pointcloud_list, data_all_list, fn = self.load_data_prompts(self.prompt_dir,
                                                                            video_size=(
                                                                            height, width),
                                                                            video_frames=n_frames,
                                                                            interp=True)
        prompts = prompt_list[0]
        images = data_list[0]
        pointcloud = pointcloud_list[0]
        images_all = data_all_list[0]

        if isinstance(images, list):
            videos = torch.stack(images, dim=0).to("cuda")
        else:
            videos = images.unsqueeze(0).to("cuda") #[1,3,16,512,512]
            videos_all = images_all.unsqueeze(0).to("cuda") #[1,3,16,512,512]
        fs = torch.tensor([fs], dtype=torch.long, device=self.device)

        # mid_idx = videos_all.shape[2] // 2  # 中间帧索引
        # img_first = videos_all[:, :, 0]  # 首帧
        # img_mid = videos_all[:, :, mid_idx]  # 中间帧
        # img_last = videos_all[:, :, -1] 
        
        # img_emb_first = self.model.embedder(img_first)  # [B, L, C]
        # img_emb_mid = self.model.embedder(img_mid)
        # img_emb_last = self.model.embedder(img_last)

        # img_emb_first = self.model.image_proj_model(img_emb_first)
        # img_emb_mid = self.model.image_proj_model(img_emb_mid)
        # img_emb_last = self.model.image_proj_model(img_emb_last)
        # img_emb = torch.cat([img_emb_first, img_emb_mid, img_emb_last], dim=1)  # [B, 3L, C]
        
        img = videos[:, :, 0]  # bchw  [1,3,512,512]
        img_emb = self.model.embedder(img)  ## blc
        img_emb = self.model.image_proj_model(img_emb)

        pc_emb = self.model.pc_embedder(pointcloud)
        
        cond_emb = self.model.get_learned_conditioning(prompts)
        cond = {"c_crossattn": [torch.cat([cond_emb, img_emb, pc_emb], dim=1)]}
        # cond = {"c_crossattn": [torch.cat([cond_emb, img_emb], dim=1)]}
        
        # masks = self.generate_mask(images_all)
        # with torch.no_grad():
        #     masks_z = get_latent_z(self.model, masks)

        if self.model.model.conditioning_key == 'hybrid':
            z = get_latent_z(self.model, videos)  # b c t h w
            # masks_z = get_latent_z(self.model, masks)
            img_cat_cond = torch.zeros_like(z)
            
            # if fn > 2:
            #     step = z.shape[2] // fn
            #     for i in range(1, fn - 1):
            #         idx = i * step
            #         img_cat_cond[:, :, idx, :, :] = z[:, :, idx, :, :] * (1 - masks_z[:, :, idx, :, :])
            img_cat_cond[:, :, 0, :, :] = z[:, :, 0, :, :]
            img_cat_cond[:, :, -1, :, :] = z[:, :, -1, :, :]
            cond["c_concat"] = [img_cat_cond]  # b c 1 h w [1, 4, 16, 64, 64]

        # z = z * (1 - masks_z)
        out = [z, cond]

        if return_fs:
            out.append(fs)

        return out

    def generate_mask(self, image, threshold=0.1):
        mean_image = image.mean(dim=(0, 2, 3), keepdim=True)  # [c, 1, 1, 1]
        edge_map = torch.abs(image - mean_image)  # [c, t, h, w]
        mask = (edge_map.mean(dim=0, keepdim=True) < threshold).float()  # [1, t, h, w]
        mask = mask.expand(image.shape)  # [c, t, h, w]，让 mask 复制到所有通道
        return mask.unsqueeze(0).to("cuda")
    
    
    def load_data_prompts(self, data_dir, video_size=(256, 256), video_frames=16, interp=False):
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
        default_idx = min(default_idx, len(prompt_file) - 1)
        if len(prompt_file) > 1:
            print(
                f"Warning: multiple prompt files exist. The one {os.path.split(prompt_file[default_idx])[1]} is used.")
        ## only use the first one (sorted by name) if multiple exist

        ## load video
        file_list = self.get_filelist(data_dir, ['jpg', 'png', 'jpeg', 'JPEG', 'PNG', 'JPG'])
        # assert len(file_list) == n_samples, "Error: data and prompts are NOT paired!"
        data_list = []
        filename_list = []
        data_all_list = []
        pointcloud_list = []
        prompt_list = self.load_prompts(prompt_file[default_idx])
        n_samples = len(prompt_list)
        for idx in range(n_samples):
            if interp:
                image_tensors = []
                fn = len(file_list)
                for img_file in file_list:
                    image = Image.open(img_file).convert('RGB')
                    image_tensor = transform(image).unsqueeze(1)  # [c,1,h,w]
                    image_tensors.append(image_tensor)
                    
                frame_tensors = [
                    repeat(img, 'c t h w -> c (repeat t) h w', repeat=video_frames // fn)
                    for img in image_tensors
                ]
                
                frame_tensor = torch.cat(frame_tensors, dim=1) 
                data_all_list.append(frame_tensor)
               
                image1 = Image.open(file_list[0]).convert('RGB')
                image_tensor1 = transform(image1).unsqueeze(1)  # [c,1,h,w]
                # image2 = Image.open(file_list[2 * idx + 1]).convert('RGB')
                image2 = Image.open(file_list[-1]).convert('RGB')
                image_tensor2 = transform(image2).unsqueeze(1)  # [c,1,h,w]
                
                frame_tensor1 = repeat(image_tensor1, 'c t h w -> c (repeat t) h w', repeat=video_frames // 2)
                frame_tensor2 = repeat(image_tensor2, 'c t h w -> c (repeat t) h w', repeat=video_frames // 2)
                frame_tensor = torch.cat([frame_tensor1, frame_tensor2], dim=1)
                _, filename = os.path.split(file_list[idx*2])
            else:
                image = Image.open(file_list[idx]).convert('RGB')
                image_tensor = transform(image).unsqueeze(1)  # [c,1,h,w]
                frame_tensor = repeat(image_tensor, 'c t h w -> c (repeat t) h w', repeat=video_frames)
                _, filename = os.path.split(file_list[idx])

            data_list.append(frame_tensor)
        pointcloud_tensor = self.load_pointcloud(os.path.join(data_dir, "points3D.ply"))
        pointcloud_list.append(pointcloud_tensor)
        filename_list.append(filename)
        return filename_list, data_list, prompt_list, pointcloud_list, data_all_list, fn

    def load_pointcloud(self, ply_file, sample_ratio=1 / 8):
        plydata = PlyData.read(ply_file)
        vertices = plydata['vertex']
        positions = torch.tensor(np.vstack([vertices['x'], vertices['y'], vertices['z']]).T,
                                 dtype=torch.float32)  # [N,3]
        colors = torch.tensor(np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0,
                              dtype=torch.float32)  # [N,3]
        total_points = positions.size(0)
        sample_size = int(total_points * sample_ratio)
        indices = torch.randperm(total_points)[:sample_size]
        sampled_positions = positions[indices]
        sampled_colors = colors[indices]

        pointcloud = torch.cat([sampled_positions, sampled_colors], dim=1)  # [N, 6]

        return pointcloud.unsqueeze(0)  # [1, N, 6]

    def load_pose(self, pose_file):
        images = read_images_binary(pose_file)
        poses = []

        for _, data in images.items():
            qvec = data.qvec
            tvec = data.tvec

            pose = np.concatenate([qvec, tvec])
            poses.append(pose)

        poses_np = np.array(poses, dtype=np.float32) 
        poses_tensor = torch.from_numpy(poses_np)
        return poses_tensor.unsqueeze(0).to(self.device)  # [2, 7]

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
        # best_loss = float("inf")
        pbar = tqdm(range(1, self.epochs + 1), desc="Training Progress", unit="epoch")
        self.model.train()
        prefix = 'train' if self.training else 'val'
        for epoch in pbar:
            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                loss, loss_dict = self.training_step()
                
            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # if loss.item() < best_loss:
            #     best_loss = loss.item()
            #     torch.save(self.model.state_dict(), os.path.join(save_dir, "best_model.ckpt"))

            if epoch % self.save_every_n_epoch == 0:
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, f"model_{epoch}.ckpt"))

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}"
                # "pose_loss": f"{loss_dict.get(f'{prefix}/loss_pose', torch.tensor(0.0)).item():.4f}"
            })

        # torch.save(self.model.state_dict(), os.path.join(save_dir, "last.ckpt"))
        print("Done!")
        
    def inference(self, save_path, height=512, width=512, fs=3, n_frames=16, ddim_steps=50, ddim_eta=1., **kwargs):
        self.model.eval()
        h, w = height // 8, width // 8
        filename_list, data_list, prompt_list, pointcloud_list, data_all_list, _ = self.load_data_prompts(self.prompt_dir,
                                                                            video_size=(
                                                                            height, width),
                                                                            video_frames=n_frames,
                                                                            interp=True)
        prompts = prompt_list[0]
        images = data_list[0]
        # images = data_all_list[0]
        pointcloud = pointcloud_list[0]
        images_all = data_all_list[0]
        
        channels = self.model.model.diffusion_model.out_channels
        noise_shape = [1, channels, n_frames, h, w]
        batch_size = noise_shape[0]
       
        
        with torch.no_grad(), torch.amp.autocast('cuda'):
            if isinstance(images, list):
                videos = torch.stack(images, dim=0).to("cuda")
            else:
                videos = images.unsqueeze(0).to("cuda") #[1,3,16,512,512]
                videos_all = images_all.unsqueeze(0).to("cuda") #[1,3,16,512,512]
            
            fs = torch.tensor([fs] * batch_size, dtype=torch.long, device=self.model.device)
            ddim_sampler = DDIMSampler(self.model)
            
            img = videos[:,:,0] # bchw
            img_emb = self.model.embedder(img) ## blc
            img_emb = self.model.image_proj_model(img_emb)
            
            cond_emb = self.model.get_learned_conditioning(prompts)
            cond = {"c_crossattn": [torch.cat([cond_emb, img_emb], dim=1)]}
            
            z = get_latent_z(self.model, videos)
            img_cat_cond = torch.zeros_like(z)
            img_cat_cond[:,:,0,:,:] = z[:,:,0,:,:]
            img_cat_cond[:,:,-1,:,:] = z[:,:,-1,:,:]
            cond["c_concat"] = [img_cat_cond]

            cond_z0 = None
            cond_mask = None
            kwargs.update({"unconditional_conditioning_img_nonetext": None})

            batch_variants = []
            samples, _ = ddim_sampler.sample(S=ddim_steps,
                                            conditioning=cond,
                                            batch_size=batch_size,
                                            shape=noise_shape[1:],
                                            verbose=False,
                                            unconditional_guidance_scale=1.0,
                                            unconditional_conditioning=None,
                                            eta=ddim_eta,
                                            cfg_img=None, 
                                            mask=cond_mask,
                                            x0=cond_z0,
                                            fs=fs,
                                            timestep_spacing="uniform",
                                            guidance_rescale=0.0,
                                            **kwargs
                                            )
            
            batch_images = self.model.decode_first_stage(samples)
            batch_variants.append(batch_images)
            batch_variants = torch.stack(batch_variants)
            batch_samples = batch_variants.permute(1, 0, 2, 3, 4, 5)
            for nn, samples in enumerate(batch_samples):
                prompt = prompt_list[nn]
                filename = filename_list[nn]
                save_results_seperate(prompt, samples, filename, save_path, fps=8, loop=False)

        
    def decode_and_save_video(self, save_path="final_video.mp4", fps=8):
        z, _, _ = self.get_batch_input(return_fs=True)
        video = self.model.decode_first_stage(z).detach().cpu()
        video = torch.clamp(video, -1., 1.)
        video = (video + 1) / 2
        video = (video * 255).to(torch.uint8)  # [1, 3, T, H, W]
        video = video[0].permute(1, 2, 3, 0)  # [T, H, W, C]
        torchvision.io.write_video(save_path, video, fps=fps, video_codec='h264', options={"crf": "10"})
        print(f"Saved video to: {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", "-s", type=int, default=20230211, help="seed for seed_everything")
    parser.add_argument("--name", "-n", type=str, default="", help="experiment name, as saving folder")

    parser.add_argument("--base", "-b", nargs="*", metavar="base_config.yaml",
                        help="paths to base configs. Loaded from left-to-right. "
                             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
                        default=list())

    parser.add_argument("--train", "-t", action='store_true', default=False, help='train')
    parser.add_argument("--val", "-v", action='store_true', default=False, help='val')
    parser.add_argument("--test", action='store_true', default=False, help='test')

    parser.add_argument("--logdir", "-l", type=str, default="logs", help="directory for logging dat shit")
    parser.add_argument("--auto_resume", action='store_true', default=False, help="resume from full-info checkpoint")
    parser.add_argument("--auto_resume_weight_only", action='store_true', default=False,
                        help="resume from weight-only checkpoint")
    parser.add_argument("--debug", action='store_true', default=False, help="enable post-mortem debugging")
    parser.add_argument("--data_dir", "-d", type=str, default="../data/prompts")

    name = "training_512_v1.0"
    config_file = "configs/" + name + "/config_interp.yaml"
    save_dir = "checkpoints"
   # data_dir = "../data/prompts"

    args = parser.parse_args()

    diffusionModule = DiffusionModule(parser, [config_file], save_dir, args.data_dir, epochs=10, save_every_n_epoch=10)
    diffusionModule.train()
    diffusionModule.inference("results")