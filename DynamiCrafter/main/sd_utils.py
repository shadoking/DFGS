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
    def __init__(self, parser, config_dir, save_dir, prompt_dir, device="cuda"):
        super().__init__()
        self.save_dir = save_dir
        self.prompt_dir = prompt_dir
        self.device = device
        
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
        
        self.model.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=False)
        
        print(f"epoch:{self.current_epoch} [globalstep:{self.global_step}]: loss={loss}")
        return loss
    
    def shared_step(self, **kwargs):
        x, c, fs = self.get_batch_input(return_fs=True)
        kwargs.update({"fs": fs.long()})
        loss, loss_dict = self.model(x, c, **kwargs)
        return loss, loss_dict
        
        
    def get_batch_input(self, return_fs=False, fs=3):
        filename_list, data_list, prompt_list, pointcloud_list = self.get_data()
        prompts = prompt_list[0] 
        videos = data_list[0]
        filenames = filename_list[0]
        pointcloud = pointcloud_list[0]
        
        if isinstance(videos, list):
            videos = torch.stack(videos, dim=0).to("cuda")
        else:
            videos = videos.unsqueeze(0).to("cuda")
            
        fs = torch.tensor([fs], dtype=torch.long, device=self.device)
        
        img = videos[:,:,0]  #bchw
        img_emb = self.model.embedder(img)  ## blc
        img_emb = self.model.image_proj_model(img_emb)
        
        #pc_emb = self.model.pc_embedder(pointcloud)
        
        cond_emb = self.model.get_learned_conditioning(prompts)
        #cond = {"c_crossattn": [torch.cat([cond_emb, img_emb, pc_emb], dim=1)]}
        cond = {"c_crossattn": [torch.cat([cond_emb, img_emb], dim=1)]}
        if self.model.model.conditioning_key == 'hybrid':
            z = get_latent_z(self.model, videos) # b c t h w
            img_cat_cond = torch.zeros_like(z)
            img_cat_cond[:,:,0,:,:] = z[:,:,0,:,:]
            img_cat_cond[:,:,-1,:,:] = z[:,:,-1,:,:]
            
            cond["c_concat"] = [img_cat_cond] # b c 1 h w [1, 4, 16, 64, 64]


        out = [z, cond]
        
        if return_fs:
            out.append(fs)
        
        return out
        
    def get_data(self, height=512, width=512, n_frames=16):
        os.makedirs(self.save_dir, exist_ok=True)
        
        assert os.path.exists(self.prompt_dir)
        filename_list, data_list, prompt_list, pointcloud_list = self.load_data_prompts(self.prompt_dir, video_size=(height, width), video_frames=n_frames, interp=True)

        return filename_list, data_list, prompt_list, pointcloud_list

    def load_data_prompts(self, data_dir, video_size=(256,256), video_frames=16, interp=False):
        transform = transforms.Compose([
            transforms.Resize(min(video_size)),
            transforms.CenterCrop(video_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
        ## load prompts
        prompt_file = self.get_filelist(data_dir, ['txt'])
        assert len(prompt_file) > 0, "Error: found NO prompt file!"
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
        pointcloud_list.append(pointcloud_tensor)
        return filename_list, data_list, prompt_list, pointcloud_list

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
        
    def train_step(
        self,
        pred_rgb,
        step_ratio=None,
        guidance_scale=100,
        as_latent=False
    ):
        batch_size = pred_rgb.shape[0]
        pred_rgb = pred_rgb.to(self.weights_dtype)
        
        if as_latent:
            latents = F.interpolate(pred_rgb, (64, 64), mode="bilinear", align_corners=False) * 2 - 1
        else:
            pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode="bilinear", align_corners=False)
            latents = self.encode_imgs(pred_rgb_512)
            
        if step_ratio is not None:
            t = np.round((1 - step_ratio) * self.num_train_timesteps).clip(self.min_step, self.max_step)
            t = torch.full((batch_size,), t, dtype=torch.long, device=self.device)
        else:
            t = torch.randint(self.min_step, self.max_step + 1, (batch_size, ), dtype=torch.long, device=self.device)
        
        w = (1 - self.alphas[t]).view(batch_size, 1, 1, 1)
        
        with torch.no_grad():
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            
            latent_model_input = torch.cat([latents_noisy] * 2)
            tt = torch.cat([t] * 2)
            
            noise_pred = self.unet(
                latent_model_input, tt, encoder_hidden_states=self.embeddings.repeat(batch_size, 1, 1)
            ).sample
            
            noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_pos - noise_pred_uncond
            )
        
        grad = w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)
        
        target = (latents - grad).detach()
        loss = 0.5 * F.mse_loss(latents.float(), target, reduction='sum') / latents.shape[0]

        return loss
    
    
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
    
    print(config_file)


    diffusionModule = DiffusionModule(parser, [config_file], save_dir, data_dir)
    with torch.amp.autocast('cuda'):
        diffusionModule.training_step()