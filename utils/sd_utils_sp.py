import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from diffusers import (
    StableDiffusionPipeline,
    DDPMScheduler
)

@dataclass
class Config:
    pretrained_model_path_sd = "./pretrained_models/stable-diffusion-1-5"
    half_precision_weights: bool = True
    t_range = [0.02, 0.98]
    
class StableDiffusion(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.config = Config()
        
        self.device = torch.device("cuda")
        self.weights_dtype = torch.float16 if self.config.half_precision_weights else torch.float32
        
        pipe_sd = StableDiffusionPipeline.from_pretrained(
            self.config.pretrained_model_path_sd,
            torch_dtype=self.weights_dtype
        ).to(self.device)
        
        self.vae = pipe_sd.vae
        self.tokenizer = pipe_sd.tokenizer
        self.text_encoder = pipe_sd.text_encoder
        self.unet = pipe_sd.unet
        
        self.scheduler = DDPMScheduler.from_pretrained(
            self.config.pretrained_model_path_sd,
            subfolder='scheduler',
            torch_dtype=self.weights_dtype,
        )
        
        del pipe_sd
        
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * self.config.t_range[0])
        self.max_step = int(self.num_train_timesteps * self.config.t_range[1])
        
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)
        
        self.embeddings = None
        
    def encode_txt(self, prompt):
        inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt"
        )
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]
        return embeddings
    
    @torch.no_grad()
    def get_text_embeds(self, prompts, negative_prompts):
        pos_embeds = self.encode_txt(prompts)
        neg_embeds = self.encode_txt(negative_prompts)
        self.embeddings = torch.cat([neg_embeds, pos_embeds], dim=0)
    
    @torch.no_grad()
    def produce_latents(
        self,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        latents=None
    ):
        if latents is None:
            latents = torch.randn((
                    self.embeddings.shape[0] // 2,
                    self.unet.in_channels,
                    height // 8,
                    width // 8,
                ),
                device=self.device,
                dtype=self.weights_dtype
            )
            
        self.scheduler.set_timesteps(num_inference_steps)
        
        for i, t in enumerate(self.scheduler.timesteps):
            latent_model_input = torch.cat([latents] * 2)
            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=self.embeddings    
            ).sample
            
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )
            
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            
        return latents

    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents

        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs
    
    def prompt_to_img(
        self, 
        prompts,
        negative_prompts="",
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        latents=None
    ):
        if isinstance(prompts, str):
            prompts = [prompts]
        
        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]
            
        self.get_text_embeds(prompts, negative_prompts)
        
        latents = self.produce_latents(
            height=height,
            width=width,
            latents=latents,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
        
        imgs = self.decode_latents(latents)
        
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype("uint8")
        
        return imgs
    
    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor

        return latents
    
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
        

if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt
    
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", type=str)
    parser.add_argument("--negative", default="", type=str)
    
    parser.add_argument("-H", type=int, default=512)
    parser.add_argument("-W", type=int, default=512)
    parser.add_argument("--steps", type=int, default=50)
    
    opt = parser.parse_args()
    sd = StableDiffusion()
    imgs = sd.prompt_to_img(opt.prompt, opt.negative, opt.H, opt.W, opt.steps)
    
    plt.imshow(imgs[0])
    plt.show()

        