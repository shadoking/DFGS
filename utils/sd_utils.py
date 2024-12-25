import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from diffusers import (
    StableDiffusionPipeline,
    DDIMScheduler,
    DDPMScheduler,
    AutoPipelineForInpainting
)

@dataclass
class Config:
    pretrained_model_path: str = "./pretrained_models/stable-diffusion-inpainting"
    pretrained_model_path_sd: str = "./pretrained_models/stable-diffusion-1-5"
    
    guidance_scale: float = 7.5
    
    half_precision_weights: bool = True
    
    
class StableDiffusion:
    def __init__(self, blip_rst="", guidance_scale=7.5):
        self.config = Config()
        self.blip_rst = blip_rst
        self.guidance_scale = guidance_scale
        
    def configure(self) -> None:    
        self.device = torch.device("cuda")
        self.weights_dtype = torch.float16 if self.config.half_precision_weights else torch.float32
        
        pipe_kwargs = {
            "safety_checker": None,
            "requires_safety_checker": False,
            "torch_dtype": self.weights_dtype,
        }
        
        self.pipe = AutoPipelineForInpainting.from_pretrained(
            self.config.pretrained_model_path,
            **pipe_kwargs,
        ).to(self.device)
         
        self.pipe_sd = StableDiffusionPipeline.from_pretrained(
            self.config.pretrained_model_path_sd,
            **pipe_kwargs,
        ).to(self.device)
        del self.pipe_sd.text_encoder
        
        self.pipe.text_encoder.eval()
        self.pipe.vae.eval()
        self.pipe.unet.eval()
        self.pipe_sd.unet.eval()
        
        
        for p in self.text_encoder.parameters():
            p.requires_grad_(False)
        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)
            
        self.scheduler = DDPMScheduler.from_pretrained(
            self.config.pretrained_model_path,
            subfolder="scheduler",
            torch_dtype=self.weights_dtype,
        )
        
        self.scheduler_sd = DDPMScheduler.from_pretrained(
            self.config.pretrained_model_path_sd,
            subfolder='scheduler',
            torch_dtype=self.weights_dtype,
        )
        
        self.scheduler_sample = DDIMScheduler.from_config(
            self.pipe.scheduler.config
        )
        
        self.scheduelr_sd_sample = DDIMScheduler.from_config(
            self.pipe_sd.scheduler.config
        )
        
        self.pipe.scheduler = self.scheduler
        self.pipe_sd.scheduler = self.scheduler_sd
        
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps

        self.alphas = self.scheduler.alphas_cumprod.to(
            self.device
        )
        
        self.num_images_per_prompt = 1
        cross_attention_kwargs = {'scale': 1.0}
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        self.config.guidance_scale = self.guidance_scale
        
        prompt_embeds_rst = self.pipe.encode_prompt(
            prompt=self.blip_rst + ', realistic, 8k',
            negative_prompt='blurry, unrealistic',
            device=self.device,
            num_images_per_prompt=self.num_images_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            lora_scale=text_encoder_lora_scale
        )
        
        self.prompt_embeds = prompt_embeds_rst[0]
        self.negative_prompt_embeds = prompt_embeds_rst[1]
    
        
        
            
    @property
    def pipe(self):
        return self.pipe
     
    @property
    def text_encoder(self):
        return self.pipe.text_encoder
    
    @property
    def vae(self):
        return self.pipe.vae
    
    @property
    def unet(self):
        return self.pipe.unet
        
        