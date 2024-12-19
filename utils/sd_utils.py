import torch
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
    
class StableDiffusion:
    def __init__(self):
        self.config = Config()
        
    def config(self):
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
        
    def prompt_to_img(self):
        pass