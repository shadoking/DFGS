from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import os
import torch

processor = BlipProcessor.from_pretrained("./pretrained_models/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("./pretrained_models/blip-image-captioning-base").to("cuda")

torch.manual_seed(42)

dataset_dir = 'data'

for scene_name in os.listdir(dataset_dir):
    if scene_name.endswith("_8"):
        scene_dir = os.path.join(dataset_dir, scene_name)
        image_dir = os.path.join(scene_dir, 'images')
        image_list = sorted(os.listdir(image_dir))
        random_image_idx = torch.randint(low=0, high=len(image_list), size=(1, ))[0]
        print(scene_name, random_image_idx, image_list[random_image_idx])
    
        raw_image = Image.open(os.path.join(image_dir, image_list[random_image_idx])).convert('RGB')
    
        text = "a photography of"
        inputs = processor(raw_image, text, return_tensors="pt").to("cuda")
        out = model.generate(**inputs)
        blip_rst = processor.decode(out[0], skip_special_tokens=True)
        image_name = image_list[random_image_idx].split('.')[0]
        
        blip_rst_name = 'blip_rst.txt'
        blip_rst_dir = os.path.join(scene_dir, blip_rst_name)
        
        with open(blip_rst_dir, 'w') as f:
            writing_content = f'random select {image_name} blip result:{blip_rst}'
            f.write(writing_content)
            print(writing_content)
            f.close()
