# make sure you're logged in with `huggingface-cli login`
from torch import autocast
import torch
from diffusers import StableDiffusionPipeline, DDPMPipeline
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import os
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Diffusion Inference')
    parser.add_argument('--model_id', type=str)
    parser.add_argument('--lora_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--prompt', type=str)
    parser.add_argument('--img_num', default=50, type=int)
    parser.add_argument('--train_text_encoder', type=int, default=1)
    args = parser.parse_args()

    if os.path.exists(output_dir)==False:
        os.mkdir(output_dir)
    
    pipe = DiffusionPipeline.from_pretrained(args.model_id, use_auth_token=True)
    if args.train_text_encoder == 1:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.unet.load_attn_procs(args.lora_dir).to("cuda")
    
    saved_idx = 0
    iter_idx = 0
    while saved_idx < args.img_num:
        with autocast("cuda"):
            image = pipe(args.prompt, guidance_scale=7.5).images[0]
        r,g,b = image.getextrema()
        iter_idx += 1
        if r[1]==0 and g[1]==0 and b[1]==0:
            continue
        else:
            image.save(output_dir+ "/"+args.prompt + " " + str(saved_idx)+".png")
            saved_idx += 1