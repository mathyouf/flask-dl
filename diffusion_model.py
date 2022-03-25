# Imports
import gc
import io
import random
import sys
import os

import lpips
import requests
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF

import numpy as np
import imageio
import boto3

from filterWords import removeStopWords
from dotenv import load_dotenv

sys.path.append('./CLIP')
sys.path.append('./guided-diffusion')

import clip
from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults


# Define necessary functions

def fetch(url_or_path):
    if str(url_or_path).startswith('http://') or str(url_or_path).startswith('https://'):
        r = requests.get(url_or_path)
        r.raise_for_status()
        fd = io.BytesIO()
        fd.write(r.content)
        fd.seek(0)
        return fd
    return open(url_or_path, 'rb')


def parse_prompt(prompt):
    if prompt.startswith('http://') or prompt.startswith('https://'):
        vals = prompt.rsplit(':', 2)
        vals = [vals[0] + ':' + vals[1], *vals[2:]]
    else:
        vals = prompt.rsplit(':', 1)
    vals = vals + ['', '1'][len(vals):]
    return vals[0], float(vals[1])


class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1., cutn_whole_portion=0.0, cutn_bw_portion=0.2, rotation_fill=[1,1,1]):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow
        self.cutn_whole_portion = cutn_whole_portion
        self.cutn_bw_portion = cutn_bw_portion
        self.rotation_fill = rotation_fill

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        if self.cutn == 1:
            cutouts.append(F.adaptive_avg_pool2d(input, self.cut_size))
            return torch.cat(cutouts)
        cut_1 = round(self.cutn * (1 - self.cutn_whole_portion))
        cut_2 = self.cutn - cut_1
        gray = transforms.Grayscale(3)
        if cut_1 > 0:
            for i in range(cut_1):
                size = int(torch.rand([]) ** self.cut_pow * (max_size - min_size) + min_size)
                offsetx = torch.randint(0, sideX - size + 1, ())
                offsety = torch.randint(0, sideY - size + 1, ())
                cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
                if i < int(self.cutn_bw_portion * cut_1):
                    cutout = gray(cutout)
                cutouts.append(F.adaptive_avg_pool2d(cutout, self.cut_size))
        if cut_2 > 0:
            for i in range(cut_2):
                cutout = TF.rotate(input, angle=random.uniform(-10.0, 10.0), expand=True, fill=self.rotation_fill)
                if i < int(self.cutn_bw_portion * cut_2):
                    cutout = gray(cutout)
                cutouts.append(F.adaptive_avg_pool2d(cutout, self.cut_size))
        return torch.cat(cutouts)


def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


def tv_loss(input):
    """L2 total variation loss, as in Mahendran et al."""
    input = F.pad(input, (0, 1, 0, 1), 'replicate')
    x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
    y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
    return (x_diff ** 2 + y_diff ** 2).mean([1, 2, 3])


def range_loss(input):
    return (input - input.clamp(-1, 1)).pow(2).mean([1, 2, 3])


def savetoS3Bucket(image_path):
    session = boto3.Session(
        aws_access_key_id=os.getenv("ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("SECRET_ACCESS_KEY"),
    )
    s3 = session.resource('s3')
    s3.meta.client.upload_file(Filename=image_path.split('/')[-1], Bucket='imagination-machine', Key=image_path)

    # client = boto3.client('s3', region_name='us-east-2')
    # client.upload_file(image_path, bucket, image_path.split('/')[-1])


    return


def do_run(model, model_params, model_list, model_config, clip_model, clip_size, device, diffusion, folder_name, session):
    torch.cuda.empty_cache()

    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])

    lpips_model = lpips.LPIPS(net='vgg').to(device)

    if model_params['seed'] is not None:
        torch.manual_seed(model_params['seed'])
    make_cutouts = {}
    for i in model_list:
        make_cutouts[i] = MakeCutouts(clip_size[i], model_params['cutn'] // len(model_list),
                                      model_params['cut_pow'], model_params['cutn_whole_portion'],
                                      model_params['cutn_bw_portion'], model_params['rotation_fill'])

    side_x = side_y = model_config['image_size']

    target_embeds, weights = {}, []
    for i in model_list:
        target_embeds[i] = []

    for prompt in model_params['prompts']:
        txt, weight = parse_prompt(prompt)
        txt = removeStopWords(txt)
        print(txt)
        for i in model_list:
            target_embeds[i].append(clip_model[i].encode_text(clip.tokenize(txt).to(device)).float())
        weights.append(weight)

    for prompt in model_params['image_prompts']:
        path, weight = parse_prompt(prompt)
        img = Image.open(fetch(path)).convert('RGB')
        img = TF.resize(img, min(side_x, side_y, *img.size), transforms.InterpolationMode.LANCZOS)
        for i in model_list:
            batch = make_cutouts[i](TF.to_tensor(img).unsqueeze(0).to(device))
            embed = clip_model[i].encode_image(normalize(batch)).float()
            target_embeds[i].append(embed)
        weights.extend([weight / model_params['cutn'] * len(model_list)] * (model_params['cutn'] // len(model_list)))
    for i in model_list:
        target_embeds[i] = torch.cat(target_embeds[i])
    weights = torch.tensor(weights, device=device)
    if weights.sum().abs() < 1e-3:
        raise RuntimeError('The weights must not sum to 0.')
    weights /= weights.sum().abs()

    init = None
    if model_params['init_image'] is not None:
        init = Image.open(fetch(model_params['init_image'])).convert('RGB')
        init = init.resize((side_x, side_y), Image.LANCZOS)
        init = TF.to_tensor(init).to(device).unsqueeze(0).mul(2).sub(1)

    cur_t = None

    def cond_fn(x, t, out, y=None):
        clip_guidance_scale_2 = model_params['clip_guidance_scale']

        n = x.shape[0]
        cur_output = out['pred_xstart'].detach()
        fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
        x_in = out['pred_xstart'] * fac + x * (1 - fac)

        my_t = torch.ones([n], device=device, dtype=torch.long) * cur_t
        loss = 0
        x_in_grad = torch.zeros_like(x_in)
        for k in range(model_params['cutn_batches']):
            losses = 0
            for i in model_list:
                if i == "":
                    clip_in = normalize(make_cutouts[i](x_in.mean(dim=1).expand(3, -1, -1).unsqueeze(0).add(1).div(2)))
                else:
                    clip_in = normalize(make_cutouts[i](x_in.add(1).div(2)))
                image_embeds = clip_model[i].encode_image(clip_in).float()
                image_embeds = image_embeds.unsqueeze(1)
                dists = spherical_dist_loss(image_embeds, target_embeds[i].unsqueeze(0))
                del image_embeds, clip_in
                dists = dists.view([model_params['cutn'] // len(model_list), n, -1])
                losses = dists.mul(weights).sum(2).mean(0)
                x_in_grad += torch.autograd.grad(losses.sum() * clip_guidance_scale_2, x_in)[0] / model_params[
                    'cutn_batches'] / len(
                    model_list)
                del dists, losses
            gc.collect()
        tv_losses = tv_loss(x_in)
        range_losses = range_loss(out['pred_xstart'])
        loss = tv_losses.sum() * model_params['tv_scale'] + range_losses.sum() * model_params['range_scale']
        if init is not None and model_params['init_scale']:
            init_losses = lpips_model(x_in, init)
            loss = loss + init_losses.sum() * model_params['init_scale']
        x_in_grad += torch.autograd.grad(loss, x_in, )[0]
        grad = -torch.autograd.grad(x_in, x, x_in_grad)[0]
        magnitude = grad.square().mean().sqrt()
        return grad * magnitude.clamp(max=model_params['clamp_max']) / magnitude

    if model_config['timestep_respacing'].startswith('ddim'):
        sample_fn = diffusion.ddim_sample_loop_progressive
    else:
        sample_fn = diffusion.p_sample_loop_progressive

    # createS3Folder(folder_name)

    iterations = 0
    for i in range(model_params['n_batches']):

        cur_t = diffusion.num_timesteps - model_params['skip_timesteps'] - 1

        samples = sample_fn(
            model,
            (model_params['batch_size'], 3, model_config['image_size'], model_config['image_size']),
            clip_denoised=model_params['clip_denoised'],
            model_kwargs={},
            cond_fn=cond_fn,
            progress=True,
            skip_timesteps=model_params['skip_timesteps'],
            init_image=init,
            cond_fn_with_grad=True,
            # randomize_class=True,
        )
        batch_size = model_params['batch_size']
        for j, sample in enumerate(samples):
            if j % 10 == 0 or cur_t == 0:
                for k, image in enumerate(sample['pred_xstart']):
                    filename = f'progress_{iterations}.png'
                    TF.to_pil_image(image.add(1).div(2).clamp(0, 1)).save(filename)
                    image_path = f'{session}/{folder_name}/{filename}'
                    savetoS3Bucket(image_path)
                    iterations+=1
                    # display.display(display.Image(filename))
            cur_t -= 1


def define_model(clip_input, folder_name, session, cutn=64, clip_guidance_scale=50000, tv_scale=80000, img_size=512, num_steps=1000):
    # Model settings
    load_dotenv()
    model_config = model_and_diffusion_defaults()
    model_config.update({
        'attention_resolutions': '32,16,8',
        'class_cond': False,
        'diffusion_steps': num_steps,
        'rescale_timesteps': True,
        'timestep_respacing': "24,48,64",
        # Modify this value to add the number of steps to each stages, will be slower but better quality                                 # timesteps.
        'image_size': img_size,
        'learn_sigma': True,
        'noise_schedule': 'linear',
        'num_channels': 256,
        'num_head_channels': 64,
        'num_res_blocks': 2,
        'resblock_updown': True,
        'use_fp16': True,
        'use_scale_shift_norm': True
    })

    # Use this with caution - only if you run this locally with rich vram
    model_list = [
        # 'RN50x16',
        "ViT-B/16",
        # "ViT-B/32"
    ]

    torch.cuda.empty_cache()

    # Load models

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    model, diffusion = create_model_and_diffusion(**model_config)
    model.load_state_dict(torch.load('512x512_diffusion_uncond_finetune_008100.pt', map_location='cpu'))
    model.requires_grad_(False).eval().to(device)
    for name, param in model.named_parameters():
        if 'qkv' in name or 'norm' in name or 'proj' in name:
            param.requires_grad_()
    if model_config['use_fp16']:
        model.convert_to_fp16()

    clip_model = {}
    clip_size = {}
    for i in model_list:
        clip_model[i] = clip.load(i, jit=False)[0].eval().requires_grad_(False).to(device)
        clip_size[i] = clip_model[i].visual.input_resolution

    model_params = {
        "prompts": [clip_input],
        "image_prompts": [],
        "batch_size": 1,
        "clip_guidance_scale": clip_guidance_scale,
        # Controls how much the image should look like the prompt. Use high value when clamping activated
        "tv_scale": tv_scale,  # Controls the smoothness of the final output.
        "range_scale": 25,  # Controls how far out of range RGB values are allowed to be.
        "clamp_max": 0.1,
        # Controls how far gradient can go - try play with it, dramatic effect when clip guidance scale is high enough
        "RGB_min": -0.9,
        "RGB_max": 0.9,  # Play with it to get different styles
        "cutn": cutn,
        "cutn_batches": 2,  # Turn this up for better result but slower speed
        "cutn_whole_portion": 0.1,  # The rotation augmentation, captures whole structure
        "rotation_fill": [1, 1, 1],
        "cutn_bw_portion": 0.1,
        # Greyscale augmentation, focus on structure rather than color info to give better structure
        "cut_pow": 1,
        "n_batches": 1,
        "init_image": None,  # This can be an URL or Colab local path and must be in quotes.
        "skip_timesteps": 0,
        # Skip unstable steps                  # Higher values make the output look more like the init.
        "init_scale": 0,  # This enhances the effect of the init image, a good value is 1000.
        "seed": 0,
        "clip_denoised": False,
    }

    torch.cuda.empty_cache()
    gc.collect()
    do_run(model, model_params, model_list, model_config, clip_model, clip_size, device, diffusion, folder_name, session)