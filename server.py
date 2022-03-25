# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.
from flask import Flask, send_file, request

# Imports

import gc
import io
import math
import sys,random

from IPython import display
import lpips
from PIL import Image
import requests
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm.notebook import tqdm

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
    def __init__(self, cut_size, cutn, cut_pow=1., cutn_whole_portion = 0.0, cutn_bw_portion = 0.2):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow
        self.cutn_whole_portion = cutn_whole_portion
        self.cutn_bw_portion = cutn_bw_portion

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        if self.cutn==1:
            cutouts.append(F.adaptive_avg_pool2d(input, self.cut_size))
            return torch.cat(cutouts)
        cut_1 = round(self.cutn*(1-self.cutn_whole_portion))
        cut_2 = self.cutn-cut_1
        gray = transforms.Grayscale(3)
        if cut_1 >0:
            for i in range(cut_1):
                size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
                offsetx = torch.randint(0, sideX - size + 1, ())
                offsety = torch.randint(0, sideY - size + 1, ())
                cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
                if i < int(self.cutn_bw_portion * cut_1):
                    cutout = gray(cutout)
                cutouts.append(F.adaptive_avg_pool2d(cutout, self.cut_size))
        if cut_2 >0:
            for i in range(cut_2):
                cutout = TF.rotate(input, angle=random.uniform(-10.0, 10.0), expand=True, fill=rotation_fill)
                if i < int(self.cutn_bw_portion * cut_2):
                    cutout =gray(cutout)
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
    return (x_diff**2 + y_diff**2).mean([1, 2, 3])


def range_loss(input):
    return (input - input.clamp(-1, 1)).pow(2).mean([1, 2, 3])

# Model settings

model_config = model_and_diffusion_defaults()
model_config.update({
    'attention_resolutions': '32,16,8',
    'class_cond': False,
    'diffusion_steps': 1000,
    'rescale_timesteps': True,
    'timestep_respacing':"24,48,64",  # Modify this value to add the number of steps to each stages, will be slower but better quality                                 # timesteps.
    'image_size': 512,
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
   #'RN50x16',
   "ViT-B/16",
  # "ViT-B/32"
]

torch.cuda.empty_cache()

# Load models

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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

normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711])
lpips_model = lpips.LPIPS(net='vgg').to(device)

prompts = ['the princess was so distressed that she decided to consult a witch | witch consulting the princess | witch inside a hut giving a potion to the princess make her fall in love first person she sees | Bryce3d; cinema4d toon and cell shader; VR perspective; finely intricate detail | 8k resolution; Unreal Engine VRay; 3d smooth materials; ArtStation; CGSociety']
image_prompts = []
batch_size = 1
clip_guidance_scale = 10000  # Controls how much the image should look like the prompt. Use high value when clamping activated
tv_scale = 1000              # Controls the smoothness of the final output.
range_scale = 10            # Controls how far out of range RGB values are allowed to be.
clamp_max=0.1              # Controls how far gradient can go - try play with it, dramatic effect when clip guidance scale is high enough

RGB_min, RGB_max = [-0.1,0.1]     # Play with it to get different styles
cutn = 32
cutn_batches = 4           # Turn this up for better result but slower speed
cutn_whole_portion = 0.1       #The rotation augmentation, captures whole structure
rotation_fill=[1,1,1]
cutn_bw_portion = 0.1         #Greyscale augmentation, focus on structure rather than color info to give better structure
cut_pow = 1
n_batches = 1
init_image = None   # This can be an URL or Colab local path and must be in quotes.
skip_timesteps = 0  # Skip unstable steps                  # Higher values make the output look more like the init.
init_scale = 0      # This enhances the effect of the init image, a good value is 1000.
seed = 0
clip_denoised = False

prompts = ['the princess was so distressed that she decided to consult a witch | witch consulting the princess | witch inside a hut giving a potion to the princess make her fall in love first person she sees | Bryce3d; cinema4d toon and cell shader; VR perspective; finely intricate detail | 8k resolution; Unreal Engine VRay; 3d smooth materials; ArtStation; CGSociety']
image_prompts = []
batch_size = 1
clip_guidance_scale = 10000  # Controls how much the image should look like the prompt. Use high value when clamping activated
tv_scale = 1000  # Controls the smoothness of the final output.
range_scale = 10  # Controls how far out of range RGB values are allowed to be.
clamp_max = 0.1  # Controls how far gradient can go - try play with it, dramatic effect when clip guidance scale is high enough

RGB_min, RGB_max = [-0.1, 0.1]  # Play with it to get different styles
cutn = 32
cutn_batches = 4  # Turn this up for better result but slower speed
cutn_whole_portion = 0.1  # The rotation augmentation, captures whole structure
rotation_fill = [1, 1, 1]
cutn_bw_portion = 0.1  # Greyscale augmentation, focus on structure rather than color info to give better structure
cut_pow = 1
n_batches = 1
init_image = None  # This can be an URL or Colab local path and must be in quotes.
skip_timesteps = 0  # Skip unstable steps                  # Higher values make the output look more like the init.
init_scale = 0  # This enhances the effect of the init image, a good value is 1000.
seed = 0
clip_denoised = False

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
nltk.download('punkt')
def removeStopWords(txt):
  textarray = txt.split('|')
  for i,textprompt in enumerate(textarray):
    print(len(textprompt))
    if len(textprompt) > 77:
      # Cut it off
      text_tokens = word_tokenize(txt)
      tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]
      filtered_sentence = (" ").join(tokens_without_sw)
      textarray[i] = filtered_sentence
  fixedtext=(" ").join(textarray)
  return fixedtext


torch.cuda.empty_cache()
gc.collect()


def do_run():
    if seed is not None:
        torch.manual_seed(seed)
    make_cutouts = {}
    for i in model_list:
        make_cutouts[i] = MakeCutouts(clip_size[i], cutn // len(model_list), cut_pow, cutn_whole_portion,
                                      cutn_bw_portion)

    side_x = side_y = model_config['image_size']

    target_embeds, weights = {}, []
    for i in model_list:
        target_embeds[i] = []

    for prompt in prompts:
        txt, weight = parse_prompt(prompt)
        txt = removeStopWords(txt)
        print(txt)
        for i in model_list:
            target_embeds[i].append(clip_model[i].encode_text(clip.tokenize(txt).to(device)).float())
        weights.append(weight)

    for prompt in image_prompts:
        path, weight = parse_prompt(prompt)
        img = Image.open(fetch(path)).convert('RGB')
        img = TF.resize(img, min(side_x, side_y, *img.size), transforms.InterpolationMode.LANCZOS)
        for i in model_list:
            batch = make_cutouts[i](TF.to_tensor(img).unsqueeze(0).to(device))
            embed = clip_model[i].encode_image(normalize(batch)).float()
            target_embeds[i].append(embed)
        weights.extend([weight / cutn * len(model_list)] * (cutn // len(model_list)))
    for i in model_list:
        target_embeds[i] = torch.cat(target_embeds[i])
    weights = torch.tensor(weights, device=device)
    if weights.sum().abs() < 1e-3:
        raise RuntimeError('The weights must not sum to 0.')
    weights /= weights.sum().abs()

    init = None
    if init_image is not None:
        init = Image.open(fetch(init_image)).convert('RGB')
        init = init.resize((side_x, side_y), Image.LANCZOS)
        init = TF.to_tensor(init).to(device).unsqueeze(0).mul(2).sub(1)

    cur_t = None

    def cond_fn(x, t, out, y=None):
        clip_guidance_scale_2 = clip_guidance_scale

        n = x.shape[0]
        cur_output = out['pred_xstart'].detach()
        fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
        x_in = out['pred_xstart'] * fac + x * (1 - fac)

        my_t = torch.ones([n], device=device, dtype=torch.long) * cur_t
        loss = 0
        x_in_grad = torch.zeros_like(x_in)
        for k in range(cutn_batches):
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
                dists = dists.view([cutn // len(model_list), n, -1])
                losses = dists.mul(weights).sum(2).mean(0)
                x_in_grad += torch.autograd.grad(losses.sum() * clip_guidance_scale_2, x_in)[0] / cutn_batches / len(
                    model_list)
                del dists, losses
            gc.collect()
        tv_losses = tv_loss(x_in)
        range_losses = range_loss(out['pred_xstart'])
        loss = tv_losses.sum() * tv_scale + range_losses.sum() * range_scale
        if init is not None and init_scale:
            init_losses = lpips_model(x_in, init)
            loss = loss + init_losses.sum() * init_scale
        x_in_grad += torch.autograd.grad(loss, x_in, )[0]
        grad = -torch.autograd.grad(x_in, x, x_in_grad)[0]
        magnitude = grad.square().mean().sqrt()
        return grad * magnitude.clamp(max=clamp_max) / magnitude

    if model_config['timestep_respacing'].startswith('ddim'):
        sample_fn = diffusion.ddim_sample_loop_progressive
    else:
        sample_fn = diffusion.p_sample_loop_progressive

    for i in range(n_batches):

        cur_t = diffusion.num_timesteps - skip_timesteps - 1

        samples = sample_fn(
            model,
            (batch_size, 3, model_config['image_size'], model_config['image_size']),
            clip_denoised=clip_denoised,
            model_kwargs={},
            cond_fn=cond_fn,
            progress=True,
            skip_timesteps=skip_timesteps,
            init_image=init,
            cond_fn_with_grad=True,
            # randomize_class=True,
        )

        for j, sample in enumerate(samples):
            if j % 10 == 0 or cur_t == 0:
                for k, image in enumerate(sample['pred_xstart']):
                    filename = f'progress_{i * batch_size + k:05}.png'
                    TF.to_pil_image(image.add(1).div(2).clamp(0, 1)).save(filename)
                    # tqdm.write(f'Batch {i}, step {j}, output {k}:')
                    # display.display(display.Image(filename))
            cur_t -= 1


def setup():
    pass

def runNetwork(clip_input):
    gc.collect()
    global prompts
    prompts = [clip_input]
    do_run()

@app.route('/getImage', methods=["POST"])
def VisualImaginationMachine():
    # Get text input
    print(request.args.get('clip_input'))
    clip_input = request.args.get('clip_input')
    # Run the Network
    runNetwork(clip_input)
    # This file is generated at the end of do_run.cond_nf
    filename="progress_00000.png"
    # Return image
    return send_file(filename, mimetype="image/png")


# main driver function
if __name__ == '__main__':
    # run() method of Flask class runs the application
    # on the local development server.
    print("Enter the 🪄Imagination Machine🔮")
    app.run(host='0.0.0.0')