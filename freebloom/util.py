import os
import imageio
import numpy as np
from PIL import Image
from typing import Union

import torch
import torchvision

from tqdm import tqdm
from einops import rearrange


def save_tensor_img(img, save_path):
    """
    :param img c, h, w , -1~1
    """
    # img = (img + 1.0) / 2.0
    img = Image.fromarray(img.mul(255).byte().numpy().transpose(1, 2, 0))
    img.save(save_path)


def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=4, fps=8):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps)


def save_videos_per_frames_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=4):
    os.makedirs(path, exist_ok=True)
    for i, video in enumerate(videos):
        video = rearrange(video, "c f h w -> f c h w")
        x = torchvision.utils.make_grid(video, nrow=n_rows)
        if rescale:
            x = (x + 1.0) / 2.0
        # x = (x * 255).numpy().astype(np.int8)
        torchvision.utils.save_image(x, f'{path}/{i}_all.jpg')
        for j, img in enumerate(video):
            save_tensor_img(img, f'{path}/{j}.jpg')

# DDIM Inversion
@torch.no_grad()
def init_prompt(prompt, pipeline):
    uncond_input = pipeline.tokenizer(
        [""], padding="max_length", max_length=pipeline.tokenizer.model_max_length,
        return_tensors="pt"
    )
    uncond_embeddings = pipeline.text_encoder(uncond_input.input_ids.to(pipeline.device))[0]
    text_input = pipeline.tokenizer(
        [prompt],
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = pipeline.text_encoder(text_input.input_ids.to(pipeline.device))[0]
    context = torch.cat([uncond_embeddings, text_embeddings])

    return context


def next_step(model_output: Union[torch.FloatTensor, np.ndarray], timestep: int,
              sample: Union[torch.FloatTensor, np.ndarray], ddim_scheduler):
    timestep, next_timestep = min(
        timestep - ddim_scheduler.config.num_train_timesteps // ddim_scheduler.num_inference_steps, 999), timestep
    alpha_prod_t = ddim_scheduler.alphas_cumprod[timestep] if timestep >= 0 else ddim_scheduler.final_alpha_cumprod
    alpha_prod_t_next = ddim_scheduler.alphas_cumprod[next_timestep]
    beta_prod_t = 1 - alpha_prod_t
    next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
    next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
    return next_sample


def get_noise_pred_single(latents, t, context, unet):
    noise_pred = unet(latents, t, encoder_hidden_states=context)["sample"]
    return noise_pred


@torch.no_grad()
def ddim_loop(pipeline, ddim_scheduler, latent, num_inv_steps, prompt):
    context = init_prompt(prompt, pipeline)
    uncond_embeddings, cond_embeddings = context.chunk(2)

    all_latent = [latent]
    latent = latent.clone().detach()
    for i in tqdm(range(num_inv_steps)):
        t = ddim_scheduler.timesteps[len(ddim_scheduler.timesteps) - i - 1]
        noise_pred = get_noise_pred_single(latent, t, cond_embeddings, pipeline.unet)
        latent = next_step(noise_pred, t, latent, ddim_scheduler)
        all_latent.append(latent)
    return all_latent


@torch.no_grad()
def ddim_inversion(pipeline, ddim_scheduler, video_latent, num_inv_steps, prompt=""):
    ddim_latents = ddim_loop(pipeline, ddim_scheduler, video_latent, num_inv_steps, prompt)
    return ddim_latents


def invert(image_path, weight_dtype, pipeline, ddim_scheduler, num_inv_steps, prompt=""):
    image_gt = load_512(image_path)
    image = torch.from_numpy(image_gt).type(weight_dtype) / 127.5 - 1
    image = image.permute(2, 0, 1).unsqueeze(0).to(pipeline.vae.device)
    latent = pipeline.vae.encode(image)['latent_dist'].mean.unsqueeze(2)
    latent = latent * 0.18215  # pipeline.vae.config.scaling_factor
    latents = ddim_inversion(pipeline, ddim_scheduler, latent, num_inv_steps, prompt=prompt)
    return latents


def load_512(image_path, left=0, right=0, top=0, bottom=0):
    if type(image_path) is str:
        image = np.array(Image.open(image_path))[:, :, :3]
    else:
        image = image_path
    h, w, c = image.shape
    left = min(left, w - 1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h - bottom, left:w - right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    image = np.array(Image.fromarray(image).resize((512, 512)))
    return image


@torch.no_grad()
def latent2image(vae, latents):
    latents = 1 / vae.config.scaling_factor * latents
    image = vae.decode(latents)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image


@torch.no_grad()
def image2latent(vae, image, device, dtype=torch.float32):
    with torch.no_grad():
        if type(image) is Image:
            image = np.array(image)
        if type(image) is torch.Tensor and image.dim() == 4:
            latents = image
        else:
            image = torch.from_numpy(image).type(dtype) / 127.5 - 1
            image = image.permute(2, 0, 1).unsqueeze(0).to(device)
            latents = vae.encode(image)['latent_dist'].mean
            latents = latents * vae.config.scaling_factor
    return latents
