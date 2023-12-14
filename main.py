import argparse
import datetime
import inspect
import logging
import os
from typing import Dict, Optional

import diffusers
import numpy as np
import torch
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available
from omegaconf import OmegaConf
from transformers import CLIPTextModel, CLIPTokenizer

from freebloom.models.unet import UNet3DConditionModel
from freebloom.pipelines.pipeline_spatio_temporal import SpatioTemporalPipeline
from freebloom.util import save_videos_grid, save_videos_per_frames_grid

logger = get_logger(__name__, log_level="INFO")


def main(
        pretrained_model_path: str,
        output_dir: str,
        validation_data: Dict,
        mixed_precision: Optional[str] = "fp16",
        enable_xformers_memory_efficient_attention: bool = True,
        seed: Optional[int] = None,
        inference_config: Dict = None,
):
    *_, config = inspect.getargvalues(inspect.currentframe())

    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        project_dir=f"{output_dir}/acc_log"
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if seed is not None:
        set_seed(seed)

    # Handle the output folder creation
    if accelerator.is_main_process:
        now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        output_dir = os.path.join(output_dir, now)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/samples", exist_ok=True)
        OmegaConf.save(config, os.path.join(output_dir, 'config.yaml'))

    # Load scheduler, tokenizer and models.
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    unet = UNet3DConditionModel.from_pretrained_2d(pretrained_model_path, subfolder="unet")

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Get the validation pipeline
    validation_pipeline = SpatioTemporalPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
        scheduler=DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler"),
        disk_store=False,
        config=config
    )
    validation_pipeline.enable_vae_slicing()
    validation_pipeline.scheduler.set_timesteps(validation_data.num_inv_steps)

    # Prepare everything with our `accelerator`.
    unet = accelerator.prepare(unet)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("SpatioTemporal")

    text_encoder.eval()
    vae.eval()
    unet.eval()

    generator = torch.Generator(device=unet.device)
    generator.manual_seed(seed)

    samples = []

    prompt = list(validation_data.prompts)
    negative_prompt = config['validation_data']['negative_prompt']
    negative_prompt = [negative_prompt] * len(prompt)

    with (torch.no_grad()):
        x_base = validation_pipeline.prepare_latents(batch_size=1,
                                                     num_channels_latents=4,
                                                     video_length=len(prompt),
                                                     height=512,
                                                     width=512,
                                                     dtype=weight_dtype,
                                                     device=unet.device,
                                                     generator=generator,
                                                     store_attention=True,
                                                     frame_same_noise=True)

        x_res = validation_pipeline.prepare_latents(batch_size=1,
                                                    num_channels_latents=4,
                                                    video_length=len(prompt),
                                                    height=512,
                                                    width=512,
                                                    dtype=weight_dtype,
                                                    device=unet.device,
                                                    generator=generator,
                                                    store_attention=True,
                                                    frame_same_noise=False)

        x_T = np.cos(inference_config['diversity_rand_ratio'] * np.pi / 2) * x_base + np.sin(
            inference_config['diversity_rand_ratio'] * np.pi / 2) * x_res

        validation_data.pop('negative_prompt')
        # key frame
        key_frames, text_embedding = validation_pipeline(prompt, video_length=len(prompt), generator=generator,
                                                         latents=x_T.type(weight_dtype),
                                                         negative_prompt=negative_prompt,
                                                         output_dir=output_dir,
                                                         return_text_embedding=True,
                                                         **validation_data)
        torch.cuda.empty_cache()

        samples.append(key_frames[0])
    samples = torch.concat(samples)
    save_path = f"{output_dir}/samples/sample.gif"
    save_videos_grid(samples, save_path, n_rows=6)
    save_videos_per_frames_grid(samples, f'{output_dir}/img_samples', n_rows=6)
    logger.info(f"Saved samples to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="./configs/flowers.yaml")
    args = parser.parse_args()

    conf = OmegaConf.load(args.config)
    main(**conf)
