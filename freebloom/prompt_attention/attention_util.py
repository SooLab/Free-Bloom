import abc
import copy
import os
from typing import Union, Tuple, Dict, Optional, List

import numpy as np
import torch
from PIL import Image

from freebloom.prompt_attention import ptp_utils, seq_aligner
from freebloom.prompt_attention.ptp_utils import get_time_string


class AttentionControl(abc.ABC):

    def step_callback(self, x_t):
        self.cur_att_layer = 0
        self.cur_step += 1
        self.between_steps()
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if self.LOW_RESOURCE else 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if self.LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1

        return attn

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.LOW_RESOURCE = False
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0


class EmptyControl(AttentionControl):

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        return attn


class AttentionStore(AttentionControl):

    def step_callback(self, x_t):
        x_t = super().step_callback(x_t)
        self.latents_store[self.cur_step] = x_t
        return x_t

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [], "mid_self": [], "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        h = attn.shape[0] // self.batch_size
        if attn.shape[1] <= 32 ** 2 and is_cross:  # avoid memory overhead
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            self.step_store[key].append(attn)
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]

        if self.disk_store:
            path = self.store_dir + f'/{self.cur_step:03d}_attn.pt'
            torch.save(copy.deepcopy(self.step_store), path)

        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in
                             self.attention_store}
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def empty_cache(self):
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self, disk_store=False, config=None):
        super(AttentionStore, self).__init__()
        self.disk_store = disk_store
        if self.disk_store:
            time_string = get_time_string()
            path = f'./.temp/attention_cache_{time_string}'
            os.makedirs(path, exist_ok=True)
            self.store_dir = path
        else:
            self.store_dir = None

        if config:
            self.config = config

        self.latents_store = {}

        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.attention_type_former = config["validation_data"]["attention_type_former"]
        self.attention_type_latter = config["validation_data"]["attention_type_latter"]
        self.attention_adapt_step = config["validation_data"]["attention_adapt_step"]


class AttentionControlEdit(AttentionStore, abc.ABC):

    def step_callback(self, x_t):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store)
        return x_t

    def replace_self_attention(self, attn_base, att_replace):
        if att_replace.shape[2] <= 16 ** 2:
            return attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
        else:
            return att_replace

    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet)
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_repalce = attn[0], attn[1:]
            if is_cross:
                alpha_words = self.cross_replace_alpha[self.cur_step]
                attn_repalce_new = self.replace_cross_attention(attn_base, attn_repalce) * alpha_words + (
                        1 - alpha_words) * attn_repalce
                attn[1:] = attn_repalce_new
            else:
                attn[1:] = self.replace_self_attention(attn_base, attn_repalce)
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn

    def __init__(self, prompts, num_steps: int,
                 cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                 self_replace_steps: Union[float, Tuple[float, float]],
                 local_blend: None,
                 tokenizer=None,
                 device=torch.device('cuda') if torch.cuda.is_available() else torch.device(
                     'cpu')):  # Optional[LocalBlend]):
        super(AttentionControlEdit, self).__init__()
        self.batch_size = len(prompts)
        self.cross_replace_alpha = ptp_utils.get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps,
                                                                            tokenizer).to(device)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.local_blend = local_blend


class AttentionReplace(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        return torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper)

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend=None, tokenizer=None,
                 device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')):
        super(AttentionReplace, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend,
                                               tokenizer=tokenizer, device=device)
        self.mapper = seq_aligner.get_replacement_mapper(prompts, tokenizer).to(device)


class AttentionRefine(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
        attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend=None,
                 tokenizer=None,
                 device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')):
        super(AttentionRefine, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.mapper, alphas = seq_aligner.get_refinement_mapper(prompts, tokenizer)
        self.mapper, alphas = self.mapper.to(device), alphas.to(device)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])


class AttentionReweight(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        if self.prev_controller is not None:
            attn_base = self.prev_controller.replace_cross_attention(attn_base, att_replace)
        attn_replace = attn_base[None, :, :, :] * self.equalizer[:, None, None, :]
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float, equalizer,
                 local_blend=None, controller: Optional[AttentionControlEdit] = None,
                 device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')):
        super(AttentionReweight, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps,
                                                local_blend)
        self.equalizer = equalizer.to(device)
        self.prev_controller = controller


class AttentionTest(AttentionStore):

    def step_callback(self, x_t, inner_idx=None):
        x_t = super(AttentionTest, self).step_callback(x_t)

        if inner_idx is None:
            return x_t

        b, c, f, h, w = x_t.shape

        momentum = 0.1 if self.cur_step <= self.config['inference_config']['interpolation_step'] else 1.0
        original_idx = [i for i in range(f) if i not in inner_idx]
        for idx in inner_idx:
            search_idx = sorted(original_idx + [idx])
            find = search_idx.index(idx)
            pre_idx, next_idx = search_idx[find - 1], search_idx[find + 1]
            length = next_idx - pre_idx

            alpha = (idx - pre_idx) / length
            x_t[:, :, idx] = (1 - momentum) * ((next_idx - idx) / length * x_t[:, :, pre_idx] + (
                    idx - pre_idx) / length * x_t[:, :, next_idx]) + momentum * x_t[:, :, idx]

        return x_t

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        super(AttentionTest, self).forward(attn, is_cross, place_in_unet)
        if is_cross:
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_repalce, attn_show = attn[0], attn[1:2], attn[2:3]
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn


def get_equalizer(text: str,
                  word_select: Union[int, Tuple[int, ...]],
                  values: Union[List[float], Tuple[float, ...]],
                  tokenizer=None):
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer = torch.ones(len(values), 77)
    values = torch.tensor(values, dtype=torch.float32)
    for word in word_select:
        inds = ptp_utils.get_word_inds(text, word, tokenizer)
        equalizer[:, inds] = values
    return equalizer


def aggregate_attention(prompts,
                        attention_store: AttentionStore,
                        res: int,
                        from_where: List[str],
                        is_cross: bool,
                        select: int):
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out.cpu()


def show_cross_attention(tokenizer,
                         prompts,
                         attention_store: AttentionStore, res: int, from_where: List[str],
                         select: int = 0):
    tokens = tokenizer.encode(prompts[select])
    decoder = tokenizer.decode
    attention_maps = aggregate_attention(attention_store, res, from_where, True, select)
    images = []
    for i in range(len(tokens)):
        image = attention_maps[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((256, 256)))
        image = ptp_utils.text_under_image(image, decoder(int(tokens[i])))
        images.append(image)
    ptp_utils.view_images(np.stack(images, axis=0))


def show_self_attention_comp(attention_store: AttentionStore, res: int, from_where: List[str],
                             max_com=10, select: int = 0):
    attention_maps = aggregate_attention(attention_store, res, from_where, False, select).numpy().reshape(
        (res ** 2, res ** 2))
    u, s, vh = np.linalg.svd(attention_maps - np.mean(attention_maps, axis=1, keepdims=True))
    images = []
    for i in range(max_com):
        image = vh[i].reshape(res, res)
        image = image - image.min()
        image = 255 * image / image.max()
        image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2).astype(np.uint8)
        image = Image.fromarray(image).resize((256, 256))
        image = np.array(image)
        images.append(image)
    ptp_utils.view_images(np.concatenate(images, axis=1))
