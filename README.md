# Free-Bloom

This repository is the official implementation of [Free-Bloom](https://arxiv.org/abs/2309.14494).

**[Free-Bloom: Zero-Shot Text-to-Video Generator with LLM Director and LDM Animator](https://arxiv.org/abs/2309.14494)**

[Hanzhuo Huang∗]() , [Yufan Feng∗](), [Cheng Shi](https://chengshiest.github.io/), [Lan Xu](https://www.xu-lan.com/), [Jingyi Yu](https://vic.shanghaitech.edu.cn/vrvc/en/people/jingyi-yu/), [Sibei Yang†](https://faculty.sist.shanghaitech.edu.cn/yangsibei/)

*Equal contribution; †Corresponding Author

[![arXiv](https://img.shields.io/badge/arXiv-FreeBloom-b31b1b.svg)](https://arxiv.org/abs/2309.14494) ![Pytorch](https://img.shields.io/badge/PyTorch->=1.10.0-Red?logo=pytorch)


![image-20230924124604776](__assets__/teaser.png)

## Setup

### Requirements
```cmd
conda create -n fb python=3.10
conda activate fb
pip install -r requirements.txt
```

Installing [xformers](https://github.com/facebookresearch/xformers) is highly recommended for more efficiency and speed on GPUs. 
To enable xformers, set `enable_xformers_memory_efficient_attention=True` (default).




## Usage

### Generate
```cmd
python main.py --config configs/flowers.yaml
```

Change the path of diffusion models to your own for  the `pretrained_model_path` key in config yaml file.





## Results

**A Flower is blooming**

<table class="center">
    <tr>
    <td><img src="__assets__/flower_bloom.gif"></td>
    <td><img src="__assets__/Van_Gogh_flower.gif"></td>
    <td><img src="__assets__/yellow_flower.gif"></td>
    <td><img src="__assets__/long_video.gif"></td>
    </tr>
</table>



**Volcano eruption**

<table class="center">
    <tr>
    <td><img src="__assets__/volcano_eruption1.gif"></td>
    <td><img src="__assets__/volcano_eruption2.gif"></td>
    <td><img src="__assets__/volcano_eruption3.gif"></td>
    <td><img src="__assets__/volcano_eruption4.gif"></td>
    </tr>
</table>

**A rainbow is forming**
<table class="center">
    <tr>
    <td><img src="__assets__/rainbow_forming1.gif"></td>
    <td><img src="__assets__/rainbow_forming2.gif"></td>
    <td><img src="__assets__/rainbow_forming3.gif"></td>
    <td><img src="__assets__/rainbow_forming4.gif"></td>
    </tr>
</table>


## Citation

```
@article{freebloom,
	title={Free-Bloom: Zero-Shot Text-to-Video Generator with LLM Director and LDM Animator},
	author={Huang, Hanzhuo and Feng, Yufan and Shi, Cheng and Xu, Lan and Yu, Jingyi and Yang, Sibei},
	journal={arXiv preprint arXiv:2309.14494},
	year={2023}
}
```
