# SPDX-License-Identifier: MIT
"""Sapiens2 checkpoint + config paths."""

from __future__ import annotations

MODEL_CONFIGS: dict[str, dict[str, str]] = {
    "1b": {
        "matting_config": "configs/matting/gss_p3m_metasim/sapiens2_1b_matting_gss_p3m_metasim-1024x768.py",
        "matting_ckpt": "matting/sapiens2_1b_matting.safetensors",
        "seg_config": "configs/seg/shutterstock_goliath/sapiens2_1b_seg_shutterstock_goliath-1024x768.py",
        "seg_ckpt": "seg/sapiens2_1b_seg.safetensors",
        "normal_config": "configs/normal/metasim_render_people/sapiens2_1b_normal_metasim_render_people-1024x768.py",
        "normal_ckpt": "normal/sapiens2_1b_normal.safetensors",
    },
    "0.4b": {
        "matting_config": "",
        "matting_ckpt": "",
        "seg_config": "configs/seg/shutterstock_goliath/sapiens2_0.4b_seg_shutterstock_goliath-1024x768.py",
        "seg_ckpt": "seg/sapiens2_0.4b_seg.safetensors",
        "normal_config": "configs/normal/metasim_render_people/sapiens2_0.4b_normal_metasim_render_people-1024x768.py",
        "normal_ckpt": "normal/sapiens2_0.4b_normal.safetensors",
    },
    "0.8b": {
        "matting_config": "",
        "matting_ckpt": "",
        "seg_config": "configs/seg/shutterstock_goliath/sapiens2_0.8b_seg_shutterstock_goliath-1024x768.py",
        "seg_ckpt": "seg/sapiens2_0.8b_seg.safetensors",
        "normal_config": "configs/normal/metasim_render_people/sapiens2_0.8b_normal_metasim_render_people-1024x768.py",
        "normal_ckpt": "normal/sapiens2_0.8b_normal.safetensors",
    },
    "5b": {
        "matting_config": "",
        "matting_ckpt": "",
        "seg_config": "configs/seg/shutterstock_goliath/sapiens2_5b_seg_shutterstock_goliath-1024x768.py",
        "seg_ckpt": "seg/sapiens2_5b_seg.safetensors",
        "normal_config": "configs/normal/metasim_render_people/sapiens2_5b_normal_metasim_render_people-1024x768.py",
        "normal_ckpt": "normal/sapiens2_5b_normal.safetensors",
    },
}
