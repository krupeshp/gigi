import os
from os.path import join as opj
from omegaconf import OmegaConf
from importlib import import_module
import argparse

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from cldm.plms_hacked import PLMSSampler
from cldm.model import create_model
from utils import tensor2img


@torch.no_grad()
def main():

    config_path = "./configs/VITON512.yaml"
    img_H = 512
    img_W = 384
    batch_size = 16
    model_load_path = "./ckpts/VITONHD.ckpt"
    data_root_dir = "./cloth_pair"
    unpair = True
    save_dir = "./generated_cloth"
    denoise_steps = "50"
    eta = 0.0
    repaint = False


    config = OmegaConf.load(config_path)
    config.model.params.img_H = img_H
    config.model.params.img_W = img_W
    params = config.model.params

    model = create_model(config_path=None, config=config)
    model.load_state_dict(torch.load(model_load_path, map_location="cpu"))
    model = model.cuda()
    model.eval()

    sampler = PLMSSampler(model)
    dataset = getattr(import_module("dataset"), config.dataset_name)(
        data_root_dir=data_root_dir,
        img_H=img_H,
        img_W=img_W,
        is_paired=not unpair,
        is_test=True,
        is_sorted=True
    )
    dataloader = DataLoader(dataset, num_workers=4, shuffle=False, batch_size=batch_size, pin_memory=True)

    shape = (4, img_H//8, img_W//8) 
    save_dir = opj(save_dir, "unpair" if unpair else "pair")
    os.makedirs(save_dir, exist_ok=True)
    for batch_idx, batch in enumerate(dataloader):
        print(f"{batch_idx}/{len(dataloader)}")
        z, c = model.get_input(batch, params.first_stage_key)
        bs = z.shape[0]
        c_crossattn = c["c_crossattn"][0][:bs]
        if c_crossattn.ndim == 4:
            c_crossattn = model.get_learned_conditioning(c_crossattn)
            c["c_crossattn"] = [c_crossattn]
        uc_cross = model.get_unconditional_conditioning(bs)
        uc_full = {"c_concat": c["c_concat"], "c_crossattn": [uc_cross]}
        uc_full["first_stage_cond"] = c["first_stage_cond"]
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.cuda()
        sampler.model.batch = batch

        ts = torch.full((1,), 999, device=z.device, dtype=torch.long)
        start_code = model.q_sample(z, ts)     

        samples, _, _ = sampler.sample(
            denoise_steps,
            bs,
            shape, 
            c,
            x_T=start_code,
            verbose=False,
            eta=eta,
            unconditional_conditioning=uc_full,
        )

        x_samples = model.decode_first_stage(samples)
        for sample_idx, (x_sample, fn,  cloth_fn) in enumerate(zip(x_samples, batch['img_fn'], batch["cloth_fn"])):
            x_sample_img = tensor2img(x_sample)  # [0, 255]
            if repaint:
                repaint_agn_img = np.uint8((batch["image"][sample_idx].cpu().numpy()+1)/2 * 255)   # [0,255]
                repaint_agn_mask_img = batch["agn_mask"][sample_idx].cpu().numpy()  # 0 or 1
                x_sample_img = repaint_agn_img * repaint_agn_mask_img + x_sample_img * (1-repaint_agn_mask_img)
                x_sample_img = np.uint8(x_sample_img)

            to_path = opj(save_dir, f"{fn.split('.')[0]}_{cloth_fn.split('.')[0]}.jpg")
            cv2.imwrite(to_path, x_sample_img[:,:,::-1])

if __name__ == "__main__":
    
    main()
