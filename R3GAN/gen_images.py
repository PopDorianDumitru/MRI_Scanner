# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import os
import re
from typing import List, Optional, Union

import click
import R3GAN.dnnlib as dnnlib
import numpy as np
import PIL.Image
import torch

import R3GAN.legacy as legacy


def generate_images_ui(
    network_pkl: str,
    seeds: List[int],
    class_idx: Optional[int]
) -> List[PIL.Image.Image]:
    print(f'Loading networks from "{network_pkl}"...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)

    # Create label tensor
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            raise ValueError('Must specify class label for conditional generation')
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print('Warning: class label ignored for unconditional network')

    images = []

    for seed_idx, seed in enumerate(seeds):
        print(f"Generating image for seed {seed} ({seed_idx + 1}/{len(seeds)})...")

        # Generate latent z
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)

        print(f"Starting to generate image")
        with torch.no_grad():
            img = G(z, label)  # img shape: [1, 1, H, W] for grayscale
        print(f"Image generated")
        # Scale to [0, 255] and convert to uint8
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)

        # Convert to numpy array and squeeze channel dimension
        img_np = img[0].cpu().numpy()  # shape: [1, H, W]
        if img_np.shape[0] == 1:
            img_np = img_np[0]  # shape: [H, W]
        else:
            raise ValueError(f"Expected 1-channel grayscale image, got shape: {img_np.shape}")

        img_pil = PIL.Image.fromarray(img_np, mode='L')
        images.append(img_pil)

    return images

#----------------------------------------------------------------------------
def parse_range(s: Union[str, List]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=parse_range, help='List of random seeds (e.g., \'0,1,4-6\')', required=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
def generate_images(
    network_pkl: str,
    seeds: List[int],
    outdir: str,
    class_idx: Optional[int]
):
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    os.makedirs(outdir, exist_ok=True)

    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            raise click.ClickException('Must specify class label with --class when using a conditional network')
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print ('warn: --class=lbl ignored when running on an unconditional network')

    # Generate images.
    for seed_idx, seed in enumerate(seeds):
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        img = G(z, label)
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)  # no permute
        img_np = img[0, 0].cpu().numpy()  # shape: [H, W]
        PIL.Image.fromarray(img_np.astype(np.uint8), mode='L').save(f'{outdir}/seed{seed:04d}.png')


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------