from PIL import Image
from pathlib import Path
import torch
from torch import nn
from torchvision import transforms
from torchvision.utils import save_image
from typing import Tuple, Set, List

__all__ = ['preprocess_into_patches', 'reconstruct_imgs']


IMAGE_FILES = ['.jpg', '.jpeg', '.png']


def preprocess_into_patches(path_in: Path, path_out: Path, patch_size_range: Tuple[int, int]) -> None:
    """
    Reads in images in a directory and attempts to find the best patch multiple to use within a range.
    Converts to patches and saves in the `path_out` directory with format:
    <old_name>_<patch_idx>.<extension>

    Note: patch_size_range is the range of finding the common denomenator of the width / height by which
    we do the patching operations
    """
    mult_set = None
    imgs = [p for p in path_in.iterdir() if p.suffix in IMAGE_FILES]
    min_multiple, max_multiple = patch_size_range

    path_out.mkdir(parents=True, exist_ok=True)
    for _p in imgs:
        f_name, suffix = _p.name.split(".")
        im = Image.open(_p)

        if mult_set is None:
            mult_set = find_mult_to_use(
                im, min_multiple=min_multiple, max_multiple=max_multiple)

        tens = transforms.ToTensor()(im).unsqueeze(0)
        patches = extract_patches_2d(
            tens, (mult_set, mult_set), batch_first=True).squeeze()
        for idx in range(patches.shape[0]):
            _patch = patches[idx]
            out_file = str(path_out.joinpath(f"{f_name}_{idx}.{suffix}"))
            save_image(_patch, out_file)

    return


def reconstruct_imgs(path_in: Path, path_out: Path, original_img_ref: Path) -> None:
    """
    Takes patches of a larger image and stitches them together, saving them out
    to the `path_out` directory. This takes reference to the original image for sizing
    of assembly.
    """
    name_set: Set[str] = set()
    path_out.mkdir(parents=True, exist_ok=True)

    imgs = [p for p in path_in.iterdir() if p.suffix in IMAGE_FILES]
    for img in imgs:
        # print(img)
        real_name = "_".join(img.name.split("_")[:-1])
        if real_name not in name_set:
            def patch_is_part_of_original(to_match: str, to_check: str) -> bool:
                return "_".join(to_check.split("_")[:-1]) == to_match
            patches = [img for img in imgs if patch_is_part_of_original(
                real_name, img.name)]
            suffix = patches[0].suffix
            tensors = []
            for patch_idx in range(len(patches)):
                to_read_file = img.parents[0].joinpath(
                    f"{real_name}_{patch_idx}{suffix}")
                im = Image.open(to_read_file)
                tens = transforms.ToTensor()(im)
                tensors.append(tens)
            tensor = torch.stack(tensors, dim=0).unsqueeze(0)
            ref_h, ref_w = Image.open(
                original_img_ref.joinpath(f"{real_name}{suffix}")).size
            tensor = reconstruct_from_patches_2d(
                tensor, (ref_w, ref_h), batch_first=True).squeeze()

            save_image(tensor, path_out.joinpath(f"{real_name}{suffix}"))
            name_set.add(real_name)
    return


def find_multiples(dim1: int, dim2: int) -> List[int]:
    """
    Helper to get different available patching sizes
    by finding the common denominator.
    """
    multiples = []
    for i in range(1, min(dim1, dim2)):
        if dim1 % i == dim2 % i == 0:
            multiples.append(i)
    return multiples


def find_mult_to_use(im: Image, min_multiple: int, max_multiple: int) -> int:
    """
    Tries to smartly find the correct patch size to use
    """
    multiples: List[int] = []
    w, h = im.size
    while not len(multiples) > 0:
        multiples = find_multiples(w, h)
        multiples = [m for m in multiples if m >=
                     min_multiple and m <= max_multiple]
        if w < h:
            w += 1
        elif h < w:
            h += 1
        elif w == h:
            w += 1
            h += 1

    if len(multiples) > 0:
        return multiples[-1]


def extract_patches_2d(img: torch.Tensor, patch_shape: Tuple[int, int], step: List[int] = [1.0, 1.0], batch_first: bool = False) -> torch.Tensor:
    """
    General tensor manipulation to actually convert to patches
    """
    patch_H, patch_W = patch_shape[0], patch_shape[1]
    if(img.size(2) < patch_H):
        num_padded_H_Top = (patch_H - img.size(2))//2
        num_padded_H_Bottom = patch_H - img.size(2) - num_padded_H_Top
        padding_H = nn.ConstantPad2d(
            (0, 0, num_padded_H_Top, num_padded_H_Bottom), 0)
        img = padding_H(img)
    if(img.size(3) < patch_W):
        num_padded_W_Left = (patch_W - img.size(3))//2
        num_padded_W_Right = patch_W - img.size(3) - num_padded_W_Left
        padding_W = nn.ConstantPad2d(
            (num_padded_W_Left, num_padded_W_Right, 0, 0), 0)
        img = padding_W(img)
    step_int = [0, 0]
    step_int[0] = int(
        patch_H*step[0]) if(isinstance(step[0], float)) else step[0]
    step_int[1] = int(
        patch_W*step[1]) if(isinstance(step[1], float)) else step[1]
    patches_fold_H = img.unfold(2, patch_H, step_int[0])
    if((img.size(2) - patch_H) % step_int[0] != 0):
        patches_fold_H = torch.cat(
            (patches_fold_H, img[:, :, -patch_H:, ].permute(0, 1, 3, 2).unsqueeze(2)), dim=2)
    patches_fold_HW = patches_fold_H.unfold(3, patch_W, step_int[1])
    if((img.size(3) - patch_W) % step_int[1] != 0):
        patches_fold_HW = torch.cat(
            (patches_fold_HW, patches_fold_H[:, :, :, -patch_W:, :].permute(0, 1, 2, 4, 3).unsqueeze(3)), dim=3)
    patches = patches_fold_HW.permute(2, 3, 0, 1, 4, 5)
    patches = patches.reshape(-1, img.size(0), img.size(1), patch_H, patch_W)
    if(batch_first):
        patches = patches.permute(1, 0, 2, 3, 4)
    return patches


def reconstruct_from_patches_2d(patches: torch.Tensor, img_shape: Tuple[int, int], step: List[int] = [1.0, 1.0], batch_first: bool = False) -> torch.Tensor:
    """
    General tensor manipulation to actually reconstruct the patches
    """
    if(batch_first):
        patches = patches.permute(1, 0, 2, 3, 4)
    patch_H, patch_W = patches.size(3), patches.size(4)
    img_size = (patches.size(1), patches.size(2), max(
        img_shape[0], patch_H), max(img_shape[1], patch_W))
    step_int = [0, 0]
    step_int[0] = int(
        patch_H*step[0]) if(isinstance(step[0], float)) else step[0]
    step_int[1] = int(
        patch_W*step[1]) if(isinstance(step[1], float)) else step[1]
    nrow, ncol = 1 + (img_size[-2] - patch_H)//step_int[0], 1 + \
        (img_size[-1] - patch_W)//step_int[1]
    r_nrow = nrow + 1 if((img_size[2] - patch_H) % step_int[0] != 0) else nrow
    r_ncol = ncol + 1 if((img_size[3] - patch_W) % step_int[1] != 0) else ncol
    patches = patches.reshape(
        r_nrow, r_ncol, img_size[0], img_size[1], patch_H, patch_W)
    img = torch.zeros(img_size, device=patches.device)
    overlap_counter = torch.zeros(img_size, device=patches.device)
    for i in range(nrow):
        for j in range(ncol):
            img[:, :, i*step_int[0]:i*step_int[0]+patch_H, j *
                step_int[1]:j*step_int[1]+patch_W] += patches[i, j, ]
            overlap_counter[:, :, i*step_int[0]:i*step_int[0] +
                            patch_H, j*step_int[1]:j*step_int[1]+patch_W] += 1
    if((img_size[2] - patch_H) % step_int[0] != 0):
        for j in range(ncol):
            img[:, :, -patch_H:, j*step_int[1]:j *
                step_int[1]+patch_W] += patches[-1, j, ]
            overlap_counter[:, :, -patch_H:, j *
                            step_int[1]:j*step_int[1]+patch_W] += 1
    if((img_size[3] - patch_W) % step_int[1] != 0):
        for i in range(nrow):
            img[:, :, i*step_int[0]:i*step_int[0] +
                patch_H, -patch_W:] += patches[i, -1, ]
            overlap_counter[:, :, i*step_int[0]:i *
                            step_int[0]+patch_H, -patch_W:] += 1
    if((img_size[2] - patch_H) % step_int[0] != 0 and (img_size[3] - patch_W) % step_int[1] != 0):
        img[:, :, -patch_H:, -patch_W:] += patches[-1, -1, ]
        overlap_counter[:, :, -patch_H:, -patch_W:] += 1
    img /= overlap_counter
    if(img_shape[0] < patch_H):
        num_padded_H_Top = (patch_H - img_shape[0])//2
        num_padded_H_Bottom = patch_H - img_shape[0] - num_padded_H_Top
        img = img[:, :, num_padded_H_Top:-num_padded_H_Bottom, ]
    if(img_shape[1] < patch_W):
        num_padded_W_Left = (patch_W - img_shape[1])//2
        num_padded_W_Right = patch_W - img_shape[1] - num_padded_W_Left
        img = img[:, :, :, num_padded_W_Left:-num_padded_W_Right]
    return img
