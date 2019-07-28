[![CircleCI](https://circleci.com/gh/mark-hoffmann/tiling_utils.svg?style=svg)](https://circleci.com/gh/mark-hoffmann/tiling_utils)

# Tiling Utils

This package was written to help deal with large images when building image segmentation models. This is very common in domains such as satelite imagery, electronic part dies, engineering documents, etc.

This package assumes a particular workflow for segementation, but if you are doing something slightly different feel free to open a PR!

# Installation

```
pip install git+https://github.com/mark-hoffmann/tiling_utils
```

# Usage

## Step 1

You should have your images and labels in two different directories such as `/images` and `/labels`.

## Step 2

We are going to read in all files within those two directories and create tiled versions of them where the output files will follow the convention of `<original_file_name>_<tile_number>.<file_extension>`. Hence, `img1.png` becomes `img1_0.png`, `img1_1.png`, `img1_2.png`, etc.

In order to do this, we have to specify 3 things:

-   `path_in` - The path to where our large images are
-   `path_out` - The path we want to write our segments to
-   `patch_size_range` - This is a tuple of minimum and maximum patch sizes that we want to attempt to use for patching. So if we had inputs of size 512x512, using `(256, 256)` as our `patch_size_range` would result in 4 patches saved out for this image.

```
from tiling_utils import preprocess_into_patches
from pathlib import Path

path_in = Path('~/images')
path_out = Path('~/images_tiled')
patch_size_range = (256, 256)

preprocess_into_patches(path_in, path_out, patch_size_range=patch_size_range)

path_in = Path('~/lables')
path_out = Path('~/labels_tiled')
patch_size_range = (256, 256)

preprocess_into_patches(path_in, path_out, patch_size_range=patch_size_range)
```

## Step 3

Do your segmentation modeling. For inference or if you want to reconstruct some validation images, simply save your output segmentation masks to a directory following the segmentation path pattern mentioned above: `<original_file_name>_<tile_number>.<file_extension>`. Such as `img1_0.png`, `img1_1.png`, `img1_2.png`, etc.

## Step 4

Once you have your output segmentation masks saved to disk somewhere such as an `outputs` directory, we can call the reconstruction with the following parameters:

-   `path_in` - The directory where our outputs live
-   `path_out` - The directory where we will put the stitched together files
-   `original_img_ref` - The directory of the original large images. This is used so we know how to appropriately assemble the patches.

```
from pathlib import Path
from tiling_utils import reconstruct_imgs

path_in = Path('~/outputs_patched')
path_out = Path('~/outputs_reassembled')
original_img_ref = Path('~/outputs_originals')
reconstruct_imgs(path_in, path_out, original_img_ref)
```

And now you should have your reconstructed segmentation masks!

# License

[Apache 2.0](https://github.com/mark-hoffmann/tiling_utils/blob/master/LICENSE)

# Authors

tiling_utils was written by [Mark Hoffmann](mailto:markkhoffmann@gmail.com)
