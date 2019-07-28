import pytest
from torchvision import transforms
import torch
from tiling_utils.tiling_utils import find_mult_to_use, find_multiples, extract_patches_2d, reconstruct_from_patches_2d


class TestHelpers:

    @pytest.fixture()
    def tensor(self):
        t = torch.randn(3, 96, 128)
        return t

    @pytest.fixture()
    def img(self):
        t = torch.randn(3, 96, 128)
        img = transforms.ToPILImage()(t)
        return img

    @pytest.fixture()
    def batch(self):
        t = torch.randn(3, 96, 128)
        batch = torch.stack([t, t, t, t], dim=0)
        return batch

    @pytest.fixture()
    def patches(self):
        t = torch.randn(4, 12, 3, 32, 32)
        return t

    def test_find_multiples(self):
        assert find_multiples(96, 128) == [1, 2, 4, 8, 16, 32]

    def test_find_mult_to_use(self, img):
        assert find_mult_to_use(img, 32, 32) == 32
        assert find_mult_to_use(img, 4, 8) == 8

    def test_to_patches(self, batch):
        assert extract_patches_2d(
            batch, (32, 32), batch_first=True).shape == torch.Size([4, 12, 3, 32, 32])
        assert extract_patches_2d(
            batch, (32, 32), batch_first=False).shape == torch.Size([12, 4, 3, 32, 32])

    def test_from_patches(self, patches):
        assert reconstruct_from_patches_2d(
            patches, (96, 128), batch_first=True).shape == torch.Size([4, 3, 96, 128])
