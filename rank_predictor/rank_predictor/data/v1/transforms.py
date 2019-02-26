from torch import Tensor
from torchvision import transforms
from typing import Dict
from torchvision.transforms import ToTensor


class ImageTransform:
    """
    Applies an image transformation to the image of a sample from the dataset.
    """

    def __init__(self, transform: transforms.Compose):
        self.transform = transform

    def __call__(self, sample: Dict[str, any]) -> Dict[str, any]:
        sample['img'] = self.transform(sample['img'])
        return sample


class ToCudaTensor(ToTensor):
    """
    ToTensor conversion with placement on a given device.
    """

    def __init__(self, device) -> None:
        super().__init__()
        self.device = device

    def __call__(self, pic):
        tensor: Tensor = super().__call__(pic)
        return tensor.to(self.device)
