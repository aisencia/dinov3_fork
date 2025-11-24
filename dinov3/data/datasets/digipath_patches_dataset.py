from typing import Any, Callable, Optional, Tuple, Union
from PIL import Image

from digipath.datasets import PatchesDataset
from torchvision.datasets import VisionDataset


_Target = int


def _get_patches_dataset_image(dataset, index):
    row = dataset._df.iloc[index]
    image = dataset._load_image(row)

    if dataset._transforms is not None:
        transformed = dataset._transforms(image=image)
        image = transformed["image"]

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    return image

class DigipathPatchesDataset(VisionDataset):
    Target = Union[_Target]

    def __init__(
        self,
        *,
        root: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)

        self._dataset = PatchesDataset.load_from_file(root)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image = _get_patches_dataset_image(self._dataset, index)
        target = None

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self._dataset)
