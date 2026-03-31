import torch
from torch.utils.data import Dataset
from datasets import load_from_disk
from torchvision import transforms
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class DiffusionDBDataset(Dataset):
    """
    DiffusionDB dataset wrapper for LoRA fine-tuning.
    Loads image-caption pairs from saved HuggingFace dataset.
    """

    def __init__(self, dataset_path: str, tokenizer, config: dict):
        self.config = config
        self.tokenizer = tokenizer

        logger.info(f"Loading dataset from {dataset_path}")
        self.dataset = load_from_disk(dataset_path)
        logger.info(f"Dataset loaded: {len(self.dataset)} examples")

        self.image_transforms = self._build_transforms()

    def _build_transforms(self) -> transforms.Compose:
        resolution = self.config["data"]["resolution"]
        transform_list = [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        ]
        if self.config["data"]["center_crop"]:
            transform_list.append(transforms.CenterCrop(resolution))
        else:
            transform_list.append(transforms.RandomCrop(resolution))

        if self.config["data"]["random_flip"]:
            transform_list.append(transforms.RandomHorizontalFlip())

        transform_list += [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
        return transforms.Compose(transform_list)

    def _tokenize_caption(self, caption: str):
        inputs = self.tokenizer(
            caption,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return inputs.input_ids.squeeze()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        example = self.dataset[idx]

        # Image
        image = example[self.config["data"]["image_column"]]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert("RGB")
        pixel_values = self.image_transforms(image)

        # Caption
        caption = example[self.config["data"]["caption_column"]]
        if not isinstance(caption, str):
            caption = str(caption)
        input_ids = self._tokenize_caption(caption)

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "caption": caption,
        }


def collate_fn(examples: list) -> dict:
    pixel_values = torch.stack([e["pixel_values"] for e in examples])
    input_ids = torch.stack([e["input_ids"] for e in examples])
    captions = [e["caption"] for e in examples]
    return {
        "pixel_values": pixel_values.to(memory_format=torch.contiguous_format).float(),
        "input_ids": input_ids,
        "captions": captions,
    }