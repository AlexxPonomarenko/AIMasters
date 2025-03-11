# Don't erase the template code, except "Your code here" comments.

import subprocess
import sys

# List any extra packages you need here. Please, fix versions so reproduction of your results would be less painful.
PACKAGES_TO_INSTALL = ["gdown==4.4.0",]
subprocess.check_call([sys.executable, "-m", "pip", "install"] + PACKAGES_TO_INSTALL)


import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

import numpy as np

from PIL import Image
import matplotlib.pyplot as plt

from glob import glob

import tqdm

import imgaug as ia
import imgaug.augmenters as iaa

import os
from pathlib import Path

import wandb


class TINDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None, augmentation=None):
        """
        Параметры:
        - root_dir: базовый каталог, в котором хранится 'tiny-imagenet-200'
        - split: тип выборки ('train', 'val' или 'test')
        - transform: преобразования для изображений
        - augmentation: аугментации для изображений
        """
        self.root_dir = root_dir # /content/tiny-imagenet-200
        self.split = split
        self.transform = transform
        self.augmentation = augmentation

        self.data_dir = os.path.join(self.root_dir, split) # e.g. /content/tiny-imagenet-200/train

        if split in ["train", "val"]:
            self.image_paths = []
            self.labels = []

            class_dirs = sorted(os.listdir(self.data_dir)) # e.g. /content/tiny-imagenet-200/train/n01443537
            self.classes = class_dirs
            self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_dirs)} # e.g. { n01443537 : 0, n01629819 : 1, ... }

            for cls_name in class_dirs:
                cls_dir = os.path.join(self.data_dir, cls_name) # e.g. /content/tiny-imagenet-200/train/n01443537
                if split == "train":
                    cls_dir = os.path.join(cls_dir, "images") # e.g. /content/tiny-imagenet-200/train/n01443537/images

                cls_image_paths = glob(os.path.join(cls_dir, "*"))
                self.image_paths.extend(cls_image_paths)
                label = self.class_to_idx[cls_name]
                self.labels.extend([label] * len(cls_image_paths))

        elif split == "test":
            test_data_dir = os.path.join(self.data_dir, "images")
            self.image_paths = glob(os.path.join(test_data_dir, "*"))
            self.labels = [-1] * len(self.image_paths)  # метки неизвестны
            
        self.imgs = [(img_path, label) for img_path, label in zip(self.image_paths, self.labels)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        image = np.array(image)

        if self.augmentation is not None:
            image = self.augmentation.augment_image(image)

        image = Image.fromarray(image)

        if self.transform is not None:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx])
        return image, label

num_workers = os.cpu_count()
config = {
    "learning_rate": 3e-4,
    "weight_decay": 0.01,
    "batch_size": 50,
    "num_epochs": 5,
    "optimizer": torch.optim.AdamW,
    "scheduler": torch.optim.lr_scheduler.LambdaLR
}
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
loss_fn = nn.CrossEntropyLoss()

def get_dataloader(path, kind):
    """
    Return dataloader for a `kind` split of Tiny ImageNet.
    If `kind` is 'val' or 'test', the dataloader should be deterministic.
    path:
        `str`
        Path to the dataset root - a directory which contains 'train' and 'val' folders.
    kind:
        `str`
        'train', 'val' or 'test'

    return:
    dataloader:
        `torch.utils.data.DataLoader` or an object with equivalent interface
        For each batch, should yield a tuple `(preprocessed_images, labels)` where
        `preprocessed_images` is a proper input for `predict()` and `labels` is a
        `torch.int64` tensor of shape `(batch_size,)` with ground truth class labels.
    """
    augmentation = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Flipud(0.2),
        iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 2.0))),
        iaa.Sometimes(0.5, iaa.CropAndPad(percent=(-0.1, 0.2), pad_mode=ia.ALL, pad_cval=(0, 255))),
        iaa.Sometimes(0.5, iaa.Affine(
            scale={"x": (0.8, 1.5), "y": (0.8, 1.5)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-45, 45),
            shear=(-16, 16),
            order=[0, 1],
            cval=(0, 255),
            mode=ia.ALL
        )),
        iaa.Sometimes(0.5, iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2)),
        iaa.Sometimes(0.5, iaa.Multiply((0.8, 1.2), per_channel=0.2)),
        iaa.Sometimes(0.5, iaa.LinearContrast((0.75, 1.5), per_channel=0.5)),
    ], random_order=True)

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.4802, 0.4481, 0.3975], [0.2764, 0.2689, 0.2816])
    ])
    
    if kind == "train":
        dataset = TINDataset(root_dir=path, split="train", transform=transform, augmentation=augmentation)
    else:
        dataset = TINDataset(root_dir=path, split=kind, transform=transform)
        
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True if kind == "train" else False,
        drop_last=True if kind == "train" else False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=4,
    )
    
    return loader


class CNNBlock(nn.Module):
    def __init__(self, in_channels=3, out_channels=32, kernel_size=3, padding=1, stride=1, pool=False, conv=True):
        super(CNNBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding, stride, bias=False) if conv else nn.Identity()
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool= nn.MaxPool2d((2, 2)) if pool else nn.Identity()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x
    
def get_model():
    """
    Create neural net object, initialize it with raw weights, upload it to GPU.

    return:
    model:
        `torch.nn.Module`
    """
    model = nn.Sequential(
        CNNBlock(3, 32),
        CNNBlock(32, 32, pool=True), # 32x32
        CNNBlock(32, 64),
        CNNBlock(64, 64, pool=True), # 16x16
        CNNBlock(64, 128),
        CNNBlock(128, 128, pool=True), # 8x8
        CNNBlock(128, 256),
        CNNBlock(256, 256, pool=True), # 4x4
        # v NOTE THIS v
        nn.AdaptiveAvgPool2d((1, 1)),  # B x 256 x 1 x 1
        # ^ NOTE THIS ^
        nn.Flatten(),  # B x 256
        nn.Linear(256, 200)
    ).to(device)
    
    return model

def get_optimizer(model):
    """
    Create an optimizer object for `model`, tuned for `train_on_tinyimagenet()`.

    return:
    optimizer:
        `torch.optim.Optimizer`
    """
    optimizer = config['optimizer'](model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    
    return optimizer

def predict(model, batch):
    """
    model:
        `torch.nn.Module`
        The neural net, as defined by `get_model()`.
    batch:
        unspecified
        A batch of Tiny ImageNet images, as yielded by `get_dataloader(..., 'val')`
        (with same preprocessing and device).

    return:
    prediction:
        `torch.tensor`, shape == (N, 200), dtype == `torch.float32`
        The scores of each input image to belong to each of the dataset classes.
        Namely, `prediction[i, j]` is the score of `i`-th minibatch sample to
        belong to `j`-th class.
        These scores can be 0..1 probabilities, but for better numerical stability
        they can also be raw class scores after the last (usually linear) layer,
        i.e. BEFORE softmax.
    """
    model.eval()
    batch = batch.to(device)
    
    with torch.no_grad():
        prediction = model(batch)
        
    return prediction

def validate(dataloader, model):
    """
    Run `model` through all samples in `dataloader`, compute accuracy and loss.

    dataloader:
        `torch.utils.data.DataLoader` or an object with equivalent interface
        See `get_dataloader()`.
    model:
        `torch.nn.Module`
        See `get_model()`.

    return:
    accuracy:
        `float`
        The fraction of samples from `dataloader` correctly classified by `model`
        (top-1 accuracy). `0.0 <= accuracy <= 1.0`
    loss:
        `float`
        Average loss over all `dataloader` samples.
    """
    model.eval()
    model.to(device)
    losses = []
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            xs, ys_true = batch

            ys_pred = model(xs.to(device))
            loss = loss_fn(ys_pred, ys_true.to(device))

            ys_pred = ys_pred.to(device="cpu")
            _, predicted_labels = torch.max(ys_pred, dim=1)
            total_correct += (predicted_labels == ys_true).sum().item()
            total_samples += ys_true.size(0)
            
            losses.append(loss.detach().cpu().item())

    # Средняя ошибка и точность
    accuracy = total_correct / total_samples

    return accuracy, np.mean(losses)


def run_epoch(stage, model, dataloader, loss_fn, optimizer, scheduler, epoch, device):
    # v NOTE THIS v
    if stage == "train":
        model.train()
        torch.set_grad_enabled(True)
    else:
        torch.set_grad_enabled(False)
        model.eval()
    # ^ NOTE THIS ^

    model = model.to(device)
    num_steps = len(dataloader)
    losses = []

    total_correct = 0
    total_samples = 0

    for i, batch in enumerate(tqdm.tqdm(dataloader, total=len(dataloader), desc=f"epoch: {str(epoch).zfill(3)} | {stage:5}")):
        xs, ys_true = batch

        ys_pred = model(xs.to(device))
        loss = loss_fn(ys_pred, ys_true.to(device))

        ys_pred = ys_pred.to(device="cpu")
        _, predicted_labels = torch.max(ys_pred, dim=1)
        total_correct += (predicted_labels == ys_true).sum().item()
        total_samples += ys_true.size(0)

        if stage == "train":
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            # логгируем номер шага при обучении
            wandb.log({
                "training_step": i + num_steps * epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "loss": loss,
            })

        losses.append(loss.detach().cpu().item())

    epoch_accuracy = total_correct / total_samples

    return np.mean(losses), epoch_accuracy

def save_checkpoint(model, optimizer, scheduler, epoch, loss, filename):
    """Сохранение состояния модели, оптимизатора, номера эпохи и потерь в файл."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }
    with open(filename, "wb") as fp:
        torch.save(checkpoint, fp)

def load_checkpoint(model,optimizer,scheduler,device,filename):
    """Загрузка состояния модели, оптимизатора, номера эпохи и потерь из файла."""
    with open(filename, "rb") as fp:
        checkpoint = torch.load(fp, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return epoch, loss

output_dir = Path("content/checkpoints_tr_demonstration")

def train_on_tinyimagenet(dataloader_train, dataloader_val, model, optimizer):
    """
    Train `model` on `train_dataloader` using `optimizer`. Use best-accuracy settings.

    train_dataloader:
    val_dataloader:
        See `get_dataloader()`.
    model:
        See `get_model()`.
    optimizer:
        See `get_optimizer()`.
    """
    wandb.login()
    wandb.init(
        project="HW2",
        name="training demonstration",
        reinit=True,
        config=config,
    )
    
    scheduler = config['scheduler'](optimizer, lr_lambda=lambda epoch: 1.0)

    best_val_loss = np.inf
    best_val_loss_epoch = -1
    best_val_loss_fn = None

    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(config["num_epochs"]):
        train_loss, train_accuracy = run_epoch("train", model, dataloader_train, loss_fn, optimizer, scheduler, epoch, device)

        val_loss, val_accuracy = run_epoch("val", model, dataloader_val, loss_fn, optimizer, None, epoch, device)

        wandb.log({"epoch_loss_train": train_loss, "epoch_loss_val": val_loss, "epoch_accuracy_train": train_accuracy, "epoch_accuracy_val": val_accuracy, "epoch": epoch})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_loss_epoch = epoch

            output_fn = os.path.join(output_dir, f"epoch={str(epoch).zfill(2)}_valloss={best_val_loss:.3f}.pth.tar")
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, output_fn)
            print(f"New checkpoint saved to {output_fn}\n")

            best_val_loss_fn = output_fn

    print(f"Best val_loss = {best_val_loss:.3f} reached at epoch {best_val_loss_epoch}")
    load_checkpoint(model, optimizer, scheduler, device, best_val_loss_fn)

def load_weights(model, checkpoint_path):
    """
    Initialize `model`'s weights from `checkpoint_path` file.

    model:
        `torch.nn.Module`
        See `get_model()`.
    checkpoint_path:
        `str`
        Path to the checkpoint.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint)
    
    return model

def get_checkpoint_metadata():
    """
    Return hard-coded metadata for 'checkpoint.pth'.
    Very important for grading.

    return:
    md5_checksum:
        `str`
        MD5 checksum for the submitted 'checkpoint.pth'.
        On Linux (in Colab too), use `$ md5sum checkpoint.pth`.
        On Windows, use `> CertUtil -hashfile checkpoint.pth MD5`.
        On Mac, use `$ brew install md5sha1sum`.
    google_drive_link:
        `str`
        View-only Google Drive link to the submitted 'checkpoint.pth'.
        The file must have the same checksum as in `md5_checksum`.
    """
    md5_checksum = "aabc7b9ac22ec3dfc357a042ef5730fe"
    google_drive_link = "https://drive.google.com/file/d/1Sc12f0Z1vvsUujKDxq__if4ferkfWXwp/view?usp=sharing"

    return md5_checksum, google_drive_link
