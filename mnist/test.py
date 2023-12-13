import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import CIFAR10



class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(
            self,
            batch_size: int,
            num_workers: int = 0,
            pin_memory: bool = False,
            drop_last: bool = False,
            data_dir: str = "./"
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last

        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307, ), (0.3081,))])

    def prepare_data(self):
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            cifar_full = CIFAR10(self.data_dir, train=True, transform=self.transform)
            self.cifar_train, self.cifar_val = random_split(cifar_full, [45000, 5000])
        if stage == "test" or stage is None:
            self.cifar_test = CIFAR10(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self)->DataLoader:
        return DataLoader(
            self.cifar_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            shuffle=True
        )
    
    def val_dataloader(self)->DataLoader:
        return DataLoader(
            self.cifar_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            shuffle=False
        )
    
    def test_dataloader(self)->DataLoader:
        return DataLoader(
            self.cifar_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            shuffle=False
        )

cifar = CIFAR10DataModule(
    batch_size=64,
    num_workers=8,
    pin_memory=True,
    drop_last=True,
    data_dir="./data/cifar10",
)
cifar.setup(stage="test")
train_dataloader = cifar.test_dataloader()
print(train_dataloader)