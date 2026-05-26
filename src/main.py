import warnings
warnings.filterwarnings("ignore", message="triton not found.*", module="torch.utils.flop_counter")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")

import random
import numpy as np
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

from data_module import LeukemiaDataModule
from lightning_model import LeukemiaLightningModel

try:
    from lightning.pytorch.callbacks import RichProgressBar
    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    set_seed(42)

    datamodule = LeukemiaDataModule(
        data_dir='dataset',
        batch_size=16,
        num_workers=2,
        n_segments=100,
        compactness=10,
    )
    datamodule.setup()

    print(f"Classes : {datamodule.classes}")
    print(f"Train   : {len(datamodule.train_dataset)}")
    print(f"Val     : {len(datamodule.val_dataset)}")

    model = LeukemiaLightningModel(
        num_classes=datamodule.num_classes,
        pretrained=True,
        lr=1e-4,
        weight_decay=1e-4,
    )

    checkpoint_cb = ModelCheckpoint(
        dirpath='checkpoints',
        filename='leukemia-{epoch:02d}-{val_loss:.4f}',
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        save_last=True,
    )

    callbacks = [checkpoint_cb]
    if _RICH_AVAILABLE:
        callbacks.append(RichProgressBar())

    trainer = L.Trainer(
        max_epochs=30,
        accelerator='auto',
        devices='auto',
        callbacks=callbacks,
        gradient_clip_val=1.0,
        log_every_n_steps=10,
    )

    trainer.fit(model, datamodule=datamodule)
    print(f"Best checkpoint: {checkpoint_cb.best_model_path}")


if __name__ == '__main__':
    main()
