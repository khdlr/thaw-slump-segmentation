import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from deep_learning import Trainer
from deep_learning.models import UNet
from data_loading import get_loaders


def showexample(batch, pred, idx, filename):
    m = 0.02
    gridspec_kw = dict(left=m, right=1-m, top=1-m, bottom=m, hspace=m, wspace=m)
    fig, ((a1, a2), (a3, a4)) = plt.subplots(2, 2, figsize=(8, 8), gridspec_kw=gridspec_kw)
    batch_img, batch_target = batch
    batch_img = batch_img.to(torch.float)

    rgb = batch_img[idx].cpu().numpy()[[2, 1, 0]]
    a1.imshow(np.clip(rgb.transpose(1, 2, 0), 0, 1))
    a1.axis('off')
    a2.imshow(batch_target[idx, 0].cpu(), cmap='coolwarm', vmin=0, vmax=1)
    a2.axis('off')
    tcvis = batch_img[idx].cpu().numpy()[[4, 5, 6]]
    a3.imshow(np.clip(tcvis.transpose(1, 2, 0), 0, 1))
    a3.axis('off')
    a4.imshow(torch.sigmoid(pred[idx, 0]).cpu(), cmap='coolwarm', vmin=0, vmax=1)
    # a4.imshow((pred[idx, 0] > 0).cpu(), cmap='coolwarm', vmin=0, vmax=1)
    a4.axis('off')
    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":
    model = UNet(7, 1, base_channels=8)
    trainer = Trainer(model)
    trainer.loss_function = nn.BCEWithLogitsLoss(pos_weight=150 * torch.ones([])).to(trainer.dev)
    trainer.optimizer = torch.optim.Adam(trainer.model.parameters(), 1e-4)
    # Print model summary
    # summary(trainer.model, [(7, 256, 256)])
    train_loader, val_loader = get_loaders(batch_size=32, augment=True)

    interesting_tiles = [129, 92, 332, 169, 142, 424]
    interesting_batch = list(zip(*[val_loader.dataset[i] for i in interesting_tiles]))
    interesting_batch = [torch.stack(i, dim=0) for i in interesting_batch]
    interesting_imgs = interesting_batch[0].to(trainer.dev)

    for epoch in range(20):
        trainer.train_epoch(train_loader)
        trainer.val_epoch(val_loader)

        with torch.no_grad():
            pred = trainer.model(interesting_imgs)
        for i, idx in enumerate(interesting_tiles):
            filename = f'logs/{idx}_{trainer.epoch}.png'
            showexample(interesting_batch, pred, i, filename)

