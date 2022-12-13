from PIL import Image
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from .UnifiedLoader import UnifiedLoader

from .bases import ImageDataset
from .preprocessing import RandomErasing
from .sampler import RandomIdentitySampler


def train_collate_fn(batch):
    """ """
    imgs, pids, _, = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)

    return torch.stack(imgs, dim=0), pids



def val_collate_fn(batch):
    imgs, pids, img_paths = zip(*batch)
    return torch.stack(imgs, dim=0), pids,  img_paths


def make_dataloader(cfg):
    train_transforms = T.Compose([
        T.Resize(cfg.INPUT_SIZE),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),

        ### center crop?
        T.RandomCrop([256, 768]),
        T.RandomRotation(12, resample=Image.BICUBIC, expand=False, center=None),
        
        T.ToTensor(),
        T.Normalize(mean=[0.850, 0.850, 0.850], std=[0.0217, 0.0217, 0.0217]),
        #RandomErasing(probability=0.5, sh=0.4, mean=(0.4914, 0.4822, 0.4465))
    ])

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT_SIZE),
        #T.CenterCrop([256, 768]),
        T.ToTensor(),
        T.Normalize(mean=[0.850, 0.850, 0.850], std=[0.0217, 0.0217, 0.0217]),
    ])

    num_workers = cfg.DATALOADER_NUM_WORKERS

    dataroot = cfg.LOG_DIR
    istrain = cfg.ISTRAIN
    dataset = UnifiedLoader(dataset_name='fast',data_dir=dataroot, istrain=istrain, verbose=True)

    if istrain:
      num_classes = dataset.num_train_pids
    else:
      num_classes = dataset.num_test_pids
      if num_classes == 1:
        num_classes += 1;
    print('num_classes:', num_classes)

    if istrain == True:
      print('loading the train samples ...')

      train_set = ImageDataset(dataset.train, train_transforms)

      if cfg.group_wise:
          print('using multi-attention training')
          train_loader = DataLoader(train_set,
                                    batch_size=cfg.BATCH_SIZE,
                                    num_workers=num_workers,
                                    sampler=RandomIdentitySampler(dataset.train, cfg.BATCH_SIZE, cfg.NUM_IMG_PER_ID),
                                    collate_fn=train_collate_fn  # customized batch sampler
                                    )
      elif cfg.group_wise==False:
          print('using baseline training')
          train_loader = DataLoader(train_set,
                                    batch_size=cfg.BATCH_SIZE,
                                    shuffle=True,
                                    num_workers=num_workers,
                                    sampler=None,
                                    collate_fn=train_collate_fn,  # customized batch sampler
                                    drop_last=True
                                    )
      else:
          print('unsupported training strategy!   got {} for co-attention training'.format(cfg.CO_ATT))

      val_set = ImageDataset(dataset.test, val_transforms)
      val_loader = DataLoader(val_set,
                              batch_size=cfg.TEST_IMS_PER_BATCH,
                              shuffle=False, num_workers=num_workers,
                              collate_fn=val_collate_fn
                              )
      return train_loader, val_loader, len(dataset.test), num_classes

    else:
      print('loading the test samples ...')
      val_set = ImageDataset(dataset.test, val_transforms)
      val_loader = DataLoader(val_set,
                              batch_size=cfg.TEST_IMS_PER_BATCH,
                              shuffle=False, num_workers=num_workers,
                              collate_fn=val_collate_fn
                              )
      return val_loader, len(dataset.test), num_classes
