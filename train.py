import os
from torch.backends import cudnn
import torch
from config import Config
from utils.logger import setup_logger
from datasets import make_dataloader
from model import make_model
from solver import make_optimizer, WarmupMultiStepLR
from loss import make_loss
from processor import do_train
from imgcrop import accrop, fastcrop
from path2txt import path2txt
import argparse
import numpy as np
#from thop import profile
from ptflops import get_model_complexity_info

parser = argparse.ArgumentParser(description='FAST Pulsar Recognization.')

## For data path
parser.add_argument('--dataroot', required=False, default='./datadir', help='path to data.')
parser.add_argument('--rawdir', required=False, default='/train', help='subpath to raw data.')

## For raw image crop
parser.add_argument('--croptype', required=False, default='fast', help='Choose "fast" or "ac" for fast cropping (fast) or accurate cropping (ac).')
parser.add_argument('--cropregion', type=int, default=[35,829,1289,1568], help='If using the fast form, the region of the cropped subimage are set as [x1, x2, y1, y2] which indicate the start row, last row, start column and last column, respectively.')
parser.add_argument('--skipcrop', action='store_true', help='If true, skip the raw image crop processing.')

## For cropped image listing
parser.add_argument('--skiplist', action='store_true', help='If true, skip the image path listing.')

parser.add_argument('--istrain', action='store_true', help='if true, training the model; if false, testing.')
parser.add_argument('--model', required=False, default='D:/dzhao/CODE/RFI_CLASS/pulsar_class_resnet50/output/resnet50_30.pth', help='path of the pretrained model.')
parser.add_argument('--gpu_id', type=str, default="0", help='Number of the GPU.')
args = parser.parse_args()


if args.croptype == 'ac':
    crop_path = accrop(dataroot=args.dataroot, rawdir=args.rawdir, skip=args.skipcrop)
elif args.croptype == 'fast':
    crop_path = fastcrop(dataroot=args.dataroot, rawdir=args.rawdir, region=args.cropregion, skip=args.skipcrop) 
else:
    print('Wrong croptype! Croptype should be "ac" or "fast".')
print('cropped samples are saved in', crop_path)

#print('Data Listing is Begining!')
if not args.istrain:
    samp_path = path2txt(dataroot=args.dataroot, skip=args.skiplist)
    print('samples are listed in the .txt in', samp_path)





if __name__ == '__main__':
    cfg = Config()
    if not args.istrain:
        cfg.LOG_DIR = crop_path
    else:
        cfg.LOG_DIR = args.dataroot 
    cfg.DEVICE_ID = args.gpu_id
    cfg.TEST_WEIGHT = args.model
    cfg.ISTRAIN = args.istrain

    if not os.path.exists(cfg.LOG_DIR):
        os.mkdir(cfg.LOG_DIR)
    logger = setup_logger('{}'.format(cfg.PROJECT_NAME), cfg.LOG_DIR)
    logger.info("Running with config:\n{}".format(cfg.CFG_NAME))
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.DEVICE_ID

    cudnn.benchmark = True
    # This flag allows you to enable the inbuilt cudnn auto-tuner to find the best algorithm to use for your hardware.

    

    train_loader, val_loader, num_query, num_classes = make_dataloader(cfg)
    model = make_model(cfg, num_class=num_classes)
    
    #flops, params = profile(model, input_size=(1, 3, 256,768))
    #print('FLOPs = ' + str(flops/1000**3) + 'G')
    #print('Params = ' + str(params/1000**2) + 'M')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model1 = model.to(device)
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model1, (3, 256, 768), as_strings=True, print_per_layer_stat=True, verbose=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)

    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)
    scheduler = WarmupMultiStepLR(optimizer, cfg.STEPS, cfg.GAMMA,
                                  cfg.WARMUP_FACTOR,
                                  cfg.WARMUP_EPOCHS, cfg.WARMUP_METHOD)
    

    do_train(
        cfg,
        model,
        center_criterion,
        train_loader,
        val_loader,
        optimizer,
        optimizer_center,
        scheduler,  # modify for using self trained model
        loss_func,
        num_query
    )
