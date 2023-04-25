##

# PRO
# Image Crop -> Path List -> Pulsar Recognization #


import os
from torch.backends import cudnn

from config import Config
from datasets import make_dataloader
from model import make_model
from processor import do_inference
from utils.logger import setup_logger
from imgcrop import accrop, fastcrop
from path2txt import path2txt
import argparse
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


parser = argparse.ArgumentParser(description='FAST Pulsar Recognization.')

## For data path
parser.add_argument('--dataroot', required=False, default='D:/dzhao/CODE/RFI_CLASS/pulsar_class_resnet50/datadir/', help='path to data.')
parser.add_argument('--rawdir', required=False, default='gray_data/', help='subpath to raw data.')

## For raw image crop
parser.add_argument('--croptype', required=False, default='fast', help='Choose "fast" or "ac" for fast cropping (fast) or accurate cropping (ac).')
parser.add_argument('--cropregion', type=int, default=[35,829,1289,1568], help='If using the fast form, the region of the cropped subimage are set as [x1, x2, y1, y2] which indicate the start row, last row, start column and last column, respectively.')
parser.add_argument('--skipcrop', action='store_true', help='If true, skip the raw image crop processing.')

## For cropped image listing
parser.add_argument('--skiplist', action='store_true', help='If true, skip the image path listing.')

parser.add_argument('--istrain', action='store_true', help='if true, training the model; if false, testing.')
parser.add_argument('--model', required=False, default='D:/dzhao/CODE/RFI_CLASS/pulsar_class_resnet50/output/resnet50_30.pth', help='path of the pretrained model.')

parser.add_argument('--threshold', type=int, default=0, help='threshold of the score to define pulsar or not pulsar')
parser.add_argument('--pulsar_save', required=False, default='/pulsar_img', help='saved path of the detected pulsar images')
parser.add_argument('--nopulsar_save', required=False, default='/nopulsar_img', help='saved path of the detected no_pulsar images')

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
samp_path = path2txt(dataroot=args.dataroot, skip=args.skiplist)
print('samples are listed in the .txt in', samp_path)


if __name__ == "__main__":
    cfg = Config()
    cfg.LOG_DIR = crop_path
    cfg.DEVICE_ID = args.gpu_id
    cfg.TEST_WEIGHT = args.model
    cfg.ISTRAIN = args.istrain

    log_dir = args.dataroot
    logger = setup_logger('{}.test'.format(cfg.PROJECT_NAME), log_dir)
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.DEVICE_ID
    cudnn.benchmark = True

    # Set 'cfg' for your testing:
    # 

    

    val_loader, num_query, num_classes = make_dataloader(cfg)
    print('num_query:', num_query)
    model = make_model(cfg, num_classes)
    #model.load_param(cfg.TEST_WEIGHT)

    results_path = do_inference(cfg,
                 model,
                 val_loader,
                 num_query)


##
    count = 1
    all_count = 1
    filename = results_path
    Score_list = []
    with open(filename, "r") as ins: 
        for line in ins: 
            all_count += 1
            line = line.replace(".png"+"\t", ".png|") 
            line = line.replace("\t", "&")
            
            if '|'in line:
                equal_symbol_idx = line.index('|') 
                or_symbol_idx = line.index('&')
                
                count+=1
            else:
                continue

            score = float(line[or_symbol_idx+1:]) 
            name = line[:equal_symbol_idx]
            Score_list.append((score, name))

    Score_list = sorted(Score_list, key=lambda x: -x[0]) 

    predict_sortedpath = './predict_value'
    if not os.path.exists(predict_sortedpath):
        os.mkdir(predict_sortedpath)
    prediet_sortedtxt=open(predict_sortedpath+'/predict_sorted.txt','w')
    if not os.path.exists(predict_sortedpath + nopulsar_save):
        os.mkdir(predict_sortedpath + nopulsar_save)
    if not os.path.exists(predict_sortedpath + pulsar_save):
        os.mkdir(predict_sortedpath + pulsar_save)

    for ImagePath in Score_list: 
        prediet_sortedtxt.write('Score'+'\t'+str(ImagePath[0])+'\t'+'image'+'\t'+str(ImagePath[1])+'\n')
        img = cv2.imread(ImagePath[1])
        if ImagePath[0] > args.threshold:
            imgname = ImagePath[1].split("/")
            cv2.imwrite(predict_sortedpath + pulsar_save + '/' + imgname[-1], img)
        else:
            imgname = ImagePath[1].split("/")
            cv2.imwrite(predict_sortedpath + nopulsar_save + '/' + imgname[-1], img)


    prediet_sortedtxt.close()






