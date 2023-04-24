import logging
import numpy as np
import os
import time
import torch
import torch.nn as nn

from utils.meter import AverageMeter
from utils.metrics import R1_mAP
from model.sync_batchnorm.replicate import *

def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query):
    log_period = cfg.LOG_PERIOD
    sleep_period = cfg.SLEEP_PERIOD
    checkpoint_period = cfg.CHECKPOINT_PERIOD
    eval_period = cfg.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.MAX_EPOCHS

    logger = logging.getLogger('{}.train'.format(cfg.PROJECT_NAME))
    logger.info('start training')

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
            patch_replication_callback(model)
        model.to(device)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP(num_query, max_rank=50, feat_norm=cfg.FEAT_NORM)
    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        scheduler.step()
        model.train()
        for n_iter, (img, vid) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            
            img = img.to(device)
            target = vid.to(device)


            cls_g = model(img, target)

            #cls_g = model(img, target)

            cls_g = cls_g.to(device)
            #cls_1 = cls_1.to(device)
            #cls_2 = cls_2.to(device)
            #cls_3 = cls_3.to(device)
            #cls_4 = cls_4.to(device)

            loss_g = loss_fn(cls_g, cls_g, target)
            #loss_1 = loss_fn(cls_1, cls_1, target)
            #loss_2 = loss_fn(cls_2, cls_2, target)
            #loss_3 = loss_fn(cls_3, cls_3, target)
            #loss_4 = loss_fn(cls_4, cls_4, target)

            #alpha = ((epoch + 1) / cfg.MAX_EPOCHS) * 1

            #loss_p = (loss_1 + loss_2 + loss_3) / 10
            loss = loss_g #+ loss_p #* alpha
            loss.backward()



            #score, feat = model(img, target)
            #loss = loss_fn(score, feat, target)


            optimizer.step()
            if 'center' in cfg.LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.CENTER_LOSS_WEIGHT)
                optimizer_center.step()

            acc = (cls_g.max(1)[1] == target).float().mean()
            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, acc_meter.avg, scheduler.get_lr()[0]))
            if (n_iter+1) % sleep_period == 0:
                print('begin time sleep')
                time.sleep(30)
                print('stop time sleep')

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if not os.path.exists(cfg.OUTPUT_DIR):
            os.mkdir(cfg.OUTPUT_DIR)

        if epoch % checkpoint_period == 0:
            torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.MODEL_NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            model.eval()
            for n_iter, (img, vid,_) in enumerate(val_loader):
                with torch.no_grad():
                    img = img.to(device)
                    cls_score = model(img)
                    #cls_score, cls_1, cls_2, cls_3,cls_4 = model(img)
                    #cls_score = model(img, target)
                    #score, feat = model(img)
                    evaluator.update((cls_score, vid))

            acc = evaluator.compute_acc()
            #cmc, mAP, _, _, _, _, _ = evaluator.compute()
            logger.info("Validation Results - Epoch: {}".format(epoch))
            logger.info("acc: {:.1%}".format(acc))



def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger('{}.test'.format(cfg.PROJECT_NAME))
    logger.info("Enter inferencing")
    evaluator = R1_mAP(num_query, max_rank=50, feat_norm=cfg.FEAT_NORM, \
                       method=cfg.TEST_METHOD, reranking=cfg.RERANKING)
    evaluator.reset()
    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)
        param_dict = torch.load(cfg.TEST_WEIGHT)
        # multi-gpu for parallel computing:
        if cfg.DEVICE_ID != '0':
            model.load_state_dict(param_dict)
        # single-gpu:
        if cfg.DEVICE_ID == '0':
            model.load_state_dict({k.replace('module.',''):v for k,v in torch.load(cfg.TEST_WEIGHT).items()})


    model.eval()
    img_path_list = []
    
    predict_savepath = '/predict_value'
    if not os.path.exists(predict_savepath):
        os.mkdir(predict_savepath)
    prediet_txt=open(predict_savepath+'/predict.txt','w')

    total = len(val_loader)
    iter_nurm = 0

    for n_iter, (img, pid, imgpath) in enumerate(val_loader):
        
        with torch.no_grad():
            img = img.to(device)

            if cfg.FLIP_FEATS == 'on':
                feat = torch.FloatTensor(img.size(0), 2048).zero_().cuda()
                for i in range(2):
                    if i == 1:
                        inv_idx = torch.arange(img.size(3) - 1, -1, -1).long().cuda()
                        img = img.index_select(3, inv_idx)
                    f = model(img)
                    feat = feat + f
            else:
                feat = model(img)

                for l in range(0, len(img)):

                    feat_4 = [float('{:.4f}'.format(i)) for i in feat[l]]
                    prediet_txt.write(str(imgpath[l])+'\t'+str(np.array(torch.tensor(feat_4[0], device='cpu')))+'\t'+str(np.array(torch.tensor(feat_4[1], device='cpu')))+'\n')

            iter_nurm = iter_nurm + 1
            step = int(100 / total * (iter_nurm + 1))
            str1 = '\r[%3d%%] %s' % (step, '>' * step)
            print(str1, end='', flush=True)

            evaluator.update((feat, pid))
            img_path_list.extend(imgpath)

        

    #cmc, mAP, distmat, pids, camids, qfeats, gfeats = evaluator.compute()

    prediet_txt.close()
    acc = evaluator.compute_acc()
    # cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("acc: {:.1%}".format(acc))

    return predict_savepath+'/predict.txt'

    # np.save(os.path.join(cfg.LOG_DIR, cfg.DIST_MAT) , distmat)
    # np.save(os.path.join(cfg.LOG_DIR, cfg.PIDS), pids)
    # np.save(os.path.join(cfg.LOG_DIR, cfg.CAMIDS), camids)
    # np.save(os.path.join(cfg.LOG_DIR, cfg.IMG_PATH), img_path_list[num_query:])
    # torch.save(qfeats, os.path.join(cfg.LOG_DIR, cfg.Q_FEATS))
    # torch.save(gfeats, os.path.join(cfg.LOG_DIR, cfg.G_FEATS))

    # logger.info("Validation Results")
    # logger.info("mAP: {:.1%}".format(mAP))
    # for r in [1, 5, 10]:
    #     logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
