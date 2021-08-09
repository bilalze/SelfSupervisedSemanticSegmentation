from torch.utils.data import dataset
from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np


from torch.utils import data
# from datasets import VOCSegmentation, Cityscapes
from utils import ext_transforms as et
from metrics import StreamSegMetrics
from dataloader_mine2 import XviewsDataset2,creator
import torch
import torch.nn as nn
from utils.visualizer import Visualizer
import math
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
# from deeplab2 import




# def get_dataset(opts):
#     """ Dataset And Augmentation
#     """
#     if opts.dataset == 'voc':
#         train_transform = et.ExtCompose([
#             et.ExtResize(size=opts.crop_size),
#             et.ExtRandomScale((0.5, 2.0)),
#             et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
#             et.ExtRandomHorizontalFlip(),
#             et.ExtToTensor(),
#             et.ExtNormalize(mean=[0.485, 0.456, 0.406],
#                             std=[0.229, 0.224, 0.225]),
#         ])
#         if opts.crop_val:
#             val_transform = et.ExtCompose([
#                 et.ExtResize(opts.crop_size),
#                 et.ExtCenterCrop(opts.crop_size),
#                 et.ExtToTensor(),
#                 et.ExtNormalize(mean=[0.485, 0.456, 0.406],
#                                 std=[0.229, 0.224, 0.225]),
#             ])
#         else:
#             val_transform = et.ExtCompose([
#                 et.ExtToTensor(),
#                 et.ExtNormalize(mean=[0.485, 0.456, 0.406],
#                                 std=[0.229, 0.224, 0.225]),
#             ])
#         train_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
#                                     image_set='train', download=opts.download, transform=train_transform)
#         val_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
#                                   image_set='val', download=False, transform=val_transform)

#     if opts.dataset == 'cityscapes':
#         train_transform = et.ExtCompose([
#             #et.ExtResize( 512 ),
#             et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
#             et.ExtColorJitter( brightness=0.5, contrast=0.5, saturation=0.5 ),
#             et.ExtRandomHorizontalFlip(),
#             et.ExtToTensor(),
#             et.ExtNormalize(mean=[0.485, 0.456, 0.406],
#                             std=[0.229, 0.224, 0.225]),
#         ])

#         val_transform = et.ExtCompose([
#             #et.ExtResize( 512 ),
#             et.ExtToTensor(),
#             et.ExtNormalize(mean=[0.485, 0.456, 0.406],
#                             std=[0.229, 0.224, 0.225]),
#         ])

#         train_dst = Cityscapes(root=opts.data_root,
#                                split='train', transform=train_transform)
#         val_dst = Cityscapes(root=opts.data_root,
#                              split='val', transform=val_transform)
#     return train_dst, val_dst


def validate( model, loader, device, metrics, ret_samples_ids=None):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []
    if True:
        if not os.path.exists('results'):
            os.mkdir('results')
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
        img_id = 0

    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):
            
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)
            if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
                ret_samples.append(
                    (images[0].detach().cpu().numpy(), targets[0], preds[0]))

            if True:
                for i in range(len(images)):
                    image = images[i].detach().cpu().numpy()
                    target = targets[i]
                    pred = preds[i]

                    image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                    target = loader.dataset.decode_target(target).astype(np.uint8)
                    pred = loader.dataset.decode_target(pred).astype(np.uint8)

                    Image.fromarray(image).save('results/%d_image.png' % img_id)
                    Image.fromarray(target).save('results/%d_target.png' % img_id)
                    Image.fromarray(pred).save('results/%d_pred.png' % img_id)

                    fig = plt.figure()
                    plt.imshow(image)
                    plt.axis('off')
                    plt.imshow(pred, alpha=0.7)
                    ax = plt.gca()
                    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    plt.savefig('results/%d_overlay.png' % img_id, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    img_id += 1

        score = metrics.get_results()
    return score, ret_samples


def main(ckpt):
    # opts = get_argparser().parse_args()
    # if opts.dataset.lower() == 'voc':
    #     opts.num_classes = 21
    # elif opts.dataset.lower() == 'cityscapes':
    #     opts.num_classes = 19

    # Setup visualization
    # vis = Visualizer(port=opts.vis_port,
    #                  env=opts.vis_env) if opts.enable_vis else None
    vis=None
    # if vis is not None:  # display options
        # vis.vis_table("Options", vars(opts))

    # os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup random seed
    # torch.manual_seed(opts.random_seed)
    # np.random.seed(opts.random_seed)
    # random.seed(opts.random_seed)

    # Setup dataloader
    # if opts.dataset=='voc' and not opts.crop_val:
        # opts.val_batch_size = 1
    train_transform = et.ExtCompose([
            et.ExtResize( (513,513) ),
            et.ExtRandomCrop(size=(513,513)),
            et.ExtColorJitter( brightness=0.5, contrast=0.5, saturation=0.5 ),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

    val_transform = et.ExtCompose([
            et.ExtResize( (513,513) ),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
    train_dst, val_dst = creator('D:/Downloads/rdata512/rdata512/train_pre/images',
                                    'D:/Downloads/rdata512/rdata512/train_pre/targets',train_transform,val_transform)
   
    # dataset
    train_loader = data.DataLoader(
        train_dst, batch_size=2, shuffle=True)
    val_loader = data.DataLoader(
        val_dst, batch_size=2, shuffle=True)
    # print("Dataset: %s, Train set: %d, Val set: %d" %
        #   (opts.dataset, len(train_dst), len(val_dst)))
    device=torch.device('cuda')
    model = network.deeplabv3plus_resnet50(num_classes=1, output_stride=16,pretrained_backbone=False)
    network.convert_to_separable_conv(model.classifier)
    # Set up model
    # utils.set_bn_momentum(model.backbone, momentum=0.01)
    
    # Set up metrics
    metrics = StreamSegMetrics(1)

    # Set up optimizer
    # optimizer = torch.optim.SGD(params=[
    #     {'params': model.backbone.parameters(), 'lr': 0.007},
    #     {'params': model.classifier.parameters(), 'lr': 0.007},
    # ], lr=0.007, momentum=0.9, weight_decay=4*math.exp(-5))
    # #optimizer = torch.optim.SGD(params=model.parameters(), lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    # #torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.lr_decay_step, gamma=opts.lr_decay_factor)
    # # if opts.lr_policy=='poly':
    # scheduler = utils.PolyLR(optimizer, 600, power=0.9)
    # elif opts.lr_policy=='step':
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)

    # Set up criterion
    #criterion = utils.get_loss(opts.loss_type)
    # if opts.loss_type == 'focal_loss':
        # criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    # elif opts.loss_type == 'cross_entropy':
    criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

    def save_ckpt(path):
        """ save current model
        """
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)
    
    utils.mkdir('checkpoints')
    # Restore
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0
    if ckpt is not None and os.path.isfile(ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model2=network._deeplab.DeepLabV3_self2(model,network._deeplab.classifier2(1))
        optimizer = torch.optim.SGD(params=[
        {'params': model2.model.backbone.parameters(), 'lr': 0.007},
        {'params': model2.model.classifier.parameters(), 'lr': 0.007},
        ], lr=0.007, momentum=0.9, weight_decay=4*math.exp(-5))
        scheduler = utils.PolyLR(optimizer, 600, power=0.9)
        model2 = nn.DataParallel(model2)
        model2.to(device)
        # if opts.continue_training:
            # optimizer.load_state_dict(checkpoint["optimizer_state"])
            # scheduler.load_state_dict(checkpoint["scheduler_state"])
            # cur_itrs = checkpoint["cur_itrs"]
            # best_score = checkpoint['best_score']
            # print("Training state restored from %s" % opts.ckpt)
        # print("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

    #==========   Train Loop   ==========#
    # vis_sample_id = np.random.randint(0, len(val_loader), opts.vis_num_samples,
    #                                   np.int32) if opts.enable_vis else None  # sample idxs for visualization
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images

    # if opts.test_only:
    #     model.eval()
    #     val_score, ret_samples = validate(
    #         opts=opts, model=model, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
    #     print(metrics.to_str(val_score))
    #     return

    interval_loss = 0
    while True: #cur_itrs < opts.total_itrs:
        # =====  Train  =====
        model2.train()
        cur_epochs += 1
        for (images, labels) in train_loader:
            cur_itrs += 1
            print(images.shape)
            print(labels.shape)
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            # print(torch.count_nonzero(labels))
            optimizer.zero_grad()
            outputs = model2(images)
            print(outputs.shape)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            np_loss = loss.detach().cpu().numpy()
            interval_loss += np_loss
            if vis is not None:
                vis.vis_scalar('Loss', cur_itrs, np_loss)

            if (cur_itrs) % 10 == 0:
                interval_loss = interval_loss/10
                print("Epoch %d, Itrs %d/%d, Loss=%f" %
                      (cur_epochs, cur_itrs,600, interval_loss))
                interval_loss = 0.0

            if (cur_itrs) % 100 == 0:
                save_ckpt('checkpoints/latest_%s_%s_os%d.pth' %
                          ('deeplabv3_resnet50','xview','os167'))
                print("validation...")
                model2.eval()
                val_score, ret_samples = validate(
                     model=model2, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=None)
                print(metrics.to_str(val_score))
                if val_score['Mean IoU'] > best_score:  # save best model
                    best_score = val_score['Mean IoU']
                    save_ckpt('checkpoints/best_%s_%s_os%d.pth' %
                              ('deeplabv3_resnet50','xview','os167'))

                if vis is not None:  # visualize validation score and samples
                    vis.vis_scalar("[Val] Overall Acc", cur_itrs, val_score['Overall Acc'])
                    vis.vis_scalar("[Val] Mean IoU", cur_itrs, val_score['Mean IoU'])
                    vis.vis_table("[Val] Class IoU", val_score['Class IoU'])

                    for k, (img, target, lbl) in enumerate(ret_samples):
                        img = (denorm(img) * 255).astype(np.uint8)
                        target = train_dst.decode_target(target).transpose(2, 0, 1).astype(np.uint8)
                        lbl = train_dst.decode_target(lbl).transpose(2, 0, 1).astype(np.uint8)
                        concat_img = np.concatenate((img, target, lbl), axis=2)  # concat along width
                        vis.vis_image('Sample %d' % k, concat_img)
                model2.train()
            scheduler.step()  

            if cur_itrs >=  600:
                return

        
if __name__ == '__main__':
    main('latest_deeplabv3_resnet50_xview_os166.pth')
