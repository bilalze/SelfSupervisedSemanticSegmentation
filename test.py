import torch
# from torch._C import device
# from torchinfo import summary
from torchvision.transforms.functional import InterpolationMode
import network
from PIL import Image
from torchvision import transforms as T
# from datasets import VOCSegmentation
from torch.utils import data
import utils
from network.utils import WithinImageLoss,WithinImageLoss1
import math
import torch.nn.functional as F
from dataloader_mine import XviewsDataset
from utils.visualizer import Visualizer
# # model = network.deeplabv3_resnet101(num_classes=21, output_stride=16)
# # model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3plus_resnet50', pretrained=True)
# # model.summary()
# # summary(model, (1,3, 513, 513))
def main():
    def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
        color_jitter = T.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
        rnd_color_jitter = T.RandomApply([color_jitter], p=0.8)
        rnd_gray = T.RandomGrayscale(p=0.2)
        color_distort = T.Compose([
        rnd_color_jitter,
        rnd_gray])
        return color_distort

    # vis = Visualizer(port=13579,
    #                  env='main') 
    device=torch.device('cuda')
    model = network.deeplabv3plus_resnet50(num_classes=1, output_stride=16,pretrained_backbone=True)
    network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3plus_resnet50', pretrained=True)
    # model.summary()
    # summary(model)  
    # summary(model, input_size=(1,3, 513, 513),col_names=["input_size","output_size"])
    # print(model)
    # for x in model.named_children():
    #     print(x)
    
    

    optimizer = torch.optim.SGD(params=[
            {'params': model.backbone.parameters(), 'lr': 0.1},
            {'params': model.classifier.parameters(), 'lr': 0.1},
        ], lr=0.1, momentum=0.9, weight_decay=4*math.exp(-5))

    transform = T.Compose([
                T.Resize([513,513]),
                T.CenterCrop([513,513]),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
    transform2=T.Compose([
        get_color_distortion(),
        transform
    ])

    transform3=T.Compose([
        T.Resize([22,22],T.InterpolationMode.NEAREST),
        T.ToTensor()
    ])

    train_dst=XviewsDataset('D:/Downloads/rdata512/rdata512/train_pre/images',
                                    'D:/Downloads/rdata512/rdata512/train_pre/targets',transform,transform2,transform3)
    batchsize=1
    train_loader = data.DataLoader(
            train_dst, batch_size=batchsize, shuffle=False)
    # val_loader = data.DataLoader(
    #         val_dst, batch_size=16, shuffle=False)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3)
    # criterion=WithinImageLoss(0.15)
    criterion=WithinImageLoss1(0.15)
    cur_epochs=0
    cur_itrs=0
    model = torch.nn.DataParallel(model)
    model.to(device)
    while True:
        model.train()
        cur_epochs += 1
        if cur_epochs>300:
            return

        interval_loss = 0
        counter=0
        for samples in train_loader:
            images=torch.cat((samples['image1'],samples['image2']),dim=0)
            labels=samples['label']
            cur_itrs += 1
            images = images.to(device, dtype=torch.float32)
            # print(images.shape)
            # optimizer.zero_grad()
            outputs = model(images)
            # print(outputs.shape)
            loss=0

            for count in range(batchsize):
                label1 = labels[count].to(device, dtype=torch.long)
                label2=label1
                loss += criterion(outputs[count],outputs[count+batchsize], label1,label2)
                del label1
                torch.cuda.empty_cache()
            loss=loss/(batchsize*8)
            # return
        
            loss.backward()
            counter+=1
            if counter==8:
                optimizer.step()
                
                counter=0
                print('yes')
                scheduler.step() 
                optimizer.zero_grad()

            np_loss = loss.detach().cpu().numpy()
            interval_loss += np_loss
            # if vis is not None:
            #     vis.vis_scalar('Loss', cur_itrs, np_loss)

            if (cur_itrs) % 10 == 0:
                interval_loss = interval_loss/10
                print("Epoch %d, Itrs %d/%d, Loss=%f" %
                        (cur_epochs, cur_itrs, 3000, interval_loss))
                interval_loss = 0.0

            if (cur_itrs) % 100 == 0:
                # save_ckpt('checkpoints/latest_%s_%s_os%d.pth' %
                #             (opts.model, opts.dataset, opts.output_stride))
                print("validation...")
                # print(np_loss)
                # model.eval()
                # val_score, ret_samples = validate(
                #     opts=opts, model=model, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
                # print(metrics.to_str(val_score))
                # if val_score['Mean IoU'] > best_score:  # save best model
                #     best_score = val_score['Mean IoU']
                #     save_ckpt('checkpoints/best_%s_%s_os%d.pth' %
                #                 (opts.model, opts.dataset,opts.output_stride))

                # if vis is not None:  # visualize validation score and samples
                    # vis.vis_scalar("[Val] Overall Acc", cur_itrs, val_score['Overall Acc'])
                    # vis.vis_scalar("[Val] Mean IoU", cur_itrs, val_score['Mean IoU'])
                    # vis.vis_table("[Val] Class IoU", val_score['Class IoU'])

                    # for k, (img, target, lbl) in enumerate(ret_samples):
                        # img = (denorm(img) * 255).astype(np.uint8)
                        # target = train_dst.decode_target(target).transpose(2, 0, 1).astype(np.uint8)
                        # lbl = train_dst.decode_target(lbl).transpose(2, 0, 1).astype(np.uint8)
                        # concat_img = np.concatenate((img, target, lbl), axis=2)  # concat along width
                        # vis.vis_image('Sample %d' % k, concat_img)
                # model.train()
            # scheduler.step()  

            if cur_itrs >=  300000:
                return


# model = network.deeplabv3plus_mobilenet(num_classes=21, output_stride=16)
# network.convert_to_separable_conv(model.classifier)
# # model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3plus_resnet50', pretrained=True)
# # model.summary()
# # summary(model)  
# summary(model, input_size=(1,3, 513, 513),col_names=["input_size","output_size"])
# # print(model)
# for x in model.classifier.named_children():
#     print(x)

# transform = T.Compose([
#             T.Resize([513,513]),
#             T.CenterCrop([513,513]),
#             T.ToTensor(),
#             T.Normalize(mean=[0.485, 0.456, 0.406],
#                             std=[0.229, 0.224, 0.225]),
#         ])
# # model.load_state_dict( torch.load( 'resnet_50.pth' )['model_state']  )
# img = Image.open('test.jpg').convert('RGB')
# img = transform(img).unsqueeze(0) # To tensor of NCHW
# # img = img.to(device)
# model.eval()
# outputs = model(img)
# print(outputs)
# print(outputs.shape)
# preds = outputs.max(1)[1].detach().cpu().numpy()
# print(preds)
# print(preds.shape)
# decode_fn = VOCSegmentation.decode_target
# colorized_preds = decode_fn(preds).astype('uint8')
# colorized_preds = Image.fromarray(colorized_preds[0])
# colorized_preds.save('test2.png')

if __name__ == '__main__':
    main()