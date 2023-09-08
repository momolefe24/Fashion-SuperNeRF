import torch
import torch.nn as nn
import yaml
from torchvision.utils import make_grid
from VITON.networks import make_grid as mkgrid

import argparse
import os
import time
from dataset import FashionDataLoader, FashionNeRFDataset
from VITON.cp_dataset import CPDatasetTest, CPDataLoader
from VITON.networks import ConditionGenerator, VGGLoss, GANLoss, load_checkpoint, save_checkpoint, define_D
from tqdm import tqdm
from tensorboardX import SummaryWriter
from VITON.utils import *
from torch.utils.data import Subset
import matplotlib.pyplot as plt


to = lambda inputs,name,i: inputs[name][i].permute(1,2,0).cpu().numpy()
to2 = lambda inputs,name1,name2, i: inputs[name1][name2][i].permute(1,2,0).cpu().numpy()
to3 = lambda inputs,i: inputs[i].permute(1,2,0).detach().cpu().numpy()
def iou_metric(y_pred_batch, y_true_batch):
    B = y_pred_batch.shape[0]
    iou = 0
    for i in range(B):
        y_pred = y_pred_batch[i]
        y_true = y_true_batch[i]
        # y_pred is not one-hot, so need to threshold it
        y_pred = y_pred > 0.5
        
        y_pred = y_pred.flatten()
        y_true = y_true.flatten()

    
        intersection = torch.sum(y_pred[y_true == 1])
        union = torch.sum(y_pred) + torch.sum(y_true)

    
        iou += (intersection + 1e-7) / (union - intersection + 1e-7) / B
    return iou

def remove_overlap(seg_out, warped_cm):
    
    assert len(warped_cm.shape) == 4
    
    warped_cm = warped_cm - (torch.cat([seg_out[:, 1:3, :, :], seg_out[:, 5:, :, :]], dim=1)).sum(dim=1, keepdim=True) * warped_cm
    return warped_cm

def train_model(opt, train_loader, test_loader, validation_loader, tocg_curve_writer,tocg_images_writer, tocg, D):
    tocg.cuda()
    D.cuda()
    tocg.train()
    D.train()

    # criterion
    criterionL1 = nn.L1Loss()
    criterionVGG = VGGLoss(opt)
    if opt.tocg_fp16:
        criterionGAN = GANLoss(use_lsgan=True, tensor=torch.cuda.HalfTensor)
    else :
        if opt.cuda:
            criterionGAN = GANLoss(use_lsgan=True, tensor=torch.cuda.FloatTensor if opt.gpu_ids else torch.Tensor)
        else:
            criterionGAN = GANLoss(use_lsgan=True, tensor=torch.Tensor)

    # optimizer
    optimizer_G = torch.optim.Adam(tocg.parameters(), lr=opt.G_lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(D.parameters(), lr=opt.D_lr, betas=(0.5, 0.999))
    

    for step in tqdm(range(opt.load_step, opt.keep_step)):
        iter_start_time = time.time()
        inputs = train_loader.next_batch()

        # input1
        c_paired = inputs['cloth']['paired'].cuda()
        cm_paired = inputs['cloth_mask']['paired'].cuda()
        cm_paired = torch.FloatTensor((cm_paired.detach().cpu().numpy() > 0.5).astype(np.float32)).cuda()
        # input2
        parse_agnostic = inputs['parse_agnostic'].cuda()
        densepose = inputs['densepose'].cuda()
        openpose = inputs['pose'].cuda()
        # GT
        label_onehot = inputs['parse_onehot'].cuda()  # CE
        label = inputs['parse'].cuda()  # GAN loss
        parse_cloth_mask = inputs['pcm'].cuda()  # L1
        im_c = inputs['parse_cloth'].cuda()  # VGG
        # visualization
        im = inputs['image']

        # inputs
        input1 = torch.cat([c_paired, cm_paired], 1)
        input2 = torch.cat([parse_agnostic, densepose], 1)

        # forward
        flow_list, fake_segmap, warped_cloth_paired, warped_clothmask_paired = tocg(opt, input1, input2)

        warped_cm_onehot = torch.FloatTensor((warped_clothmask_paired.detach().cpu().numpy() > 0.5).astype(np.float32)).cuda()
        # fake segmap cloth channel * warped clothmask
        if opt.clothmask_composition != 'no_composition':
            if opt.clothmask_composition == 'detach':
                cloth_mask = torch.ones_like(fake_segmap.detach())
                cloth_mask[:, 3:4, :, :] = warped_cm_onehot
                fake_segmap = fake_segmap * cloth_mask
                
            if opt.clothmask_composition == 'warp_grad':
                cloth_mask = torch.ones_like(fake_segmap.detach())
                cloth_mask[:, 3:4, :, :] = warped_clothmask_paired
                fake_segmap = fake_segmap * cloth_mask
        if opt.occlusion:
            warped_clothmask_paired = remove_overlap(F.softmax(fake_segmap, dim=1), warped_clothmask_paired)
            warped_cloth_paired = warped_cloth_paired * warped_clothmask_paired + torch.ones_like(warped_cloth_paired) * (1-warped_clothmask_paired)

        # generated fake cloth mask & misalign mask
        fake_clothmask = (torch.argmax(fake_segmap.detach(), dim=1, keepdim=True) == 3).long()
        misalign = fake_clothmask - warped_cm_onehot
        misalign[misalign < 0.0] = 0.0
        
        # loss warping
        loss_l1_cloth = criterionL1(warped_clothmask_paired, parse_cloth_mask)
        loss_vgg = criterionVGG(warped_cloth_paired, im_c)

        loss_tv = 0
        
        if opt.edgeawaretv == 'no_edge':
            if not opt.lasttvonly:
                for flow in flow_list:
                    y_tv = torch.abs(flow[:, 1:, :, :] - flow[:, :-1, :, :]).mean()
                    x_tv = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :]).mean()
                    loss_tv = loss_tv + y_tv + x_tv
            else:
                for flow in flow_list[-1:]:
                    y_tv = torch.abs(flow[:, 1:, :, :] - flow[:, :-1, :, :]).mean()
                    x_tv = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :]).mean()
                    loss_tv = loss_tv + y_tv + x_tv
        else:
            if opt.edgeawaretv == 'last_only':
                flow = flow_list[-1]
                warped_clothmask_paired_down = F.interpolate(warped_clothmask_paired, flow.shape[1:3], mode='bilinear')
                y_tv = torch.abs(flow[:, 1:, :, :] - flow[:, :-1, :, :])
                x_tv = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :])
                mask_y = torch.exp(-150*torch.abs(warped_clothmask_paired_down.permute(0, 2, 3, 1)[:, 1:, :, :] - warped_clothmask_paired_down.permute(0, 2, 3, 1)[:, :-1, :, :]))
                mask_x = torch.exp(-150*torch.abs(warped_clothmask_paired_down.permute(0, 2, 3, 1)[:, :, 1:, :] - warped_clothmask_paired_down.permute(0, 2, 3, 1)[:, :, :-1, :]))
                y_tv = y_tv * mask_y
                x_tv = x_tv * mask_x
                y_tv = y_tv.mean()
                x_tv = x_tv.mean()
                loss_tv = loss_tv + y_tv + x_tv
                
            elif opt.edgeawaretv == 'weighted':
                for i in range(5):
                    flow = flow_list[i]
                    warped_clothmask_paired_down = F.interpolate(warped_clothmask_paired, flow.shape[1:3], mode='bilinear')
                    y_tv = torch.abs(flow[:, 1:, :, :] - flow[:, :-1, :, :])
                    x_tv = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :])
                    mask_y = torch.exp(-150*torch.abs(warped_clothmask_paired_down.permute(0, 2, 3, 1)[:, 1:, :, :] - warped_clothmask_paired_down.permute(0, 2, 3, 1)[:, :-1, :, :]))
                    mask_x = torch.exp(-150*torch.abs(warped_clothmask_paired_down.permute(0, 2, 3, 1)[:, :, 1:, :] - warped_clothmask_paired_down.permute(0, 2, 3, 1)[:, :, :-1, :]))
                    y_tv = y_tv * mask_y
                    x_tv = x_tv * mask_x
                    y_tv = y_tv.mean() / (2 ** (4-i))
                    x_tv = x_tv.mean() / (2 ** (4-i))
                    loss_tv = loss_tv + y_tv + x_tv

            if opt.add_lasttv:
                for flow in flow_list[-1:]:
                    y_tv = torch.abs(flow[:, 1:, :, :] - flow[:, :-1, :, :]).mean()
                    x_tv = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :]).mean()
                    loss_tv = loss_tv + y_tv + x_tv
            

        N, _, iH, iW = c_paired.size()
        # Intermediate flow loss
        if opt.interflowloss:
            for i in range(len(flow_list)-1):
                flow = flow_list[i]
                N, fH, fW, _ = flow.size()
                grid = mkgrid(N, iH, iW, opt)
                flow = F.interpolate(flow.permute(0, 3, 1, 2), size = c_paired.shape[2:], mode=opt.upsample).permute(0, 2, 3, 1)
                flow_norm = torch.cat([flow[:, :, :, 0:1] / ((fW - 1.0) / 2.0), flow[:, :, :, 1:2] / ((fH - 1.0) / 2.0)], 3)
                warped_c = F.grid_sample(c_paired, flow_norm + grid, padding_mode='border')
                warped_cm = F.grid_sample(cm_paired, flow_norm + grid, padding_mode='border')
                warped_cm = remove_overlap(F.softmax(fake_segmap, dim=1), warped_cm)
                loss_l1_cloth += criterionL1(warped_cm, parse_cloth_mask) / (2 ** (4-i))
                loss_vgg += criterionVGG(warped_c, im_c) / (2 ** (4-i))

        # loss segmentation
        # generator
        CE_loss = cross_entropy2d(fake_segmap, label_onehot.transpose(0, 1)[0].long())
        
        if opt.no_GAN_loss:
            loss_G = (10 * loss_l1_cloth + loss_vgg + opt.tvlambda * loss_tv) + (CE_loss * opt.CElamda)
            # step
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()
        
        else:
            fake_segmap_softmax = torch.softmax(fake_segmap, 1)

            pred_segmap = D(torch.cat((input1.detach(), input2.detach(), fake_segmap_softmax), dim=1))
            
            loss_G_GAN = criterionGAN(pred_segmap, True)
            
            if not opt.G_D_seperate:  
                # discriminator
                fake_segmap_pred = D(torch.cat((input1.detach(), input2.detach(), fake_segmap_softmax.detach()),dim=1))
                real_segmap_pred = D(torch.cat((input1.detach(), input2.detach(), label),dim=1))
                loss_D_fake = criterionGAN(fake_segmap_pred, False)
                loss_D_real = criterionGAN(real_segmap_pred, True)

                # loss sum
                loss_G = (10 * loss_l1_cloth + loss_vgg +opt.tvlambda * loss_tv) + (CE_loss * opt.CElamda + loss_G_GAN * opt.GANlambda)  # warping + seg_generation
                loss_D = loss_D_fake + loss_D_real

                # step
                optimizer_G.zero_grad()
                loss_G.backward()
                optimizer_G.step()
                
                optimizer_D.zero_grad()
                loss_D.backward()
                optimizer_D.step()
                
            else: # train G first after that train D
                # loss G sum
                loss_G = (10 * loss_l1_cloth + loss_vgg + opt.tvlambda * loss_tv) + (CE_loss * opt.CElamda + loss_G_GAN * opt.GANlambda)  # warping + seg_generation
                
                # step G
                optimizer_G.zero_grad()
                loss_G.backward()
                optimizer_G.step()
                
                # discriminator
                with torch.no_grad():
                    _, fake_segmap, _, _ = tocg(opt, input1, input2)
                fake_segmap_softmax = torch.softmax(fake_segmap, 1)
                
                # loss discriminator
                fake_segmap_pred = D(torch.cat((input1.detach(), input2.detach(), fake_segmap_softmax.detach()),dim=1))
                real_segmap_pred = D(torch.cat((input1.detach(), input2.detach(), label),dim=1))
                loss_D_fake = criterionGAN(fake_segmap_pred, False)
                loss_D_real = criterionGAN(real_segmap_pred, True)
                
                loss_D = loss_D_fake + loss_D_real
                
                optimizer_D.zero_grad()
                loss_D.backward()
                optimizer_D.step()
        # Vaildation
        if (step + 1) % 1 == 0:
            tocg.eval()
            iou_list = []
            with torch.no_grad():
                for cnt in range(20//opt.viton_batch_size):
                    inputs = validation_loader.next_batch()
                    if opt.cuda:
                        # input1
                        c_paired = inputs['cloth']['paired'].cuda()
                        cm_paired = inputs['cloth_mask']['paired'].cuda()
                        cm_paired = torch.FloatTensor((cm_paired.detach().cpu().numpy() > 0.5).astype(np.float32)).cuda()
                        # input2
                        parse_agnostic = inputs['parse_agnostic'].cuda()
                        densepose = inputs['densepose'].cuda()
                        openpose = inputs['pose'].cuda()
                        # GT
                        label_onehot = inputs['parse_onehot'].cuda()  # CE
                        label = inputs['parse'].cuda()  # GAN loss
                        parse_cloth_mask = inputs['pcm'].cuda()  # L1
                        im_c = inputs['parse_cloth'].cuda()  # VGG
                    else:
                        c_paired = inputs['cloth']['paired']
                        cm_paired = inputs['cloth_mask']['paired']
                        cm_paired = torch.FloatTensor(
                            (cm_paired.detach().cpu().numpy() > 0.5).astype(np.float32))
                        # input2
                        parse_agnostic = inputs['parse_agnostic']
                        densepose = inputs['densepose']
                        openpose = inputs['pose']
                        # GT
                        label_onehot = inputs['parse_onehot']  # CE
                        label = inputs['parse']  # GAN loss
                        parse_cloth_mask = inputs['pcm']  # L1
                        im_c = inputs['parse_cloth']  # VGG
                    # visualization
                    im = inputs['image']
                    
                    input1 = torch.cat([c_paired, cm_paired], 1)
                    input2 = torch.cat([parse_agnostic, densepose], 1)
                    
                    # forward
                    flow_list, fake_segmap, warped_cloth_paired, warped_clothmask_paired = tocg(opt, input1, input2)
                
                    # fake segmap cloth channel * warped clothmask
                    if opt.clothmask_composition != 'no_composition':
                        if opt.clothmask_composition == 'detach':
                            cloth_mask = torch.ones_like(fake_segmap.detach())
                            cloth_mask[:, 3:4, :, :] = warped_cm_onehot
                            fake_segmap = fake_segmap * cloth_mask
                            
                        if opt.clothmask_composition == 'warp_grad':
                            cloth_mask = torch.ones_like(fake_segmap.detach())
                            cloth_mask[:, 3:4, :, :] = warped_clothmask_paired
                            fake_segmap = fake_segmap * cloth_mask
    
                    # calculate iou
                    iou = iou_metric(F.softmax(fake_segmap, dim=1).detach(), label)
                    iou_list.append(iou.item())

            tocg.train()
            tocg_curve_writer.add_scalar('val/iou', np.mean(iou_list), step + 1)
        
        # tensorboard
        if (step + 1) % 1 == 0:
            # loss G
            tocg_curve_writer.add_scalar('Loss/G', loss_G.item(), step + 1)
            tocg_curve_writer.add_scalar('Loss/G/l1_cloth', loss_l1_cloth.item(), step + 1)
            tocg_curve_writer.add_scalar('Loss/G/vgg', loss_vgg.item(), step + 1)
            tocg_curve_writer.add_scalar('Loss/G/tv', loss_tv.item(), step + 1)
            tocg_curve_writer.add_scalar('Loss/G/CE', CE_loss.item(), step + 1)
            if not opt.no_GAN_loss:
                tocg_curve_writer.add_scalar('Loss/G/GAN', loss_G_GAN.item(), step + 1)
                # loss D
                tocg_curve_writer.add_scalar('Loss/D', loss_D.item(), step + 1)
                tocg_curve_writer.add_scalar('Loss/D/pred_real', loss_D_real.item(), step + 1)
                tocg_curve_writer.add_scalar('Loss/D/pred_fake', loss_D_fake.item(), step + 1)
            
            grid = make_grid([(c_paired[0].cpu() / 2 + 0.5), (cm_paired[0].cpu()).expand(3, -1, -1), visualize_segmap(parse_agnostic.cpu()), ((densepose.cpu()[0]+1)/2),
                              (im_c[0].cpu() / 2 + 0.5), parse_cloth_mask[0].cpu().expand(3, -1, -1), (warped_cloth_paired[0].cpu().detach() / 2 + 0.5), (warped_cm_onehot[0].cpu().detach()).expand(3, -1, -1),
                              visualize_segmap(label.cpu()), visualize_segmap(fake_segmap.cpu()), (im[0]/2 +0.5), (misalign[0].cpu().detach()).expand(3, -1, -1)],
                                nrow=4)
            tocg_images_writer.add_images('train_images', grid.unsqueeze(0), step + 1)
            
            if not opt.no_test_visualize:
                inputs = test_loader.next_batch()
                c_paired = inputs['cloth'][opt.test_datasetting].cuda()
                cm_paired = inputs['cloth_mask'][opt.test_datasetting].cuda()
                cm_paired = torch.FloatTensor((cm_paired.detach().cpu().numpy() > 0.5).astype(np.float32)).cuda()
                # input2
                parse_agnostic = inputs['parse_agnostic'].cuda()
                densepose = inputs['densepose'].cuda()
                openpose = inputs['pose'].cuda()
                # GT
                label_onehot = inputs['parse_onehot'].cuda()  # CE
                label = inputs['parse'].cuda()  # GAN loss
                parse_cloth_mask = inputs['pcm'].cuda()  # L1
                im_c = inputs['parse_cloth'].cuda()  # VGG
                # visualization
                im = inputs['image']

                tocg.eval()
                with torch.no_grad():
                    # inputs
                    input1 = torch.cat([c_paired, cm_paired], 1)
                    input2 = torch.cat([parse_agnostic, densepose], 1)

                    # forward
                    flow_list, fake_segmap, warped_cloth_paired, warped_clothmask_paired = tocg(opt, input1, input2)
                    warped_cm_onehot = torch.FloatTensor(
                        (warped_clothmask_paired.detach().cpu().numpy() > 0.5).astype(np.float32)).cuda()
                    if opt.clothmask_composition != 'no_composition':
                        if opt.clothmask_composition == 'detach':
                            cloth_mask = torch.ones_like(fake_segmap)
                            cloth_mask[:,3:4, :, :] = warped_cm_onehot
                            fake_segmap = fake_segmap * cloth_mask
                            
                        if opt.clothmask_composition == 'warp_grad':
                            cloth_mask = torch.ones_like(fake_segmap)
                            cloth_mask[:,3:4, :, :] = warped_clothmask_paired
                            fake_segmap = fake_segmap * cloth_mask
                    if opt.occlusion:
                        warped_clothmask_paired = remove_overlap(F.softmax(fake_segmap, dim=1), warped_clothmask_paired)
                        warped_cloth_paired = warped_cloth_paired * warped_clothmask_paired + torch.ones_like(warped_cloth_paired) * (1-warped_clothmask_paired)
                    
                    # generated fake cloth mask & misalign mask
                    fake_clothmask = (torch.argmax(fake_segmap.detach(), dim=1, keepdim=True) == 3).long()
                    misalign = fake_clothmask - warped_cm_onehot
                    misalign[misalign < 0.0] = 0.0
                
                for i in range(opt.num_test_visualize):
                    grid = make_grid([(c_paired[i].cpu() / 2 + 0.5), (cm_paired[i].cpu()).expand(3, -1, -1), visualize_segmap(parse_agnostic.cpu(), batch=i), ((densepose.cpu()[i]+1)/2),
                                    (im_c[i].cpu() / 2 + 0.5), parse_cloth_mask[i].cpu().expand(3, -1, -1), (warped_cloth_paired[i].cpu().detach() / 2 + 0.5), (warped_cm_onehot[i].cpu().detach()).expand(3, -1, -1),
                                    visualize_segmap(label.cpu(), batch=i), visualize_segmap(fake_segmap.cpu(), batch=i), (im[i]/2 +0.5), (misalign[i].cpu().detach()).expand(3, -1, -1)],
                                        nrow=4)
                    tocg_images_writer.add_images(f'test_images/{i}', grid.unsqueeze(0), step + 1)
                tocg.train()
        
        # display
        if (step + 1) % 1 == 0:
            t = time.time() - iter_start_time
            if not opt.no_GAN_loss:
                print("step: %8d, time: %.3f\nloss G: %.4f, L1_cloth loss: %.4f, VGG loss: %.4f, TV loss: %.4f CE: %.4f, G GAN: %.4f\nloss D: %.4f, D real: %.4f, D fake: %.4f"
                    % (step + 1, t, loss_G.item(), loss_l1_cloth.item(), loss_vgg.item(), loss_tv.item(), CE_loss.item(), loss_G_GAN.item(), loss_D.item(), loss_D_real.item(), loss_D_fake.item()), flush=True)

        # save
        if (step + 1) % 1 == 0:
            save_checkpoint(tocg,opt.tocg_save_step_checkpoint % (step + 1), opt)
            save_checkpoint(D,opt.tocg_discriminator_save_step_checkpoint % (step + 1), opt)

def train_condition_generator(train_dataset, opt):
    print("Start to train %s!" % opt.name)
    tocg_curve_writer = SummaryWriter(f"{opt.tocg_writer}/curves")
    tocg_image_writer = SummaryWriter(f"{opt.tocg_writer}/images")
    train_loader = FashionDataLoader(train_dataset, opt.viton_batch_size, opt.viton_workers, True)

    test_dataset = FashionNeRFDataset(opt, viton=True, mode='test', model='viton')
    test_loader = FashionDataLoader(test_dataset, opt.num_test_visualize, 1, False)
    validation_dataset = Subset(test_dataset, np.arange(50))
    validation_loader = FashionDataLoader(validation_dataset, opt.num_test_visualize, opt.viton_workers, False)
    if not os.path.exists(opt.tocg_writer):
        os.makedirs(opt.tocg_writer)
    if not os.path.exists(os.path.join(opt.tocg_basedir, opt.tocg_name)):
        os.makedirs(os.path.join(opt.tocg_basedir, opt.tocg_name))

    with open(f'{os.path.join(opt.tocg_basedir, opt.tocg_name)}/experiment.yml', 'w') as outfile:
        yaml.dump(vars(opt), outfile, default_flow_style=False)

    # Model
    input1_nc = 4  # cloth + cloth-mask
    input2_nc = opt.semantic_nc + 3  # parse_agnostic + densepose
    tocg = ConditionGenerator(opt, input1_nc=4, input2_nc=input2_nc, output_nc=opt.output_nc, ngf=96, norm_layer=nn.BatchNorm2d)
    D = define_D(input_nc=input1_nc + input2_nc + opt.output_nc, Ddownx2 = opt.tocg_Ddownx2, Ddropout = opt.tocg_Ddropout, n_layers_D=3, spectral = opt.spectral, num_D = opt.tocg_num_D)

    # Load Checkpoint
    # if not opt.tocg_checkpoint == '' and os.path.exists(opt.tocg_checkpoint):
    #     load_checkpoint(tocg, opt.tocg_checkpoint, opt)
    load_checkpoint(tocg, opt.tocg_load_final_checkpoint)

    # Train
    train_model(opt, train_loader,test_loader, validation_loader, tocg_curve_writer, tocg_image_writer,tocg, D)

    # Save Checkpoint
    save_checkpoint(tocg, os.path.join(opt.checkpoint_dir, opt.name, 'tocg_final.pth'),opt)
    save_checkpoint(D, os.path.join(opt.checkpoint_dir, opt.name, 'D_final.pth'),opt)
    print("Finished training %s!" % opt.name)

