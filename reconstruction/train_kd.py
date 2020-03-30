import sys
import logging
import pathlib
import random
import shutil
import time
import functools
import numpy as np
import argparse

import torch
import torchvision
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from torch.utils.data import DataLoader
from dataset import SliceData,KneeData
#from models import UnetModel,conv_block
#from models import DnCn1,DnCn2
#from models import DnCn2
from models import DCTeacherNet,DCStudentNet #,Vgg16
import torchvision
from torch import nn
from torch.autograd import Variable
from torch import optim
from tqdm import tqdm
from functools import reduce

#from utils import CriterionPairWiseforWholeFeatAfterPool, CriterionPairWiseforWholeFeatAfterPoolFeatureMaps  

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_datasets(args):

    train_data = SliceData(args.train_path,args.acceleration_factor,args.dataset_type)
    dev_data = SliceData(args.validation_path,args.acceleration_factor,args.dataset_type)

    return dev_data, train_data

def create_datasets_knee(args):

    train_data = KneeData(args.train_path,args.acceleration_factor,args.dataset_type)
    dev_data = KneeData(args.validation_path,args.acceleration_factor,args.dataset_type)

    return dev_data, train_data


def create_data_loaders(args):

    if args.dataset_type == 'knee':
        dev_data, train_data = create_datasets_knee(args)
    else:
        dev_data, train_data = create_datasets(args)



    display_data = [dev_data[i] for i in range(0, len(dev_data), len(dev_data) // 16)]

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        #num_workers=64,
        #pin_memory=True,
    )

    train_loader_error = DataLoader(
        dataset=train_data,
    )

    dev_loader = DataLoader(
        dataset=dev_data,
        batch_size=args.batch_size,
        #num_workers=64,
        #pin_memory=True,
    )
    display_loader = DataLoader(
        dataset=display_data,
        batch_size=16,
        #num_workers=64,
        #pin_memory=True,
    )
    return train_loader, dev_loader, display_loader,train_loader_error


def train_epoch(args, epoch,modelT,modelS,data_loader, optimizer, writer):#,error_range):# , vgg):
    
    modelT.eval() 
    modelS.train()

    avg_loss = 0.
    start_epoch = start_iter = time.perf_counter()
    global_step = epoch * len(data_loader)
    alpha = 0.5
    ssim_factor = 0.1 

    for iter, data in enumerate(tqdm(data_loader)):

        input,input_kspace,target = data # Return kspace also we can ignore that for train and test 
        input = input.unsqueeze(1).to(args.device)
        input_kspace = input_kspace.unsqueeze(1).to(args.device)
        target = target.unsqueeze(1).to(args.device)

        input = input.float()
        target = target.float()

        outputT = modelT(input,input_kspace)
        outputS = modelS(input,input_kspace)

        #outputT_VGG = vgg(outputT[-1])
        #outputS_VGG = vgg(outputS[-1])

        #print (torch.max(outputT_VGG),torch.min(outputT_VGG))
        #print (torch.max(outputS_VGG),torch.min(outputS_VGG))

        lossSG = F.l1_loss(outputS[-1],target) # ground truth loss 
        lossST = F.l1_loss(outputS[-1],outputT[-1]) / outputT[-1].numel() # student - teacher loss 
        #lossTG = F.l1_loss(outputT[-1],target)  # ground truth - teacher 
        #lossSSIM = 1 - pytorch_ssim.ssim(outputT[-1],outputS[-1]) # ssim loss between teacher and student to better recover structures 
        #lossVGG = F.mse_loss(outputT_VGG, outputS_VGG) / outputS_VGG.numel()

        #error_scaleT = 1 #- (lossTG / error_range)

        #print (error_scaleT)
        #print(lossSG, lossST, lossTG, lossVGG)
        '''

        outputT_feat = [outputT[0][2],outputT[1][2],outputT[2][2],outputT[3][2],outputT[4][2]]
        outputS_feat = [outputS[0][1],outputS[1][1],outputS[2][1],outputS[3][1],outputS[4][1]]

        outputT_feat = [torch.sum(x**2,dim=1) for x in outputT_feat]
        outputS_feat = [torch.sum(x**2,dim=1) for x in outputS_feat]
        outputT_feat = [torch.nn.functional.normalize(x,dim=0) for x in outputT_feat]
        outputS_feat = [torch.nn.functional.normalize(x,dim=0) for x in outputS_feat]

        outputT_feat = [x.unsqueeze(1) for x in outputT_feat]
        outputS_feat = [x.unsqueeze(1) for x in outputS_feat]

        '''

        #outputT_feat  = outputT_feat_list[2]
        #outputS_feat  = outputS_feat_list[2]
        #outputT_feat  = torch.sum(outputT_feat ** 2, dim = 1 )
        #outputS_feat  = torch.sum(outputS_feat ** 2, dim = 1 )
        #outputT_feat  = torch.nn.functional.normalize(outputT_feat,dim=0)
        #outputS_feat  = torch.nn.functional.normalize(outputS_feat,dim=0)

        #loss = F.l1_loss(outputT_feat,outputS_feat) / outputS_feat.numel()

        #lossSSIM = 1 - pytorch_ssim.ssim(outputT[-1],outputS[-1]) # ssim loss between teacher and student to better recover structures 
        #feat_l1_loss = [F.l1_loss(x,y)/y.numel() for x,y in zip(outputT_feat,outputS_feat)]
        #feat_ssim_loss = [1 - pytorch_ssim.ssim(x,y) for x,y in zip(outputT_feat,outputS_feat)]

        #print (feat_l1_loss,feat_ssim_loss)

        #print (feat_loss)
        #loss_l1   = reduce((lambda x,y : x + y),feat_l1_loss)  / len(feat_l1_loss) 
        #loss_ssim = reduce((lambda x,y : x + y),feat_ssim_loss)  / len(feat_ssim_loss) 

        #print (loss_l1,loss_ssim)

        #lossT = feat_loss_sum

        #print (lossG,lossT)
 
        #loss = alpha * lossSG + ( 1 - alpha) * error_scaleT * lossST 
        #loss = lossSG + lossST + lossVGG
        #print (lossSG,lossST,lossSSIM)
        
        if args.imitation_required:
            loss = lossSG + lossST 
        else:
            loss = lossSG
        #loss = torch.min(lossG,lossT)
        #loss = loss_l1 + ssim_factor * loss_ssim 

        optimizer.zero_grad()
 
        loss.backward()

        optimizer.step()

        avg_loss = 0.99 * avg_loss + 0.01 * loss.item() if iter > 0 else loss.item()
        writer.add_scalar('TrainLoss',loss.item(),global_step + iter )

        if iter % args.report_interval == 0:
            logging.info(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item():.4g} Avg Loss = {avg_loss:.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
        start_iter = time.perf_counter()
        #break

    return avg_loss, time.perf_counter() - start_epoch


def evaluate(args, epoch, modelT,modelS,data_loader, writer):

    modelT.eval()
    modelS.eval()

    #losses_mse   = []
    losses = []
    #losses_ssim  = []
 
    start = time.perf_counter()
    
    with torch.no_grad():
        for iter, data in enumerate(tqdm(data_loader)):
    
            input,input_kspace, target = data # Return kspace also we can ignore that for train and test
            input = input.unsqueeze(1).to(args.device)
            input_kspace = input_kspace.unsqueeze(1).to(args.device)
            target = target.unsqueeze(1).to(args.device)
    
            input = input.float()
            target = target.float()
    
            #output = model(input,input_kspace)
            #loss = F.mse_loss(output[-1],target)
            #losses.append(loss.item())

            #outputT = modelT(input,input_kspace)
            outputS = modelS(input,input_kspace)

            #outputT_feat = [outputT[0][2],outputT[1][2],outputT[2][2],outputT[3][2],outputT[4][2]]
            #outputS_feat = [outputS[0][1],outputS[1][1],outputS[2][1],outputS[3][1],outputS[4][1]]
            #outputT_feat  = outputT_feat_list[2]
            #outputS_feat  = outputS_feat_list[2]
            #outputT_feat  = torch.sum(outputT_feat ** 2, dim = 1 )
            #outputS_feat  = torch.sum(outputS_feat ** 2, dim = 1 )
            #outputT_feat  = torch.nn.functional.normalize(outputT_feat,dim=0)
            #outputS_feat  = torch.nn.functional.normalize(outputS_feat,dim=0)
            '''
            outputT_feat = [torch.sum(x**2,dim=1) for x in outputT_feat]
            outputS_feat = [torch.sum(x**2,dim=1) for x in outputS_feat]
            outputT_feat = [torch.nn.functional.normalize(x,dim=0) for x in outputT_feat]
            outputS_feat = [torch.nn.functional.normalize(x,dim=0) for x in outputS_feat]

            outputT_feat = [x.unsqueeze(1) for x in outputT_feat]
            outputS_feat = [x.unsqueeze(1) for x in outputS_feat]

            feat_mse_loss = [F.mse_loss(x,y)/y.numel() for x,y in zip(outputT_feat,outputS_feat)]
            feat_ssim_loss = [1 - pytorch_ssim.ssim(x,y) for x,y in zip(outputT_feat,outputS_feat)]
            '''

            #feat_loss = [F.mse_loss(x,y) for x,y in zip(outputT_feat,outputS_feat)]
            #print (feat_loss)
            #loss_mse = reduce((lambda x,y : x + y),feat_mse_loss)  / len(feat_mse_loss) 

            #loss_ssim = reduce((lambda x,y : x + y),feat_ssim_loss)  / len(feat_ssim_loss) 

            #loss = F.mse_loss(outputT_feat,outputS_feat) #/ outputS_feat.numel()

            loss = F.mse_loss(outputS[-1],target)

            losses.append(loss.item())
          
            #losses_mse.append(loss_mse)
            #losses_ssim.append(loss_ssim)
            
        writer.add_scalar('Dev_Loss',np.mean(losses),epoch)
        #writer.add_scalar('Dev_Loss_ssim',np.mean(losses_ssim),epoch)
       
    return np.mean(losses), time.perf_counter() - start


def visualize(args, epoch, model,data_loader, writer):
    
    def save_image(image, tag):
        image -= image.min()
        image /= image.max()
        grid = torchvision.utils.make_grid(image, nrow=4, pad_value=1)
        writer.add_image(tag, grid, epoch)

    model.eval()

    with torch.no_grad():
        for iter, data in enumerate(tqdm(data_loader)):

            input,input_kspace,target = data # Return kspace also we can ignore that for train and test
            #input,_,target = data # Return kspace also we can ignore that for train and test
            input = input.unsqueeze(1).to(args.device)
            input_kspace = input_kspace.unsqueeze(1).to(args.device)
            target = target.unsqueeze(1).to(args.device)

            output = model(input.float(),input_kspace)[-1]

            #print("input: ", torch.min(input), torch.max(input))
            #print("target: ", torch.min(target), torch.max(target))
            #print("predicted: ", torch.min(output), torch.max(output))

            save_image(input, 'Input')
            save_image(target, 'Target')
            save_image(output, 'Reconstruction')
            save_image(torch.abs(target.float() - output.float()), 'Reconstruction error')

            #break

def save_model(args, exp_dir, epoch, model, optimizer,best_dev_loss,is_new_best):

    out = torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_dev_loss': best_dev_loss,
            'exp_dir':exp_dir
        },
        f=exp_dir / 'model.pt'
    )

    if is_new_best:
        shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')


def build_model(args):
 
    modelT = DCTeacherNet(args).to(args.device)
    modelS = DCStudentNet(args).to(args.device)

    return modelT,modelS

def load_model(model,checkpoint_file):
  
    #print (checkpoint_file)

    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['model'])

    #print(checkpoint['model'])
    return model


def get_error_range_teacher(loader,model):

    losses = []
    print ("Finding max and min error between teacher and target")

    for data in tqdm(loader):

        input,input_kspace, target = data # Return kspace also we can ignore that for train and test
        
        input  = input.unsqueeze(1).float().to(args.device)
        input_kspace = input_kspace.unsqueeze(1).to(args.device)
        target = target.unsqueeze(1).float().to(args.device)

        output = model(input,input_kspace)[-1]
        
        loss = F.l1_loss(output,target).detach().cpu().numpy()

        losses.append(loss)

    min_error,max_error = np.min(losses),np.max(losses)
    #pdb.set_trace()

    return max_error #- min_error 


def build_optim(args, params):
    optimizer = torch.optim.Adam(params, args.lr, weight_decay=args.weight_decay)
    return optimizer


def main(args):

    args.exp_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(args.exp_dir / 'summary'))

    modelT,modelS = build_model(args)

    modelT = load_model(modelT,args.teacher_checkpoint)
    
    if not args.student_pretrained:
        modelS = load_model(modelS,args.student_checkpoint)
    #print (args.student_pretrained,args.imitation_required)
    #sys.exit(0)

    #vgg = Vgg16().to(args.device)

    optimizer = build_optim(args, modelS.parameters())

    best_dev_loss = 1e9
    start_epoch = 0

    train_loader, dev_loader, display_loader, train_loader_error = create_data_loaders(args)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step_size, args.lr_gamma)

    
    # error_range = get_error_range_teacher(train_loader_error,modelT)
    # error_range = torch.Tensor([error_range]).to(args.device)
    #error_range = 1
    #print (error_range)
    
    for epoch in range(start_epoch, args.num_epochs):

        scheduler.step(epoch)

        train_loss,train_time = train_epoch(args, epoch, modelT,modelS,train_loader,optimizer,writer)
        #train_loss,train_time = train_epoch(args, epoch, modelT,modelS,train_loader,optimizer,writer,error_range,vgg)

        dev_loss,dev_time = evaluate(args, epoch, modelT, modelS, dev_loader, writer)

        #visualize(args, epoch, modelS,display_loader, writer)

        is_new_best = dev_loss < best_dev_loss
        best_dev_loss = min(best_dev_loss,dev_loss)
        save_model(args, args.exp_dir, epoch, modelS, optimizer,best_dev_loss,is_new_best)
        logging.info(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g}'
            f'DevLoss= {dev_loss:.4g} TrainTime = {train_time:.4f}s DevTime = {dev_time:.4f}s',
        )
    writer.close()


def create_arg_parser():

    parser = argparse.ArgumentParser(description='Train setup for MR recon U-Net')
    parser.add_argument('--seed',default=42,type=int,help='Seed for random number generators')
    parser.add_argument('--batch-size', default=4, type=int,  help='Mini batch size')
    parser.add_argument('--num-epochs', type=int, default=150, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--lr-step-size', type=int, default=40,
                        help='Period of learning rate decay')
    parser.add_argument('--lr-gamma', type=float, default=0.1,
                        help='Multiplicative factor of learning rate decay')
    parser.add_argument('--weight-decay', type=float, default=0.,
                        help='Strength of weight decay regularization')

    parser.add_argument('--report-interval', type=int, default=100, help='Period of loss reporting')

    parser.add_argument('--device', type=str, default='cuda',
                        help='Which device to train on. Set to "cuda" to use the GPU')
    parser.add_argument('--exp-dir', type=pathlib.Path, default='checkpoints',
                        help='Path where model and results should be saved')
    parser.add_argument('--train-path',type=str,help='Path to train h5 files')
    parser.add_argument('--validation-path',type=str,help='Path to test h5 files')

    parser.add_argument('--acceleration_factor',type=str,help='acceleration factors')
    parser.add_argument('--dataset_type',type=str,help='cardiac,kirby')
    parser.add_argument('--usmask_path',type=str,help='us mask path')
    parser.add_argument('--teacher_checkpoint',type=str,help='teacher checkpoint')
    parser.add_argument('--student_checkpoint',type=str,help='student checkpoint')
    parser.add_argument('--student_pretrained',action='store_true',help='for selecting whether to use student_pretrained_not')
    parser.add_argument('--imitation_required',action='store_true',help='option to select imitation loss')
    
    return parser


if __name__ == '__main__':
    args = create_arg_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)
