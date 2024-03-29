import os
import time
import random
import json
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR, MultiStepLR

import numpy as np
from configs.opts import parser
from model.main_model_a2v import supv_main_model as main_model
from utils import AverageMeter, Prepare_logger, get_and_save_args
from utils.Recorder import Recorder
from dataset.AVE_dataset import AVEDataset

 # =================================  seed config ============================
SEED = 700
random.seed(SEED)
np.random.seed(seed=SEED)
torch.manual_seed(seed=SEED)
torch.cuda.manual_seed(seed=SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# =============================================================================


def main():
    # utils variable
    global args, logger, writer, dataset_configs
    # statistics variable
    global best_accuracy, best_accuracy_epoch
    best_accuracy, best_accuracy_epoch = 0, 0
    # configs
    dataset_configs = get_and_save_args(parser)
    parser.set_defaults(**dataset_configs)
    args = parser.parse_args()
    # select GPUs
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    '''Create snapshot_pred dir for copying code and saving models '''
    if not os.path.exists(args.snapshot_pref):
        os.makedirs(args.snapshot_pref)

    if os.path.isfile(args.resume):
        args.snapshot_pref = os.path.dirname(args.resume)
    
    logger = Prepare_logger(args, eval=args.evaluate)

    if not args.evaluate:
        logger.info(f'\nCreating folder: {args.snapshot_pref}')
        logger.info('\nRuntime args\n\n{}\n'.format(json.dumps(vars(args), indent=4)))
    else:
        logger.info(f'\nLog file will be save in a {args.snapshot_pref}/Eval.log.')
    
        

    '''Dataset'''
    train_dataloader = DataLoader(
        AVEDataset('/data/sliu/work/data/', split='train'),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )

    test_dataloader = DataLoader(
        AVEDataset('/data/sliu/work/data/',split='test'),
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    '''model setting'''
    mainModel = main_model()
    mainModel = nn.DataParallel(mainModel).cuda()
    learned_parameters = mainModel.parameters()
    optimizer = torch.optim.Adam(learned_parameters, lr=args.lr)
    # scheduler = StepLR(optimizer, step_size=40, gamma=0.2)
    scheduler = MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.5)
    criterion = nn.BCEWithLogitsLoss().cuda()
    criterion_event = nn.CrossEntropyLoss().cuda()
    criterion_self = nn.CrossEntropyLoss().cuda()
    '''Resume from a checkpoint'''
    if os.path.isfile(args.resume):
        logger.info(f"\nLoading Checkpoint: {args.resume}\n")
        mainModel.load_state_dict(torch.load(args.resume))
    elif args.resume != "" and (not os.path.isfile(args.resume)):
        raise FileNotFoundError
    

    '''Only Evaluate'''
    if args.evaluate:
        logger.info(f"\nStart Evaluation..")
        validate_epoch(mainModel, test_dataloader, criterion, criterion_event,criterion_self, epoch=0, eval_only=True)
        return

    '''Tensorboard and Code backup'''
    writer = SummaryWriter(args.snapshot_pref)
    recorder = Recorder(args.snapshot_pref, ignore_folder="Exps/")
    recorder.writeopt(args)

    '''Training and Testing'''
    for epoch in range(args.n_epoch):
        loss = train_epoch(mainModel, train_dataloader, criterion, criterion_event, criterion_self,optimizer, epoch)

        if ((epoch + 1) % args.eval_freq == 0) or (epoch == args.n_epoch - 1):
            acc = validate_epoch(mainModel, test_dataloader, criterion, criterion_event, criterion_self,epoch)
            #best_accuracy = acc
            #best_accuracy_epoch = epoch
            #save_checkpoint(mainModel.state_dict(),top1=best_accuracy,task='Supervised',epoch=epoch + 1)            

            if acc > best_accuracy:
                best_accuracy = acc
                best_accuracy_epoch = epoch
                save_checkpoint(
                    mainModel.state_dict(),
                    top1=best_accuracy,
                    task='Supervised',
                    epoch=epoch + 1,
                )
  
        scheduler.step()

def train_epoch(model, train_dataloader, criterion, criterion_event, criterion_self,optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    train_acc = AverageMeter()
    end_time = time.time()

    model.train()
    # Note: here we set the model to a double type precision, 
    # since the extracted features are in a double type. 
    # This will also lead to the size of the model double increases.
    model.double()
    optimizer.zero_grad()

    for n_iter, batch_data in enumerate(train_dataloader):

        data_time.update(time.time() - end_time)
        '''Feed input to model'''
        visual_feature, audio_feature, labels= batch_data
        # For a model in a float precision
        # visual_feature = visual_feature.float()
        # audio_feature = audio_feature.float()
        labels = labels.double().cuda()
        is_event_scores, event_scores,event_self = model(visual_feature, audio_feature,labels)
        is_event_scores = is_event_scores.transpose(1, 0).squeeze().contiguous()

        #labels_foreground = labels[:, :, :-1]  # [32, 10, 28]
        #labels_BCE, labels_evn = labels_foreground.max(-1)
        #labels_event, _ = labels_evn.max(-1)
        
        loss_is_event = criterion(is_event_scores, labels.cuda())
        #loss_event_class = criterion_event(event_scores, labels_event.cuda())
        #loss_event_self =criterion_self(event_self,labels_event.cuda())
        #loss = loss_is_event + 0.9*loss_event_class+0.1*loss_event_self
        loss = loss_is_event 
        loss.backward()

        '''Compute Accuracy'''
        acc = compute_accuracy_supervised(is_event_scores,  labels)
        train_acc.update(acc, visual_feature.size(0) * 10)

        '''Clip Gradient'''
        if args.clip_gradient is not None:
            total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)
            if total_norm > args.clip_gradient:
                logger.info(f'Clipping gradient: {total_norm} with coef {args.clip_gradient/total_norm}.')

        '''Update parameters'''
        optimizer.step()
        optimizer.zero_grad()

        losses.update(loss.item(), visual_feature.size(0)*10)
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        '''Add loss of a iteration in Tensorboard'''
        writer.add_scalar('Train_data/loss', losses.val, epoch * len(train_dataloader) + n_iter + 1)

        '''Print logs in Terminal'''
        if n_iter % args.print_freq == 0:
            logger.info(
                f'Train Epoch: [{epoch}][{n_iter}/{len(train_dataloader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                f'Prec@1 {train_acc.val:.3f} ({train_acc.avg: .3f})'
            )

        '''Add loss of an epoch in Tensorboard'''
        writer.add_scalar('Train_epoch_data/epoch_loss', losses.avg, epoch)

    return losses.avg



@torch.no_grad()
def validate_epoch(model, test_dataloader, criterion, criterion_event,criterion_self, epoch, eval_only=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()
    end_time = time.time()

    model.eval()
    model.double()

    for n_iter, batch_data in enumerate(test_dataloader):
        data_time.update(time.time() - end_time)

        '''Feed input to model'''
        visual_feature, audio_feature, labels  = batch_data
        # For a model in a float type
        # visual_feature = visual_feature.float()
        # audio_feature = audio_feature.float()
        labels = labels.double().cuda()
        bs = visual_feature.size(0)
        is_event_scores, event_scores,event_class = model(visual_feature, audio_feature,labels)
        is_event_scores = is_event_scores.transpose(1, 0).squeeze()

        #labels_foreground = labels[:, :, :-1]
        #labels_BCE, labels_evn = labels_foreground.max(-1)
        #labels_event, _ = labels_evn.max(-1)
        loss_is_event = criterion(is_event_scores, labels.cuda())
        #loss_event_class = criterion_event(event_scores, labels_event.cuda())
        #loss_event_self =criterion_self(event_class,labels_event.cuda())
        #loss = loss_is_event + 0.9*loss_event_class+0.1*loss_event_self        
        loss = loss_is_event 

        acc = compute_accuracy_supervised(is_event_scores,  labels)
        accuracy.update(acc, bs * 10)

        batch_time.update(time.time() - end_time)
        end_time = time.time()
        losses.update(loss.item(), bs * 10)

        '''Print logs in Terminal'''
        if n_iter % args.print_freq == 0:
            logger.info(
                f'Test Epoch [{epoch}][{n_iter}/{len(test_dataloader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                f'Prec@1 {accuracy.val:.3f} ({accuracy.avg:.3f})'
            )

    '''Add loss in an epoch to Tensorboard'''
    if not eval_only:
        writer.add_scalar('Val_epoch_data/epoch_loss', losses.avg, epoch)
        writer.add_scalar('Val_epoch/Accuracy', accuracy.avg, epoch)

    logger.info(
        f"\tEvaluation results (acc): {accuracy.avg:.4f}%."
    )
    return accuracy.avg


def compute_accuracy_supervised(is_event_scores, labels):
    # labels = labels[:, :, :-1]  # 28 denote background
    # pos pred
   
    audio_count=0
    closs_labels_np=labels.cpu().data.numpy() #[4143,10]
    labels_sum=np.sum(closs_labels_np,axis=1) #[4143,]
    labels_ind=labels_sum!=10
    count_num=np.sum(labels_ind==True) #1393

    #处理标签最佳位置
    seg = []   
    n=len(closs_labels_np)
    pred_vid = np.zeros([n,10])

    for i in range(n):
        la=closs_labels_np[i][:]
        nb=np.argwhere(la==1)
        segment=np.zeros(len(nb)).astype('int8')      
        for j in range(len(nb)):
            segment[j]=nb[j][0] 
        seg.append(segment)
     
    #预测结果获得最佳位置
    ids = []
    for j in range(n):
        l=len(seg[j])          
        score = []
        for nn in range(10-l+1):
            s = 0
            for i in range(l):
                s+=is_event_scores[j][i+nn]
            score.append(s)
            
        score = np.array(score).astype('float32')
        id = int(np.argmax(score))       
        ids.append(id)
        for k in range(ids[j],ids[j]+int(l)):
            pred_vid[j][k]=1
        

        if (seg[j][0]==id):
            audio_count+=1
                
    acc = audio_count*100/n

    return acc
def save_checkpoint(state_dict, top1, task, epoch):
    model_name = f'{args.snapshot_pref}/model_epoch_{epoch}_top1_{top1:.3f}_task_{task}_best_model.pth.tar'
    torch.save(state_dict, model_name)


if __name__ == '__main__':
    main()
