import torch
import time
import os
import sys

import torch
import torch.distributed as dist
import loss
import opts
from utils import AverageMeter, calculate_accuracy

def semi_train_epoch(epoch,
                     train_loader,
                     model,
                     criterion,
                     optimizer,
                     device,
                     current_lr,
                     epoch_logger,
                     batch_logger,
                     tb_writer=None,
                     distributed=False,
                     num_classes,
                     p_cutoff):
    print('train at epoch {}'.format(epoch))

    model.train()
    
    contrastiveLoss = ContrastiveLoss(batch_size=args.BatchSize)
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    end_time = time.time()
    
    with tqdm(zip(iter(train_loader), iter(unlabel_loader)), total =len(train_loader)) as t:
        for (input_rgb, input_htg, labels), (input_unrgb, input_unhtg, unlabels) in t:
            
            data_time.update(time.time() - end_time)
            input_rgb, input_htg, labels = input_rgb.to(device), input_htg.to(device), labels.to(device)
            
            outputrgb, rgbfeature, outputhtg, htgfeature = model(input_rgb, input_htg)
            
            sup_loss = supervised_loss(sum([outputrgb(i) for i in outputrgb]) / len(outputrgb), labels) + supervised_loss(sum([outputhtg(i) for i in outputhtg]) / len(outputhtg), labels)
            
            input_unrgb, input_unhtg, unlabels = input_rgb.to(device), input_htg.to(device), labels.to(device)
            
            outputunrgb, unrgbfeature, outputunhtg, unhtgfeature = model(input_unrgb, input_unhtg)
            
            unsup_loss1 = unsupervised_loss(sum([outputrgb(i) for i in outputrgb]) / len(outputrgb), outputunhtg, p_cutoff)
            unsup_loss2 = unsupervised_loss(sum([outputhtg(i) for i in outputhtg]) / len(outputhtg), outputunrgb, p_cutoff)
            
            acc = calculate_accuracy(outputs, labels)
            
            loss = sup_loss + lambda_cot * (unsup_loss1 + unsup_loss2) + contrastiveLoss(rgbfeature, htgfeature) + contrastiveLoss(unrgbfeature, unhtgfeature)
            
            losses.update(loss.item(), inputs.size(0))
            accuracies.update(acc, inputs.size(0))
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            batch_time.update(time.time() - end_time)
            end_time = time.time()
    
            if batch_logger is not None:
                batch_logger.log({
                    'epoch': epoch,
                    'batch': i + 1,
                    'iter': (epoch - 1) * len(data_loader) + (i + 1),
                    'loss': losses.val,
                    'acc': accuracies.val,
                    'lr': current_lr
                })
    
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(epoch,
                                                             i + 1,
                                                             len(data_loader),
                                                             batch_time=batch_time,
                                                             data_time=data_time,
                                                             loss=losses,
                                                             acc=accuracies))

    if distributed:
        loss_sum = torch.tensor([losses.sum],
                                dtype=torch.float32,
                                device=device)
        loss_count = torch.tensor([losses.count],
                                  dtype=torch.float32,
                                  device=device)
        acc_sum = torch.tensor([accuracies.sum],
                               dtype=torch.float32,
                               device=device)
        acc_count = torch.tensor([accuracies.count],
                                 dtype=torch.float32,
                                 device=device)

        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(loss_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(acc_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(acc_count, op=dist.ReduceOp.SUM)

        losses.avg = loss_sum.item() / loss_count.item()
        accuracies.avg = acc_sum.item() / acc_count.item()

    if epoch_logger is not None:
        epoch_logger.log({
            'epoch': epoch,
            'loss': losses.avg,
            'acc': accuracies.avg,
            'lr': current_lr
        })

    if tb_writer is not None:
        tb_writer.add_scalar('train/loss', losses.avg, epoch)
        tb_writer.add_scalar('train/acc', accuracies.avg, epoch)
        tb_writer.add_scalar('train/lr', accuracies.avg, epoch)
        
    t.close() # 用完进度条记得关闭
