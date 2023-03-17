import torch
import time
import os, pdb

from core.utils import AverageMeter, calculate_accuracy

"""
Code adapted from https://github.com/LiliMeng/3D-ResNets-PyTorch.git
"""


def train_epoch(epoch, data_loader, model, criterion, optimizer, opt,
                epoch_logger, batch_logger):
    print('train at epoch {}'.format(epoch))

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    end_time = time.time()
    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        if isinstance(inputs, torch.Tensor):
            inputs = inputs.cuda()
        else:
            inputs = [i.cuda() for i in inputs]
        targets = targets.cuda()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        acc = calculate_accuracy(outputs, targets)

        losses.update(loss.item(), outputs.size(0))
        accuracies.update(acc, outputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        batch_logger.log({
            'epoch': epoch,
            'batch': i + 1,
            'iter': (epoch - 1) * len(data_loader) + (i + 1),
            'loss': losses.val,
            'acc': accuracies.val,
            'lr': optimizer.param_groups[0]['lr']
        })

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                  epoch,
                  i + 1,
                  len(data_loader),
                  batch_time=batch_time,
                  data_time=data_time,
                  loss=losses,
                  acc=accuracies))

    print("epoch: {}, loss: {}, average acc: {}".format(epoch, losses.avg, accuracies.avg))
    epoch_logger.log({
        'epoch': epoch,
        'loss': losses.avg,
        'acc': accuracies.avg,
        'lr': optimizer.param_groups[0]['lr']
    })

    save_file_path = os.path.join(opt.output_dir,
                                  'save_last.pth')
    states = {
        'epoch': epoch + 1,
        'arch': opt.model_type,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(states, save_file_path)

    if epoch+1 % opt.checkpoint == 0:
        save_file_path = os.path.join(opt.output_dir,
                                      'save_{}.pth'.format(epoch))
        states = {
            'epoch': epoch + 1,
            'arch': opt.model_type,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(states, save_file_path)
