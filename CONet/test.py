from typing import Tuple

import torch

#from . import global_vars as GLOBALS
import global_vars as GLOBALS


def test_main(model, test_loader, epoch: int, device, optimizer) -> Tuple[float, float]:
    # global best_acc, performance_statistics, net, criterion, checkpoint_path
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            if GLOBALS.CONFIG['network'] == 'DARTS' or GLOBALS.CONFIG['network'] == 'DARTSPlus':
                outputs, _ = model(inputs)
            else:
                outputs = model(inputs)
            loss = GLOBALS.CRITERION(outputs, targets)

            test_loss += loss.item()
            #_, predicted = outputs.max(1)
            #total += targets.size(0)
            #correct += predicted.eq(targets).sum().item()

            acc1_temp, acc5_temp = accuracy(outputs, targets, topk=(1, 5))
            top1.update(acc1_temp[0], inputs.size(0))
            top5.update(acc5_temp[0], inputs.size(0))

            # progress_bar(
            #     batch_idx, len(test_loader),
            #     'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #     % (test_loss / (batch_idx + 1), 100. * correct / total,
            #        correct, total))

    # Save checkpoint.
    acc = top1.avg.cpu().item()
    acc5 = top5.avg.cpu().item()
    if acc > GLOBALS.BEST_ACC:
        # print('Adas: Saving checkpoint...')
        state = {
            'net': model.state_dict(),
            'acc': acc,
            'epoch': epoch + 1,
            'optimizer':optimizer.state_dict(),
            'train_loss' : GLOBALS.TRAIN_LOSS
        }
        if GLOBALS.ADAS is not None:
            state['historical_io_metrics'] = GLOBALS.METRICS.historical_metrics
        torch.save(state, str(GLOBALS.CHECKPOINT_PATH / 'ckpt.pth'))
        # if checkpoint_path.is_dir():
        #     torch.save(state, str(checkpoint_path / 'ckpt.pth'))
        # else:
        #     torch.save(state, str(checkpoint_path))
        GLOBALS.BEST_ACC = acc
    acc = acc / 100
    acc5 = acc5 / 100
    GLOBALS.PERFORMANCE_STATISTICS[f'test_acc_epoch_{epoch}'] = acc
    GLOBALS.PERFORMANCE_STATISTICS[f'test_acc5_epoch_{epoch}'] = acc5
    GLOBALS.PERFORMANCE_STATISTICS[f'test_loss_epoch_{epoch}'] = \
        test_loss / (batch_idx + 1)
    return test_loss / (batch_idx + 1), acc, acc5

def accuracy(outputs, targets, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = targets.size(0)

        _, pred = outputs.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count