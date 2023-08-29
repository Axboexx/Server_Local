"""
@Time : 2023/7/12 17:54
@Author : Axboexx
@File : main_lib.py
@Software: PyCharm
"""
import os
import time
import numpy as np
import torch


class AverageMeter(object):
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


def train(train_loader,
          model,
          criterion,
          optimizer,
          scheduler,
          epoch,
          accum=1):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            target = target.cuda()
            input = input.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        # compute output
        output = model(input_var)
        loss = criterion(output, target_var) / accum

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target_var, topk=(1, 5))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step

        loss.backward()

        if i % accum == 0:
            optimizer.step()
            optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 100 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch,
                i,
                len(train_loader),
                batch_time=batch_time,
                data_time=data_time,
                loss=losses,
                top1=top1,
                top5=top5))
    # 最后一波 清空一下梯度 感觉没啥必要但写了试试（
    # if i % accum != 0:
    #     optimizer.step()
    #     optimizer.zero_grad()
    # update scheduler
    # scheduler.step()
    # lr = scheduler.get_last_lr()[0]
    # print("LR stepped to %.4f" % (lr))


def validate(val_loader, model, criterion, num_class):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # 添加对F1-score和Recall的支持
    # 我该怎么知道总类别有多少呢？
    # 用关键字传过来好了
    num = num_class
    # 构建统计矩阵
    con_mat = np.zeros((num, num), dtype=np.int)

    # switch to evaluate mode
    model.eval()
    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if torch.cuda.is_available():
                target = target.cuda()
                input = input.cuda()
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)
            loss = torch.sum(loss)
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target_var, topk=(1, 5))

            # 更新统计指标，用于之后其他指标（如召回率，F1 Score等）的计算
            _, pred = output.topk(1, 1, True, True)
            pred = pred.t()[0].cpu()
            target_var = target_var.cpu()
            for pred, pos in zip(pred, target_var):
                con_mat[pred, pos] += 1

            losses.update(loss.data.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 100 == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i,
                    len(val_loader),
                    batch_time=batch_time,
                    loss=losses,
                    top1=top1,
                    top5=top5))

    # 计算 Precision, Recall, F1-score等
    num_tot = np.sum(con_mat)
    num_tp = [con_mat[x, x] for x in range(num)]
    sum_tp = np.trace(con_mat)
    num_pred = np.sum(con_mat, axis=0)
    num_gts = np.sum(con_mat, axis=1)
    # print(num_gts)
    # 分类别得到统计结果
    cls_lv_prec = num_tp / num_pred
    cls_lv_recall = num_tp / num_gts
    # cls_lv_recall[np.isnan(cls_lv_recall)] = 0
    cls_lv_F1 = (2 * cls_lv_prec * cls_lv_recall) / (cls_lv_prec +
                                                     cls_lv_recall)

    micro_nametag = "micro"
    micro_prec = np.sum(cls_lv_prec * num_pred) / num_tot
    micro_recall = np.sum(num_tp) / num_tot
    # micro_F1 is never calculated because it generally equals to accuracy.
    weighted_F1 = np.sum(cls_lv_F1 * num_gts) / num_tot

    macro_nametag = "macro"
    macro_prec = np.mean(cls_lv_prec)
    macro_recall = np.mean(cls_lv_recall)
    macro_F1 = np.mean(cls_lv_F1)

    top1_acc = sum_tp / num_tot

    # 激光打印.jpg
    print("%10s|%10s|%10s|%10s" % ("", "Prec.", "Recall", "F1-score"))
    print(
        f"{macro_nametag:10s}|{macro_prec * 100:10.3f}|{macro_recall * 100:10.3f}|{macro_F1 * 100:10.3f}"
    )
    print(
        f"{micro_nametag:10s}|{micro_prec * 100:10.3f}|{micro_recall * 100:10.3f}|{weighted_F1 * 100:10.3f}"
    )
    print("(The above score is weighted F1-score.)")
    print(
        f"(For checking only) Top-1 Accu = micro F1-score = micro recall = micro precision = {top1_acc * 100:.3f}\n"
    )

    # 打印Top-1/Top-5结果
    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1,
                                                                  top5=top5))

    return top1.avg, top5.avg


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape((-1,)).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def if_mkdir(path):
    if os.path.exists(path):
        return True
    else:
        os.mkdir(path)
