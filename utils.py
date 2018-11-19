import os
import torch
import shutil
import numpy as np

def accuracy(output, target, topk=(1,), nDCG=10):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    n = output.size(1)

    _, pred = output.topk(maxk, 2, True, True)
    target = target.unsqueeze(-1).expand_as(pred)
    correct = pred.eq(target)
    correct = correct[:, -1]
    res = []
    # HitRate@K
    for k in topk:
        correct_k = correct[: , : k].sum((1, 0)).to(dtype=torch.float)
        res.append(correct_k.sum(0) / batch_size)

    #nDCG@10
    nDCG = 0
    for i in range(batch_size):
        if 1 in correct[i]:
            nDCG += ( 1 / np.log2( correct[i].argmax().item() + 2 ))
    res.append(nDCG / batch_size)
    return res

def save_checkpoint(state, is_best, output_dir, model_name, filename='_checkpoint.pth.tar'):
    checkpoint_path = os.path.join(output_dir, model_name) + filename
    model_path = os.path.join(output_dir, model_name) + '_model_best.pth.tar'
    torch.save(state, checkpoint_path)
    if is_best:
        print(" > Best model found at this epoch. Saving ...")
        shutil.copyfile(checkpoint_path, model_path)

def multiple_binary_cross_entropy(input, target, output, loss):
    for batch_index, batch in enumerate(output):
        sums = torch.log(torch.ones_like(batch) - torch.sigmoid(batch) + 1e-24)
        mask_non_S = torch.ones_like(batch)

        log_correct_item = 0
        for item_index, item in enumerate(input[batch_index]):
            if item != 0:
                mask_non_S[:, item] = 0
                log_correct_item += torch.log(torch.sigmoid(batch[item_index][item]))
            else:
                mask_non_S[item, :] = 0
        
        last_item = target[batch_index][-1]
        log_correct_item += torch.log(torch.sigmoid(batch[item_index][last_item]))
        mask_non_S[:, last_item] = 0
        loss -= (log_correct_item + torch.sum(mask_non_S * sums))
    return loss
