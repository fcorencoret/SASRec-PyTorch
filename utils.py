import os
import torch
import shutil
import numpy as np
import json

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
                log_correct_item += torch.log(torch.sigmoid(batch[item_index][item]) + 1e-24)
            else:
                mask_non_S[item, :] = 0
        
        last_item = target[batch_index][-1]
        log_correct_item += torch.log(torch.sigmoid(batch[item_index][last_item]) + 1e-24)
        mask_non_S[:, last_item] = 0
        loss -= (log_correct_item + torch.sum(mask_non_S * sums))
    return loss

def plot(store, output_dir, model_name):
    import matplotlib.pyplot as plt
    # Plot train and val data

    def plot_ax(ax, title, xdata1, xdata2):
        # Losses ax1
        ax.set_title(title)
        ax.set_xlabel('Epochs')
        ax.set_ylabel(title)
        ax.plot([],[], color='red', label='Train')
        ax.plot([],[], color='green', label='Val')
        ax.lines[0].set_xdata([i for i in range(len(xdata1))])
        ax.lines[0].set_ydata(xdata1)
        ax.lines[1].set_xdata([i for i in range(len(xdata2))])
        ax.lines[1].set_ydata(xdata2)
        ax.legend()
        ax.relim()
        ax.autoscale_view()
        return ax

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots( nrows=2, ncols=2, figsize=(10, 10) )
    
    # Losses ax1
    ax1 = plot_ax(ax1, 'Loss', store['train_loss'], store['val_loss'])

    # HitRate@1 ax2
    ax2 = plot_ax(ax2, 'HitRate@1', store['train_hitrate@1'], store['val_hitrate@1'])

    # HitRate@10 ax3
    ax3 = plot_ax(ax3, 'HitRate@10', store['train_hitrate@10'], store['val_hitrate@10'])    

    # nDCG@10 ax4
    ax4 = plot_ax(ax4, 'nDCG@10', store['train_ndcg@10'], store['val_ndcg@10'])

    # save plot    
    fig.savefig(os.path.join(output_dir, model_name) + '_training.png')
    plt.close(fig)

    print(' > Training data plotted')

    with open(os.path.join(output_dir, model_name) + '_data.json', 'w') as fp:
        json.dump(store, fp)

    print(' > Training data saved')