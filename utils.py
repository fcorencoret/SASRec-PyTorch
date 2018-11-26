import os
import torch
import shutil
import numpy as np
import json

def accuracy2(output, target, topk=(1,), nDCG=10):
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

def multiple_binary_cross_entropy(seq_emb, pos_emb, pos, neg_emb):
    # Prediction layer
    pos_logits = torch.sum(pos_emb * seq_emb, dim=-1)
    neg_logits = torch.sum(neg_emb * seq_emb, dim=-1)

    # Calculate loss
    istarget = ((pos.view(pos.size()[0] * pos.size()[1])) != 0).to(dtype=torch.float32)
    loss = -torch.sum(torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget + 
                torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget) / torch.sum(istarget)
    return loss

def accuracy(test_logits):
    hitrate1, hitrate10, ndcg10 = 0, 0, 0
    rank = torch.sort(torch.sort(test_logits)[1])[1][0]
    if rank == 0:
        hitrate1 = 1
        hitrate10 = 1
        ndcg10 = 1 / np.log2(rank + 2)
    elif rank < 10:
        hitrate10 = 1
        ndcg10 = 1 / np.log2(rank + 2)
    return hitrate1, hitrate10, ndcg10 



def plot(store, output_dir, model_name):
    import matplotlib.pyplot as plt
    # Plot train and val data

    def plot_ax(ax, title, xdata1, xdata2=False):
        # Losses ax1
        ax.set_title(title)
        ax.set_xlabel('Epochs')
        ax.set_ylabel(title)
        ax.plot([],[], color='red', label='Train')
        ax.lines[0].set_xdata([i * 20 for i in range(len(xdata1))])
        ax.lines[0].set_ydata(xdata1)
        if xdata2:
            ax.plot([],[], color='green', label='Val')
            ax.lines[1].set_xdata([i * 20 for i in range(len(xdata2))])
            ax.lines[1].set_ydata(xdata2)
        ax.legend()
        ax.relim()
        ax.autoscale_view()
        return ax

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots( nrows=2, ncols=2, figsize=(10, 10) )
    
    # Losses ax1
    ax1 = plot_ax(ax1, 'Loss', store['train_loss'])

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