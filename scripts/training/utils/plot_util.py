import matplotlib.pyplot as plt
import itertools
 
def plot_loss(history):
    """plot training history"""
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    # plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPG]')
    plt.legend()
    plt.grid(True)

def plot_gt_pre(gt,pre):
    """predictioin vs ground truth in two subplots"""
    plt.figure(figsize = (12,10))
    plt.subplot(211)
    plt.legend()
    ax1=plt.subplot(2, 1, 1)
    ax2=plt.subplot(212, )
    ax1.plot(gt,label = 'ground truth')
    ax1.title.set_text('Ground Truth')
    ax2.plot(pre,label = 'predict')
    ax2.title.set_text('Prediction')

def plot_gt_pre_overlap(gt,pre):
    """predictioin vs ground truth in one plot"""
    fig = plt.figure(figsize =(12,10))
    ax = fig.add_subplot(111)
    ax.plot(gt,label = 'ground truth')
    ax.plot(pre, '.', label = 'predict', alpha = 0.5)
    plt.legend()

def plot_gt_pre_sep(gt,idx_train,pre_train,idx_test,pre_test):
    """(predictioin for X_test and X_train separately) vs (ground truth)  in one plot"""
    fig = plt.figure(figsize =(12,10))
    ax = fig.add_subplot(111)
    
    ax.plot(gt,label = 'ground truth')
    ax.plot(idx_train,pre_train,'.', label = 'prediction of training set', alpha = 0.5)
    ax.plot(idx_test,pre_test,'.', label = 'prediction of test set', alpha = 0.5)
    
    plt.legend()

def plot_gt_pre_overlap_mul(gt,pre,outputs):
    """predictioin vs ground truth in one plot
    for multiple outputs (with names listed in the argument "outputs") in one plot
    with only first 50000 points"""
    l = len(outputs)
    gt = gt.T[:,:50000]
    pre = pre.T[:,:50000]
    fig = plt.figure(figsize =(12,8*l))
    for i,output in enumerate(outputs): 
        ax = plt.subplot(l,1,i+1)
        ax.set_title(output, fontsize=16)
        ax.plot(gt[i],label = 'ground truth')
        ax.plot(pre[i], '.', label = 'predict', alpha = 0.5)
        ax.legend(bbox_to_anchor=(1.25, 1),loc='upper right')


def plot_gt_pre_sep_mul(gt,idx_train,pre_train,idx_test,pre_test,outputs):
    """(predictioin for X_test and X_train separately) vs (ground truth) 
    for multiple outputs (with names listed in the argument "outputs") in one plot"""
    l = len(outputs)
    gt = gt.T
    pre_train = pre_train.T
    pre_test = pre_test.T
    l = len(outputs)
    fig = plt.figure(figsize =(12,8*l))
    for i,output in enumerate(outputs): 
        ax = plt.subplot(l,1,i+1)
        ax.set_title(output, fontsize=16)
        ax.plot(gt[i],label = 'ground truth')
        ax.plot(idx_train,pre_train[i],'.', label = 'prediction of training set', alpha = 0.5)
        ax.plot(idx_test,pre_test[i],'.', label = 'prediction of test set', alpha = 0.5)
        ax.legend(bbox_to_anchor=(1.25, 1),loc='upper right')