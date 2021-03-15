import matplotlib.pyplot as plt

# plot training history
def plot_loss(history):
	plt.plot(history.history['loss'], label='loss')
	plt.plot(history.history['val_loss'], label='val_loss')
	# plt.ylim([0, 10])
	plt.xlabel('Epoch')
	plt.ylabel('Error [MPG]')
	plt.legend()
	plt.grid(True)

# predictioin vs ground truth in two subplots
def plot_gt_pre(gt,pre):
	plt.figure(figsize = (12,10))
	plt.subplot(211)
	plt.legend()
	ax1=plt.subplot(2, 1, 1)
	ax2=plt.subplot(212, )
	ax1.plot(gt,label = 'ground truth')
	ax1.title.set_text('Ground Truth')
	ax2.plot(pre,label = 'predict')
	ax2.title.set_text('Prediction')

# predictioin vs ground truth in one plot
def plot_gt_pre_overlap(gt,pre):
    fig = plt.figure(figsize =(12,10))
    ax = fig.add_subplot(111)
    ax.plot(gt,label = 'ground truth')
    ax.plot(pre, label = 'predict', alpha = 0.5)
    plt.legend()

# (predictioin for X_test and X_train separately) vs (ground truth)  in one plot
def plot_gt_pre_sep(gt,idx_train,pre_train,idx_test,pre_test):
    fig = plt.figure(figsize =(16,12))
    ax = fig.add_subplot(111)
    
    ax.plot(gt,label = 'ground truth')
    ax.plot(idx_train,pre_train,'.', label = 'prediction of training set', alpha = 0.5)
    ax.plot(idx_test,pre_test,'.', label = 'prediction of test set', alpha = 0.5)
    
    plt.legend()