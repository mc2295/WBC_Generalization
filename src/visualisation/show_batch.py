import matplotlib.pyplot as plt
def show_batch(batch):

    b = batch[0] # [0] to have the images, [1] to have the labels
    labels = batch[1]
    n = len(labels)
    fig, axes = plt.subplots(n//4,4, figsize = ((10,10)))
    for i, ax in enumerate(axes.flat):
        ax.imshow(b[i].permute(2,1,0).to('cpu'))
        ax.set_title(labels[i])
        ax.axis('off') 
    fig.savefig('batch.png')
