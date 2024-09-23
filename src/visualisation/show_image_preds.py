import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

short_palette = {"neu":"C1", "mon":"C3", "lym":"C0", "ery":"C5", "eos": "C2", "bas":"C4"}

def show_histo_of_pred_tensor(preds_tensor, fig_name, list_labels_cat):       
    fig, axes = plt.subplots(5,5, figsize = ((13,15)))

    for i, ax in enumerate(axes.flat):
        dic = {}
        for k in range(len(list_labels_cat)):
            dic[list_labels_cat[k][:3]] = [preds_tensor[i][k].item()]
        df = pd.DataFrame(dic)
        sns.barplot(df, ax = ax, palette = short_palette)
        ax.set
    fig.savefig(fig_name)
    
def show_images_with_preds(preds_label, preds_proba, labels, filenames, fig_name):
    fig, axes = plt.subplots(5,5, figsize = ((13,15)))
    folder = '../'
    for i, ax in enumerate(axes.flat):
        ax.imshow(plt.imread(folder + '/' + filenames[i]))
        ax.axis('off')
        ax.title.set_text('pred : {} \n actual : {} \n'.format(preds_label[i], labels[i]) + ' proba : {0:.2f}'.format(preds_proba[i]))
    fig.savefig(fig_name)