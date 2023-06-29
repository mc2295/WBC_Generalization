import cv2
import numpy as np
import matplotlib.pyplot as plt
# import umap.umap_ as umap
from sklearn.manifold import TSNE, Isomap
from umap import UMAP
from fastai.vision.all import itertools
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import TruncatedSVD, PCA, FastICA
from PIL import Image
from pylab import rcParams
import pandas as pd   # '0.25.3'
import seaborn as sns # '0.9.0'
from data.create_variables import list_labels_cat
from sklearn.metrics import pairwise_distances
import matplotlib.font_manager as fm

palette ={"neutrophil": "C1", "monocyte": "C3", "lymphocyte": "C0", "erythroblast": "C5", "eosinophil" : "C2", "basophil": "C4"}
short_palette = {"neu":"C1", "mon":"C3", "lym":"C0", "ery":"C5", "eos": "C2", "bas":"C4"}
'''
this module:
- make_wall : creates a wall with the labels on each image
- make_walls : creates and saves clusters of spectral clustering from images names and corresponding class
- project_2D : project embeddings into 2D following requested method
- create_df_of_2D_embeddings_info: returns dataframe with
    ['x', 'y', 'dataset_name', 'labels'] if we want to visualise embeddings
    ['x', 'y', 'dataset_name'] if we want to see distribution of images flattened vector in 2D depending on dataset
- scatter_plot_embeddings : ok
- visualise_images_in_region : shows images corresponding to a certain region in the former scatted plot
'''
font_path = '../references/variables/times_new_roman.ttf'
fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'Times New Roman Cyr'  # Replace 'YourCustomFont' with the actual font name



def make_wall(im, labels, order, side=7):

    Nimage = im.shape[0]
    yside = side
    xside = 1 + Nimage//side

    res =  np.zeros(250*250*xside*yside*3).reshape(250*xside, 250*yside, 3)

    for i in range(Nimage):
        img = np.array(im[[order[i]]][0,:,:,:])
        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(img,labels[[order[i]]][0],(10,250), font, 2,(0,0,0),1)
        norm = 255
        res[250*(i//side):250*(i//side)+250, 250*(i%side):250*(i%side)+250] = img/norm
    return(res)

def make_walls(res, Nimages, batch, valids, valids_class, training_source):
    Ninit = 1000
    Ncluster = 8
    sc = SpectralClustering(Ncluster, affinity='precomputed', n_init=Ninit, assign_labels='kmeans')
    a = sc.fit_predict(res)
    indiceCluster = [np.arange(Nimages)[a==i] for i in range(Ncluster)]
#     os.makedirs("matek_walls")
    # allIm = np.array([plt.imread('../../Documents/These/' + x)[:,:,:3] for x in valids[(batch-1)*Nimages : batch*Nimages]])
    img_list = [np.array(Image.open('../../Documents/These/'+ fname).convert('RGB').resize((250, 250))) for fname in valids[(batch-1)*Nimages : batch*Nimages]]
    allIm = np.array(img_list)
    for i in range(Ncluster):
        print("Walls: " + str(i+1) + "/" + str(Ncluster), end = '\r')
        distanceIntraCluster =  np.mean(res[np.ix_(indiceCluster[i], indiceCluster[i])], axis=0)
        order = np.flip(np.argsort(distanceIntraCluster))

        newWall = make_wall(allIm[indiceCluster[i]], valids_class[indiceCluster[i]+Nimages*(batch-1)], order)
        plt.imsave(arr= newWall, fname = 'reports/walls/' + training_source + "_training/batch_"+ str(batch)+ "/cluster" + str(i) + ".png")


def project_2D(X,labels,method):
    if method == 'PCA':
        X_proj = PCA(n_components=2).fit_transform(X)
    elif method == 'LDA' :
        lda = LDA(n_components = 2)
        X_proj = lda.fit_transform(X,labels)
    if method == 'UMAP':
        umap_2d = UMAP(n_components=2, init='random', random_state=0)
        X_proj = umap_2d.fit_transform(X)
    elif method == 't-SNE':
        tsne = TSNE(n_components=2, random_state=0, metric = 'manhattan')
        X_proj= tsne.fit_transform(X)
    elif method == 'SVD':
        svd = TruncatedSVD(n_components=2, random_state=42)
        X_proj = svd.fit_transform(X)
    elif method == 'FastICA':
        ICA = FastICA(n_components=3, random_state=12)
        X_proj=ICA.fit_transform(X)
    elif method == 'Isomap':
        isomap = Isomap(n_neighbors=5, n_components=2)
        X_proj = isomap.fit_transform(X)
    return X_proj

def create_df_of_2D_embeddings_info(X, labels, method, dataset, filenames):
    n = len(labels)
    if n > 0:
        ## on regarde les embeddings des images par un modèle
        X_proj = project_2D(X, labels, method)
        data = pd.DataFrame(
            dict(x=X_proj[:,0],
            y=X_proj[:,1],
            dataset=dataset,
            classes = labels,
            names = filenames))
    else:
        ## on regarde les images flattened
        X_proj = project_2D(X, dataset, method)
        data = pd.DataFrame(
            dict(x=X_proj[:,0],
            y=X_proj[:,1],
            dataset=dataset
            ))
    return data

def scatter_plot_embeddings(X, labels, method, dataset, filenames, display_classes = True):
    data = create_df_of_2D_embeddings_info(X, labels, method, dataset, filenames)
    rcParams['figure.figsize'] = 10, 10

    if display_classes:
        # les datasets sont représentés par des marqueurs différents
        ax = sns.scatterplot(data=data, x='x', y='y', style='dataset', hue = 'classes', palette = palette)
    else:
        # les datasets sont représentés par des couleurs différentes
        palette_dataset = {'barcelona': "C3", 'saint_antoine': "C7", "matek": "C4", "rabin": "C5"}
        ax = sns.scatterplot(data=data, x='x', y='y', hue = 'dataset', palette = palette_dataset)
    plt.tight_layout()
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1), fontsize = 20)
    return data


def visualise_images_in_region(entry_path, data, cell_class, x_region, y_region):
    source_mask = data.classes.isin(cell_class)
    group_class = data.loc[source_mask]
    # group_class = data
    group = group_class.loc[(group_class['x']>(x_region[0]))&(group_class['x']<(x_region[1]))
                            &(group_class['y']>(y_region[0]))&(group_class['y']<(y_region[1]))]
    fig, axes = plt.subplots(5,5, figsize = ((10,10)))
    for i, ax in enumerate(axes.flat):
        ax.imshow(plt.imread(entry_path +  group.names.iloc[i]))
        ax.set_title(group.classes.iloc[i])
        ax.axis('off')
    return group

def plot_distribution(vector, title = ''):
    plt.figure()
    fig = sns.countplot(y = np.array(vector), palette = palette, order = list_labels_cat)
    plt.yticks(fontsize=20)
    plt.margins(0.15)
    plt.tight_layout()
    fig.bar_label(fig.containers[0])
    # plt.savefig(title + '.png')
    plt.title(title, fontsize = 20)

def show_images_with_preds(preds_label, preds_proba, labels, filenames):
    fig, axes = plt.subplots(5,5, figsize = ((13,15)))
    folder = '../'
    for i, ax in enumerate(axes.flat):
        ax.imshow(plt.imread(folder + '/' + filenames[i]))
        ax.axis('off')
        ax.title.set_text('pred : {} \n actual : {} \n'.format(preds_label[i], labels[i]) + ' proba : {0:.2f}'.format(preds_proba[0][i]))

def show_histo_of_pred_tensor(preds_tensor):
    fig, axes = plt.subplots(5,5, figsize = ((13,15)))

    for i, ax in enumerate(axes.flat):
        dic = {}
        for k in range(len(list_labels_cat)):
            dic[list_labels_cat[k][:3]] = [preds_tensor[i][k].item()]
        df = pd.DataFrame(dic)
        sns.barplot(df, ax = ax, palette = short_palette)
        ax.set

def plot_confusion_matrix(cm, list_labels_cat, normalize):
        cmap = "Blues"
        title = 'Confusion matrix'
        norm_dec:int=2  # Decimal places for normalized occurrences
        plot_txt:bool=True # Display occurrence in matrix
        normalize = normalize
        fig = plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title, fontsize = 20)
        tick_marks = np.arange(len(list_labels_cat))
        plt.xticks(tick_marks, list_labels_cat, rotation=90, fontsize=20)
        plt.yticks(tick_marks, list_labels_cat, rotation=0, fontsize=20)

        if plot_txt:
            thresh = cm.max() / 2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                coeff = f'{cm[i, j]:.{norm_dec}f}' if normalize else f'{cm[i, j]}'
                plt.text(j, i, coeff, horizontalalignment="center", verticalalignment="center", color="white"
                         if cm[i, j] > thresh else "black", fontsize = 15)

        ax = fig.gca()
        ax.set_ylim(len(list_labels_cat)-.5,-.5)

        plt.tight_layout()
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.grid(False)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.show()
