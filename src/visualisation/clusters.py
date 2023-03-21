import cv2
import numpy as np
import matplotlib.pyplot as plt
# import umap.umap_ as umap
from sklearn.manifold import TSNE
# from umap import UMAP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from PIL import Image
from pylab import rcParams
from sklearn import decomposition
import pandas as pd   # '0.25.3'
import seaborn as sns # '0.9.0'

def make_wall(im, source1, labels, order, side=7):

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

def make_walls(res, Nimages, batch, valids, valids_class, source, training_source):
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

        newWall = make_wall(allIm[indiceCluster[i]], source, valids_class[indiceCluster[i]+Nimages*(batch-1)], order)
        plt.imsave(arr= newWall, fname = 'reports/' + source + "_walls/" + training_source + "_training/batch_"+ str(batch)+ "/cluster" + str(i) + ".png")

        
def project_2D(X,labels,method):
    if method == 'PCA':
        X_proj = decomposition.TruncatedSVD(n_components=2).fit_transform(X)
    elif method == 'LDA' :
        lda = LDA(n_components = 2)
        X_proj = lda.fit_transform(X,labels)
    if method == 'UMAP':
        umap_2d = UMAP(n_components=2, init='random', random_state=0)
        X_proj = umap_2d.fit_transform(X)
    elif method == 't-SNE':
        tsne = TSNE(n_components=2, random_state=0, metric = 'manhattan')
        X_proj= tsne.fit_transform(X)
    return X_proj

def create_df_of_2D_embeddings_info(X, labels, method, dataset, filenames = None):
    n = len(labels)
    if n > 0:
        ## on regarde les embeddings des images par un modèle
        X_proj = project_2D(X, labels, method)
        list_labels_cat = ['basophil',  'eosinophil', 'erythroblast', 'lymphocyte', 'monocyte', 'neutrophil']
        list_labels = [list_labels_cat[int(labels[i]) +1] if (dataset[i]== 'rabin' and int(labels[i])>=2) else list_labels_cat[int(labels[i])] for i in range(n)]
        data = pd.DataFrame(
            dict(x=X_proj[:,0],
            y=X_proj[:,1],
            dataset=dataset,
            classes = list_labels))
    else: 
        ## on regarde les images flattened 
        X_proj = project_2D(X, dataset, method)
        data = pd.DataFrame(
            dict(x=X_proj[:,0],
            y=X_proj[:,1],
            dataset=dataset
            ))   
    return data

def scatter_plot_embeddings(X, labels, method, dataset, display_classes = True):
    data = create_df_of_2D_embeddings_info(X, labels, method, dataset)
    rcParams['figure.figsize'] = 10, 10
    if display_classes:
        # les datasets sont représentés par des marqueurs différents
        ax = sns.scatterplot(data=data, x='x', y='y', style='dataset', hue = 'classes')
    else:
        # les datasets sont représentés par des couleurs différentes
        ax = sns.scatterplot(data=data, x='x', y='y', style='dataset', hue = 'dataset')
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    return data


def visualise_images_in_region(entry_path, data, cell_class, x_region, y_region):
    source_mask = data.classes.isin(cell_class)
    group_class = data.loc[source_mask]
    # group_class = data
    group = group_class.loc[(group_class['x']>(x_region[0]))&(group_class['x']<(x_region[1]))
                            &(group_class['y']>(y_region[0]))&(group_class['y']<(y_region[1]))]
    fig, axes = plt.subplots(5,5, figsize = ((10,10)))
    for i, ax in enumerate(axes.flat):
        ax.imshow(plt.imread(entry_path + group.name.iloc[i]))
        ax.set_title(group.classes.iloc[i])
        ax.axis('off')
    return group