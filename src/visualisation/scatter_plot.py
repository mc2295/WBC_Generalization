import matplotlib
from pylab import rcParams
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE, Isomap
from umap import UMAP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import TruncatedSVD, PCA, FastICA

font_path = '../references/variables/times_new_roman.ttf'  
fm.fontManager.addfont(font_path)  
plt.rcParams['font.family'] = 'Times New Roman Cyr'  # Replace 'YourCustomFont' with the actual font name
palette ={"neutrophil": "C1", "monocyte": "C3", "lymphocyte": "C0", "erythroblast": "C5", "eosinophil" : "C2", "basophil": "C4"}

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

def scatter_plot(X, labels, method, dataset, filenames,  name_fig, display_classes = True,):
    data = create_df_of_2D_embeddings_info(X, labels, method, dataset, filenames)
    rcParams['figure.figsize'] = 10, 10
    matplotlib.rcParams.update({'font.size': 20})
    # matplotlib.rcParams.update({'legend.title_fontsize' : 20})
    marker_size = 10
    data['marker_size'] = marker_size
    fig, ax = plt.subplots(1,1)
    if display_classes:
        # ax = sns.scatterplot(data=data, x='x', y='y', style='dataset', hue = 'classes', palette = palette, s = marker_size)
        # ax = sns.scatterplot(data=data, x='x', y='y', hue = 'classes', palette = palette, s = marker_size)
        ax = sns.scatterplot(data=data, x='x', y='y', hue = 'classes', s = marker_size)
    else:
        # les datasets sont représentés par des couleurs différentes
        palette_dataset = {'barcelona': "C3", 'saint_antoine': "C7", "matek": "C4", "rabin": "C5"}
        # ax = sns.scatterplot(data=data, x='x', y='y', hue = 'dataset', palette = palette_dataset)
        ax = sns.scatterplot(data=data, x='x', y='y', hue = 'dataset',s = marker_size)
    plt.tight_layout()
    fig.savefig(name_fig)
    return data
def visualise_images_in_region(entry_path, data, cell_class, x_region, y_region):
    source_mask = data.classes.isin(cell_class)
    group_class = data.loc[source_mask]
    # group_class = data
    group = group_class.loc[(group_class['x']>(x_region[0]))&(group_class['x']<(x_region[1]))
                            &(group_class['y']>(y_region[0]))&(group_class['y']<(y_region[1]))]
    fig, axes = plt.subplots(5,5, figsize = ((10,10)))
    
    for i, ax in enumerate(axes.flat):
        if i < len(group):
            ax.imshow(plt.imread(entry_path +  group.names.iloc[i]))
            ax.set_title(group.classes.iloc[i])
            ax.axis('off')
    fig.savefig('report/' + cell_class[0] +  ' in region x: ' + str(x_region[0]) + ', ' + str(x_region[1]) + 'y :' + str(y_region[0]) + ', ' + str(y_region[1])) 
    return group

