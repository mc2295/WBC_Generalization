import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import seaborn as sns
from pylab import rcParams
import numpy as np

'''
- plot_correlation: plots confusion matrix of between target labels and prediction
- plot_KNN_space : plots 2D graph with colored zones showing what prediction of KNN in this zone would be
'''


def plot_correlation(targ_valids, preds_valids, list_labels_cat):
    '''
    plot correlation's matrix to explore dependency between features
    '''
    # init figure size
    targ_valids_cat = [list_labels_cat[i] for i in targ_valids]
    preds_valids_cat = [list_labels_cat[i] for i in preds_valids]
    rcParams['figure.figsize'] = 7, 7
    df = pd.DataFrame(confusion_matrix(targ_valids_cat, preds_valids_cat, labels = list_labels_cat, normalize = 'true'), index = list_labels_cat, columns= list_labels_cat)
    fig = plt.figure()
    sns.heatmap(df, annot=True, fmt=".2f")
    plt.show()
    # fig.savefig('corr.png')



def plot_KNN_space(targ_trains, targ_valids, X_2D):
    '''
    split data, fit, classify, plot and evaluate results
    '''
    # split data into training and testing set
    X_train, y_train = X_2D[:len(targ_trains),:], targ_trains
    X_test, y_test = X_2D[len(targ_trains):,:], targ_valids
    # warnings.filterwarnings("ignore")

    #
    # init vars
    n_neighbors = 5
    h           = .02  # step size in the mesh

    # Create color maps
    #     cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF', '#FFFAAA', '#FAAAAA', '#AAAAAF', '#FFFFAA'])
    #     cmap_bold  = ListedColormap(['#FF0000', '#0000FF', '#FFF000', '#F00000', '#00000F', '#FFFF00'])
    #     colors = ["#E04848", "#EE6C6C", "#F89090", "#FEB4B4", "#FFD8D8", "#FFFFFF", "#FFFFFF", "#D8D8FF", "#B4B4FF", "#9090FF", "#6C6CFF", "#4848FF"]
    #     colors_light = ["#E04848000", "#EE6C6C000", "#F89090000", "#FEB4B4000", "#FFD8D8000", "#FFFFFF000", "#FFFFFF000", "#D8D8FF000", "#B4B4FF000", "#9090FF000", "#6C6CFF000", "#4848FF000"]
    color = ['r', 'b', 'purple', 'yellow', 'pink', 'grey']
    cmap_bold = ListedColormap(color)
    #     cmap_light = ListedColormap(colors_light)

    rcParams['figure.figsize'] = 20, 20
    for weights in ['uniform', 'distance']:
        # we create an instance of Neighbours Classifier and fit the data.
        clf = KNeighborsClassifier(n_neighbors, weights=weights)
        clf.fit(X_train, y_train)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        x_min, x_max = X_2D[:, 0].min() - 1, X_2D[:, 0].max() + 1
        y_min, y_max = X_2D[:, 1].min() - 1, X_2D[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        fig = plt.figure()
#         plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
        plt.pcolormesh(xx, yy, Z, cmap=cmap_bold, alpha = 0.5)
        # Plot also the training points, x-axis = 'Glucose', y-axis = "BMI"
        plt.scatter(X_2D[:, 0], X_2D[:, 1], c=y, cmap=cmap_bold, s=2)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title("0/1 outcome classification (k = %i, weights = '%s')" % (n_neighbors, weights))
        plt.show()
        fig.savefig(weights +'.png')

# classify_and_plot(proj_umap,y)
