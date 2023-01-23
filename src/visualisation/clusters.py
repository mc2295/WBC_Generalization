import cv2
import numpy as np
import matplotlib.pyplot as plt
# import umap.umap_ as umap
import flameplot as flameplot


from PIL import Image


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

def scatter_plot_clusters(X_2D, y, method):

    fig, ax = flameplot.scatter(X_2D[:,0], X_2D[:,1], labels=y, title=method, density=False)
    # fig.savefig('reports/' + method + '/' + training_source + '_si '+ siamese_number + ' _on_'+ source1 + '_batch_1.png')
    plt.show()
