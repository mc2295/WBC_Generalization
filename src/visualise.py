from data.main import train_loader, valid_loader
from tqdm import tqdm
from torch.autograd import Variable
import config
from visualisation.confusion_matrix import show_confusion_matrix
from visualisation.scatter_plot import scatter_plot, visualise_images_in_region
from visualisation.flatten_images import create_matrix_of_flattened_images
from visualisation.show_image_preds import show_images_with_preds, show_histo_of_pred_tensor
from visualisation.metrics import save_preds, get_bootstrap_confidence_interval
from visualisation.show_auc import show_auc_per_class, show_auc_all
from sklearn.metrics import accuracy_score
import torch
import numpy as np

model = torch.load(config.model_path).to(config.device)


filenames = []
datasets = []
labels = []
embeddings = [] # the output of the encoder
preds_tensor = [] #  the output of the whole model
preds_proba = [] # only the max of preds_tensor
preds_label = [] # the categorical pred label
preds_num = [] # the numerical pred label

for img, label_cat, img_path, dataset in tqdm(valid_loader):
        test_loss = []
        with torch.no_grad():
            img = Variable(img.to(config.device))
            labels += [k for k in label_cat]
            filenames += [k for k in img_path]
            datasets += [k for k in dataset]
            out = model(img)
            out = torch.nn.Softmax()(out)
            preds_num_batch = out.argmax(dim=1) 
            preds_proba += [out[i][preds_num_batch[i]].item() for i in range(len(preds_num_batch))]
            preds_label +=[config.list_labels_cat[i] for i in preds_num_batch]
            
            embedding_batch = model.encoder(img)
            embeddings+= [k.cpu().detach().numpy() for k in embedding_batch]
            preds_tensor += [k.cpu().detach().numpy() for k in out]
            preds_num += [k.cpu().detach().numpy() for k in preds_num_batch]

embeddings = np.stack(embeddings)
preds_tensor = np.stack(preds_tensor)
preds_num = np.stack(preds_num)
preds_proba = np.stack(preds_proba)

print('accuracy :', accuracy_score(labels, preds_label))
      
if config.show_confusion: 
    recall_per_class, precision_per_class, acc = show_confusion_matrix(preds_label, labels, config.list_labels_cat, config.plot_confusion)

if config.show_flattened_images: 
    X, dataset_flattened = create_matrix_of_flattened_images(valid_loader)
    data = scatter_plot(X, [], config.method, dataset_flattened, None, 'report/flattened_img.png',  display_classes = False,)

if config.scatter_embeddings:
    df_embeddings = scatter_plot(embeddings, labels, config.method, datasets, filenames, 'report/scatter plot.png', True)
    
if config.show_group_from_region:
    group = visualise_images_in_region('../', df_embeddings, ['erythroblast'], [0,10], [-15, -0])

if config.show_proba_per_image :
    show_images_with_preds(preds_label, preds_proba, labels, filenames, 'report/preds images.png')
    show_histo_of_pred_tensor(preds_tensor, 'report/preds histo.png', config.list_labels_cat)

if config.bootstrap:
    save_preds(preds_label, labels, config.lr, config.batchsize, config.source)
    r = get_bootstrap_confidence_interval(preds_label, labels, config.source[0], config.list_labels_cat, draw_number=1000)

if config.show_auc_per_class:
    _ = show_auc_per_class(labels, preds_tensor, config.list_labels_cat, config.source[0])
    
if config.show_auc_all:
    show_auc_all(labels, preds_tensor, config.list_labels_cat, config.source, datasets)


    
