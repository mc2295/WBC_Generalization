from data.create_variables import create_df, create_splits, list_labels_cat, create_dataloader_pairs
from data.siamese_image import Siamese2Label
from fastai.vision.all import Learner, noop
from models.siamese.siamese_params import siamese_splitter
import torch
import pandas as pd

def create_files_to_label(entry_path, unlabeled_source, unlabeled_size, list_labels_cat) :
    df = create_df(entry_path + 'references', unlabeled_source, list_labels_cat)
    splits = create_splits('../', unlabeled_source, df, unlabeled_size, balanced = False, files_to_look_at = None)
    to_label = df.iloc[splits[0]]['name'].tolist()
    real_label = df.iloc[splits[0]]['label'].tolist()
    return to_label, real_label

def label_files(entry_path, unlabeled_source, unlabeled_size, labeled_source, labeled_size, model, list_labels_cat, transform = True, batchsize = 8):
    to_label, real_label = create_files_to_label(entry_path, unlabeled_source, unlabeled_size, list_labels_cat)
    pred_label = []
    df_labeled = create_df(entry_path + 'references', labeled_source, list_labels_cat)
    splits_labeled = create_splits(entry_path, labeled_source, df_labeled, labeled_size, balanced = False, files_to_look_at = None)

    for i, k in enumerate(to_label):
        row2 = pd.Series({'index':'0', 'name' : k, 'label' : '', 'dataset' : unlabeled_source[0]})
        dls = create_dataloader_pairs(entry_path, batchsize, Siamese2Label, splits_labeled, df_labeled, transform = transform, row2label = row2)
        learn = Learner(dls, model, splitter = siamese_splitter,loss_func=noop)
        preds,targs = learn.get_preds(dl = dls.valid, with_targs = False)
        index = torch.argmax(preds).item()
        pred_label.append(learn.dls.valid_ds[index][2])

    return real_label, pred_label
