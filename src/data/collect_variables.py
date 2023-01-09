import pickle
import pandas as pd
import numpy as np
from fastai.vision.all import TfmdLists, Resize, ToTensor, IntToFloatTensor

def create_df(dic_classes, source_df, list_labels_cat):


    dataframe = pd.read_csv('references/variables/dataframes/df_labeled_images.csv')


    dataframe_source0 = dataframe[dataframe['image_dataset'] == source_df ]
    dataframe_source = dataframe_source0.copy()
    for k in dic_classes:

        dataframe_source.loc[dataframe_source0['image_class']== k, 'image_class'] = dic_classes[k]

    print(dataframe_source['image_class'])
    dataframe_source = dataframe_source.loc[dataframe_source['image_class'].isin(list_labels_cat)]

    return dataframe_source


def import_variables(source1, source2):
    file = open('references/variables/dic_classes.obj', 'rb')
    dic_classes = pickle.load(file)

    list_labels_cat = (['basophil', 'eosinophil', 'erythroblast', 'lymphocyte', 'neutrophil']
                    if source1 =='matek'
                    else ['basophil', 'eosinophil', 'erythroblast', 'lymphocyte', 'neutrophil', 'monocyte'])
    list_labels = [0,1,2,3,4] if source1 == 'matek' else [0,1,2,3,4,5]


    dataframe_source = create_df(dic_classes,source2, list_labels_cat)

    files = list(dataframe_source['image_path'])
    array_files = np.array(files)


    class_list = list(dataframe_source['image_class'])
    array_class = np.array(class_list)

    file = open('references/variables/' + source1 + '_lbl2files.obj', 'rb')
    lbl2files = pickle.load(file)
    return dic_classes, list_labels_cat, list_labels, dataframe_source, files, array_files, class_list, array_class, lbl2files

def create_dataloader(source1, array_files, array_class):
    file = open('references/variables/' + source1 + '_tfm.obj', 'rb')
    tfm = pickle.load(file)
    file2 = open('references/variables/' + source1 + '_splits.obj', 'rb')
    splits = pickle.load(file2)
    trains  = array_files[splits[0]]
    valids = array_files[splits[1]]
    valids_class = array_class[splits[1]]
    # assert not [v for v in valids if v in array_files[splits[0]]]
    tls = TfmdLists(array_files, tfm, splits=splits)
    # dls = tls.dataloaders(after_item=[Resize(224), ToTensor],
    #                       after_batch=[IntToFloatTensor, Normalize.from_stats(*imagenet_stats)])
    dls = tls.dataloaders(after_item=[Resize(224), ToTensor],
                        after_batch=[IntToFloatTensor], bs = 32)
    dls.cuda()
    return trains, valids, valids_class, tls, dls
