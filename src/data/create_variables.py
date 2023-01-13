import pickle
import pandas as pd
import numpy as np
from fastai.vision.all import TfmdLists, Resize, ToTensor, IntToFloatTensor
from sklearn.model_selection import train_test_split
import os as os

def create_df(reference_path, dic_classes, source_df, list_labels_cat):
    dataframe = pd.read_csv(reference_path + '/variables/dataframes/df_labeled_images.csv')
    dataframe_source0 = dataframe[dataframe['image_dataset'] == source_df ]
    dataframe_source = dataframe_source0.copy()
    for k in dic_classes:
        dataframe_source.loc[dataframe_source0['image_class']== k, 'image_class'] = dic_classes[k]
    print(dataframe_source['image_class'])
    dataframe_source = dataframe_source.loc[dataframe_source['image_class'].isin(list_labels_cat)]
    return dataframe_source

def create_variables(reference_path, source1, source2):
    file = open(reference_path + '/variables/dic_classes.obj', 'rb')
    dic_classes = pickle.load(file)

    list_labels_cat = (['basophil', 'eosinophil', 'erythroblast', 'lymphocyte', 'neutrophil']
                    if source1 =='matek'
                    else ['basophil', 'eosinophil', 'erythroblast', 'lymphocyte', 'neutrophil', 'monocyte'])
    list_labels = [0,1,2,3,4] if source1 == 'matek' else [0,1,2,3,4,5]

    dataframe_source = create_df(reference_path, dic_classes,source2, list_labels_cat)

    files = list(dataframe_source['image_path'])
    array_files = np.array(files)

    class_list = list(dataframe_source['image_class'])
    array_class = np.array(class_list)

    return dic_classes, list_labels_cat, list_labels, dataframe_source, files, array_files, class_list, array_class

def create_valid_train_test_splits(df, array_files, array_class):
    mapping_dic = {}
    for i, file in enumerate(array_files):
        mapping_dic[file] = i

    X_train_valid, X_test, y_train_valid, y_test = train_test_split(array_files, array_class, test_size=0.1, random_state=42)
    test_index = [mapping_dic[i] for i in X_test]
    file = open('references/variables/matek_test_index.obj', 'wb')
    pickle.dump(test_index, file)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_valid, y_train_valid, test_size = 0.2, random_state = 42)
    train_index = [mapping_dic[i] for i in X_train]
    print('fini train')
    valid_index = [mapping_dic[i] for i in X_valid]
    print('fini valid')
    splits = train_index, valid_index
    file = open('references/variables/matek_splits.obj', 'wb')
    pickle.dump(splits, file)

def class_proportion(y):
    return (np.unique(y, return_counts=True)[1]/len(y))

def create_dataloader(reference_path, source1, array_files, array_class, SiameseTransform, df, list_labels_cat, list_labels):
    if not os.path.exists(reference_path + '/variables/' + source1 + '_splits.obj'):
        create_valid_train_test_splits(df, array_files, array_class)

    file2 = open(reference_path + '/variables/' + source1 + '_splits.obj', 'rb')
    splits = pickle.load(file2)

    if not os.path.exists(reference_path + '/variables/' + source1 + '_tfm.obj'):
        tfm = SiameseTransform(array_files, splits, df, list_labels_cat, list_labels)
        file = open('../references/variables/matek_tfm.obj', 'wb')
        pickle.dump(tfm, file)

    file = open(reference_path + '/variables/' + source1 + '_tfm.obj', 'rb')
    tfm = pickle.load(file)

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
