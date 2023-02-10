import pickle
import pandas as pd
import numpy as np
from fastai.vision.all import TfmdLists, Resize, ToTensor, IntToFloatTensor, ImageDataLoaders
from sklearn.model_selection import train_test_split
import os as os

def create_df(reference_path, source_df, list_labels_cat):
    file = open(reference_path + '/variables/dic_classes.obj', 'rb')
    dic_classes = pickle.load(file)
    dataframe = pd.read_csv(reference_path + '/variables/dataframes/df_labeled_images.csv')
    source_mask = dataframe.image_dataset.isin(source_df)
    dataframe = dataframe.loc[source_mask,['image_path', 'image_class']]
    dataframe.image_class = [dic_classes[x] for x in dataframe.image_class]
    label_mask = dataframe.image_class.isin(list_labels_cat)
    dataframe = dataframe.loc[label_mask,['image_path', 'image_class']]

    files = list(dataframe['image_path'])
    class_list = list(dataframe['image_class'])

    return files, class_list

def create_variables(reference_path, source):

    list_labels_cat = (['basophil', 'eosinophil', 'erythroblast', 'lymphocyte', 'neutrophil', 'monocyte'])

    files = []
    len_array_files = 0
    class_list = []
    splits = [], []

    for k in range(len(source)):

        files_one_source, class_one_source = create_df(reference_path, [source[k]], list_labels_cat)
        files += files_one_source
        class_list += class_one_source

        file_splits = open(reference_path + '/variables/'+ source[k]+'_splits.obj', 'rb')
        splits_one_source = pickle.load(file_splits)
        splits = [splits[j] + [[k + len_array_files for k in splits_one_source[i]] for i in range(2)][j] for j in range(2)]
        # splits_translated = [[k + len_array_files for k in splits_one_source[i]] for i in range(2)]
        # splits = splits[0] + splits_translated[0], splits[1] + splits_translated[1]
        len_array_files += len(files_one_source)

    array_class = np.array(class_list)
    array_files = np.array(files)

    dic_labels = {}
    for k in range(len(files)):
        dic_labels[files[k]] = class_list[k]

    return array_files, array_class, splits, dic_labels, list_labels_cat

def create_valid_train_test_splits(array_files, array_class, source):
    mapping_dic = {}
    for i, file in enumerate(array_files):
        mapping_dic[file] = i

    X_train_valid, X_test, y_train_valid, y_test = train_test_split(array_files, array_class, test_size=0.1, random_state=42)
    test_index = [mapping_dic[i] for i in X_test]
    file = open('references/variables/' + source + '_test_index.obj', 'wb')
    pickle.dump(test_index, file)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_valid, y_train_valid, test_size = 0.2, random_state = 42)
    train_index = [mapping_dic[i] for i in X_train]
    print('fini train')
    valid_index = [mapping_dic[i] for i in X_valid]
    print('fini valid')
    splits = train_index, valid_index
    file = open('references/variables/' + source + '_splits.obj', 'wb')
    pickle.dump(splits, file)

def class_proportion(y):
    return (np.unique(y, return_counts=True)[1]/len(y))

def create_siamese_dataloader(array_files, array_class, splits, dic_labels, list_labels_cat, SiameseTransform, batchsize):
    tfm = SiameseTransform(array_files, splits, dic_labels, list_labels_cat)
    tls = TfmdLists(array_files, tfm, splits=splits)
    dls = tls.dataloaders(after_item=[Resize(224), ToTensor],
                    after_batch=[IntToFloatTensor], bs = batchsize)
    dls.cuda()
    trains  = array_files[splits[0]]
    valids = array_files[splits[1]]
    valids_class = array_class[splits[1]]
    return trains, valids, valids_class, tls, dls

def create_dataloader(array_files, array_class, splits):
    # dls_unlabeled = ImageDataLoaders.from_df(df, item_tfms=Resize(224), path = '../data/Single_cells/Pred2_Image_60.vsi - 40x_BF_EFI_01/', seed = 42)
    # dls_unlabeled.cuda()
    df = pd.DataFrame(list(zip(array_files, array_class, [i in splits[1] for i in range(len(array_files))])), columns = ['name', 'label', 'is_valid'])
    dls = ImageDataLoaders.from_df(df, item_tfms=Resize(224), path = '../', valid_col='is_valid')  
    return dls 