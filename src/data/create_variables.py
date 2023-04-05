import pickle
import pandas as pd
import numpy as np
from data.siamese_image import SiameseTransform, show_batch
from fastai.vision.all import *
from sklearn.model_selection import train_test_split
from data.transform_functions import transform_color_transfer, transform_resolution, transform_crop
from torchvision import transforms

'''
This module creates the following variables:
- Image names corresponding to source(s) dataset(s), and their associated classes following dic_classes nomenclature (array_files, array_class)
- same info but in a dic (dic_label['image_name'] = classes)
- splits indexes (train, valid, test)
- list of classes : list_labels_cat
- dataloaders from images (siamese_dataloader for siamese batches, dataloader for normal image batch)
'''

def create_df(reference_path, source_df, list_labels_cat):
    '''
    input: 
    - reference_path : the path to go to 'reference' folder, 
    - source_df : the name of the dataset(s), 
    - list_label_cat:list of labels
    return: 
    - files: list of images names from the dataset
    - class_list : associated label following dic_classes nomenclature (if it's in the list of labels) 
    '''

    file = open(reference_path + '/variables/dic_classes.obj', 'rb')
    dic_classes = pickle.load(file)
    dataframe = pd.read_csv(reference_path + '/variables/dataframes/df_labeled_images.csv')
    source_mask = dataframe.image_dataset.isin(source_df)
    dataframe = dataframe.loc[source_mask,['image_path', 'image_class', 'transformed_image_path']]
    dataframe.image_class = [dic_classes[x] for x in dataframe.image_class]
    label_mask = dataframe.image_class.isin(list_labels_cat)
    dataframe = dataframe.loc[label_mask,['image_path', 'image_class', 'transformed_image_path']]
#     if transform: 
#         files = list(dataframe['transformed_image_path'])
    files = list(dataframe['image_path'])
    class_list = list(dataframe['image_class'])

    return files, class_list

def create_variables(entry_path, source, transform = False):

    '''
    input : 
    - entry_path : '../' pour un .ipynb, '' pour un .py
    - source : source dataset
    - transform : only if I want to point to folder that are in data/transformed
    returns :
    - array_files : array(files names of images)
    - array_class : array(associated classes)
    - splits : indice of train, indice of valid among array_files (contains files for one or multiple sources)
    - dic_label : ['image_name'] = classes
    - list_label_cat : list of labels
    '''

    list_labels_cat = ['basophil',  'eosinophil' ,'erythroblast', 'lymphocyte', 'neutrophil', 'monocyte']
    if source == ['rabin']:
        # no erythroblast in rabin
        list_labels_cat = ['basophil', 'eosinophil' ,'lymphocyte', 'neutrophil','monocyte']
    files = []
    len_array_files = 0
    class_list = []
    splits = [], []

    for k in range(len(source)):

        files_one_source, class_one_source = create_df(entry_path + 'references', [source[k]], list_labels_cat)
        files += files_one_source
        class_list += class_one_source

        file_splits = open(entry_path + 'references' + '/variables/'+ source[k]+'_splits.obj', 'rb')
        splits_one_source = pickle.load(file_splits)
#         if source[k] == 'matek' : 
#             splits_one_source = splits_one_source[0][:30000], splits_one_source[1][:5000]
        splits = [splits[j] + [[k + len_array_files for k in splits_one_source[i]] for i in range(2)][j] for j in range(2)]
        # splits from every dataset with an offset for already counted data_source
        len_array_files += len(files_one_source)

    array_class = np.array(class_list)
    array_files = np.array(files)

    dic_labels = {}
    for k in range(len(files)):
        dic_labels[files[k]] = class_list[k]
    return array_files, array_class, splits, dic_labels, list_labels_cat



def create_dataloader_siamese(entry_path, source, batchsize, SiameseTransform, transform = False):
    '''
    input : 
    - entry_path : '../' pour un .ipynb, '' pour un .py
    - source : source dataset
    - transform if I want transforms
    returns :
    - dls : siamese dataloader
    '''
    array_files, array_class, splits, dic_labels, list_labels_cat = create_variables(entry_path, source)
    
    tfm = SiameseTransform(array_files, splits, dic_labels, list_labels_cat, entry_path)
    tls = TfmdLists(array_files, tfm, splits=splits)
    if transform:
        if source[0] in ['rabin', 'matek']:
            # no need to degrade quality of image or zoom for these already low resolution/zoomed image
            tfm = transforms.Compose([Transform(transform_color_transfer)])
        else: 
            tfm = transforms.Compose([Transform(transform_resolution), Transform(transform_crop), Transform(transform_color_transfer)])

        dls = tls.dataloaders(after_item=tfm,
                after_batch=[IntToFloatTensor], bs = batchsize)
    else:
        dls = tls.dataloaders(after_item=[Resize(224), ToTensor],
                        after_batch=[IntToFloatTensor], bs = batchsize)
    dls.cuda()
    return dls

def create_dataloader(entry_path, source, batchsize, size = 0, transform = False):
    '''
    input : 
    - entry_path : '../' pour un .ipynb, '' pour un .py
    - source : source dataset
    - transform if I want transforms
    returns :
    - dls : siamese dataloader
    '''

    array_files, array_class, splits, dic_labels, list_labels_cat = create_variables(entry_path , source)
    small_splits = splits[0][:size], splits[1]
    if size != 0 :
        small_array_files, small_array_class = array_files[small_splits[0]], array_class[small_splits[0]]
        while len(np.unique(small_array_class))< len(list_labels_cat):
            random.shuffle(splits[0])
            splits = splits[0], splits[1]           
            small_splits = splits[0][:size], splits[1]
            small_array_files, small_array_class = array_files[small_splits[0]], array_class[small_splits[0]]
        array_files, array_class = array_files[small_splits[0]+splits[1]], array_class[small_splits[0]+splits[1]]
        splits = small_splits
    else:
        array_files, array_class = array_files[splits[0]+splits[1]], array_class[splits[0]+splits[1]]
    df = pd.DataFrame(list(zip(array_files, array_class, [i>len(splits[0]) for i in range(len(splits[0]+splits[1]))])), columns = ['name', 'label', 'is_valid'])
    if transform:
        if source[0] in ['rabin', 'matek']:
            # no need to degrade quality of image or zoom for these already low resolution/zoomed image
            tfm = transforms.Compose([Transform(transform_color_transfer)])
        else: 
            tfm = transforms.Compose([Transform(transform_resolution), Transform(transform_crop), Transform(transform_color_transfer)])
        dls = ImageDataLoaders.from_df(df, item_tfms= tfm, path = entry_path, valid_col='is_valid', bs = batchsize)       
    else: 
        dls = ImageDataLoaders.from_df(df, item_tfms=Resize(224), path = entry_path, valid_col='is_valid', bs = batchsize)    
        dls.cuda()
    return dls

    
def create_dls(entry_path, source, siamese_head, batchsize = 32, size=0, transform = False):
        '''
        return a dataloader (siamese or not) from the desired dataset
        '''
        if siamese_head: 
            return create_dataloader_siamese(entry_path, source, batchsize, SiameseTransform, transform = transform)
        else: 
            return create_dataloader(entry_path, source, batchsize, size=size, transform = transform)
        
def create_valid_train_test_splits(array_files, array_class, source):
    '''
    create a test split and save it for final evaluation
    '''
    mapping_dic = {}
    for i, file in enumerate(array_files):
        mapping_dic[file] = i

    X_train_valid, X_test, y_train_valid, y_test = train_test_split(array_files, array_class, test_size=0.1, random_state=42)
    test_index = [mapping_dic[i] for i in X_test]
    file = open('references/variables/' + source[0] + '_test_index.obj', 'wb')
    pickle.dump(test_index, file)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_valid, y_train_valid, test_size = 0.2, random_state = 42)
    train_index = [mapping_dic[i] for i in X_train]
    print('fini train')
    valid_index = [mapping_dic[i] for i in X_valid]
    print('fini valid')
    splits = train_index, valid_index
    file = open('references/variables/' + source[0] + '_splits.obj', 'wb')
    pickle.dump(splits, file)

