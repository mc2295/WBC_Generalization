import pickle
import pandas as pd
import numpy as np
from fastai.vision.all import *
from sklearn.model_selection import train_test_split
# from data.transform_functions import transform_color_transfer, transform_resolution, transform_crop
from torchvision import transforms
from data.transform_functions import get_item

'''
This module creates dataloader of images
'''
list_labels_cat = ['basophil', 'eosinophil', 'erythroblast', 'lymphocyte', 'monocyte', 'neutrophil']


def create_df(reference_path, source, list_labels_cat):
    '''
    input:
    - reference_path : the path to go to 'reference' folder,
    - source : the name of the dataset(s), ex : ['barcelona', 'saint_antoine']
    - list_label_cat:list of labels, ex: ['basophil', 'eosinophil', 'eyrtroblast', 'monocyte']
    return:
    - a dataframe [name, label, dataset], ex: ['data/Single_cells/matek/NGB/NGB_01600.jpg', 'neutrophil', 'matek'] with normalised label, and images from source dataset.
    '''

    file = open(reference_path + '/variables/dic_classes.obj', 'rb') # the dic of correspondance between the name of the class in the dataset and the normalised name. Ex: {'neutrophile_0' : 'neutrophil'}
    dic_classes = pickle.load(file)
    dataframe = pd.read_csv(reference_path + '/variables/dataframes/df_labeled_images.csv')  # dataframe with : [image_path,image_name,image_class,image_dataset,size,transformed_image_path]
    source_mask = dataframe.image_dataset.isin(source) # take images from the dataset source
    dataframe = dataframe.loc[source_mask,['image_path', 'image_class', 'image_dataset']] #keep only columns 'image_path', 'image_class', 'image_dataset'
    dataframe.image_class = [dic_classes[x] for x in dataframe.image_class]  # maps the class names to corresponding normalised name
    label_mask = dataframe.image_class.isin(list_labels_cat) # keeps only classes that are in list_labels_cat
    dataframe = dataframe.loc[label_mask]

    dataframe = dataframe.reset_index()
    dataframe= dataframe.rename(columns={'image_path': 'name', 'image_class': 'label', 'image_dataset': 'dataset'})
    return dataframe

def create_splits(entry_path, source, df, size, balanced = True, df_filename = None, full_evaluation = False):
    '''
    input :
    - entry_path : '../' pour un .ipynb, '' pour un .py
    - source : source dataset ex: ['barcelona', 'saint_antoine']
    - df : the dataframe from create_df
    - size : the number of images I want by dataset. 0 means I want all images.
    - balanced: if I want the images to be balanced in terms of class
    - files_to_look_at : if I want to take images only among certain images path (ex: the worst loss)
    returns :
    - splits : the indexes of the dataframe rows that will be used by the data loader. splits = index_train, index_test
    '''


    if df_filename is not None:
        df_0 = pd.DataFrame(columns = ['index', 'name', 'label', 'dataset', 'is_valid'])
        for i in df_filename:
            file = open(i, 'rb')
            df_1 = pickle.load(file)
            df_0 = pd.concat([df_0, df_1], axis = 0)

        df_reduced = df_0
        if full_evaluation:
            filename = df_reduced.loc[df_reduced['is_valid'] == False].name.tolist()
            list_index_tot = df['index'].tolist()
            list_index_reduced = df.index.tolist()
            splits_train = df_reduced.loc[df_reduced['is_valid'] == False]['index'].tolist()
            random.shuffle(list_index_tot) #only for matek
            splits_test = [list_index_tot[i] for i in range(len(list_index_reduced)) if df.iloc[list_index_reduced[i]]['name'] not in filename][:10000]

        else:
            splits_train = df_reduced.loc[df_reduced['is_valid'] == False]['index'].tolist()
            splits_test = df_reduced.loc[df_reduced['is_valid'] == True]['index'].tolist()
        return splits_train, splits_test
    nb_images = 0
    splits = [], []

    for k in range(len(source)):
        # for a given dataset :
        file_splits = open(entry_path + 'references' + '/variables/'+ source[k] +'_splits.obj', 'rb') # train and test splits are stored in a file associated to the dataset
        splits_one_source = pickle.load(file_splits)
        splits_one_source = [[n + nb_images for n in splits_one_source[i]] for i in range(2)] # I add an offset of nb_of_image for each new dataset.

        if size[k] !=0: # I want to take only a certain number of images (size[k]) this dataset
            res = []
            random.shuffle(splits_one_source[0])
            df_one_source = df.loc[splits_one_source[0]] # I take rows of this dataset (train images)
            classes_one_source = df_one_source['label'].unique() # all classes in this dataset

            if balanced: # I want size[k] images with balanced class
                class_prop = [size[k]//len(classes_one_source) for i in range(len(classes_one_source))] # nb of images by class
                for i, class_name in enumerate(classes_one_source):
                    df_by_class = df_one_source.loc[df_one_source['label'] == class_name] # I take rows of one class
                    splits_by_class = df_by_class.index.tolist() # I take indexes of these rows
                    res += splits_by_class[:class_prop[i]] # I keep only class_prop[i] indexes for this class.

                splits_one_source = res, splits_one_source[1][:size[k]]

            else:
                res = splits_one_source[0][:size[k]] # keep only size[k] images
                classes_fine_tuning = df['label'].loc[splits_one_source[0][:size[k]]].unique() # the classes in the fine tuning subset. We want to have all the classes of the dataset in this subset.
                while len(classes_fine_tuning) < len(classes_one_source):
                    random.shuffle(splits_one_source[0])
                    classes_fine_tuning = df['label'].loc[splits_one_source[0][:size[k]]].unique()
                    res = splits_one_source[0][:size[k]]
            splits_one_source = res, splits_one_source[1][:size[k]]
        splits = [splits[j] + df.iloc[splits_one_source[j]]['index'].tolist() for j in range(2)]
        # splits = [splits[j] + splits_one_source[j] for j in range(2)] # I add the splits of this dataset in the global split
        nb_images += len(df.loc[df['dataset'] == source[k]]) # add the total number of images of this dataset

    return splits

class CustomImageDataset():
    '''
    __getitem__ function returns image and path from a row index, applying a transform if needed. This transform will be different depending on the dataset.
    '''
    def __init__(self, df, entry_path, vocab, transform = False, valid = False):
        self.transform = transform
        if valid:
            self.df = df.loc[df['is_valid'] == True]
        else:
            self.df = df.loc[df['is_valid'] == False]
        self.entry_path = entry_path # '../' for a notebook, '' for a Python file.
        self.list_labels_cat = list_labels_cat
        # self.vocab = list_labels_cat # labels of a dataloader
        self.vocab = [list_labels_cat]
        # self.dic_label = {k : i for i, k in enumerate(self.vocab)} # gives a number per label {'basophil': 1, ...}
        self.dic_label = {k : i for i, k in enumerate(list_labels_cat)}
        self.name = self.df['name'].tolist()
        self.dataset = self.df['dataset'].tolist()
        self.label = self.df['label'].tolist()
        self.index = self.df['index'].tolist()

    def __getitem__(self, idx):
        if self.transform:
            img = get_item(self.df.iloc[idx], self.entry_path) # transformed image, with transform depending on the dataset.
        else:
            img = PILImage.create(self.entry_path + self.name[idx]) # create img without transform
            img = Resize(226)(img)

        return img, self.dic_label[self.label[idx]] # the label has to be numerical !

    def __len__(self):
        return len(self.df)

def create_dataloader_single_image(entry_path, batchsize, df, splits, transform = False):
    '''
    input :
    - entry_path : '../' pour un .ipynb, '' pour un .py
    - batchsize
    - df: from create_df
    - splits: from create_splits

    return :
    - dls : dataloader
    '''

    df = df.loc[df['index'].isin(splits[0] + splits[1])]
    df = df.assign(is_valid = False)

    # Set 'is_valid' to True for rows where 'index' is in 'valid_index'
    df.loc[df['index'].isin(splits[1]), 'is_valid'] = True
    dls = ImageDataLoaders.from_df(df, path = entry_path, bs = batchsize, fn_col='name', label_col='label', valid_col = 'is_valid')
    custom_ds_train = CustomImageDataset(df, entry_path, dls.vocab, transform = transform, valid = False) # to add transforms
    custom_ds_valid = CustomImageDataset(df, entry_path, dls.vocab, transform = transform, valid = True) # to add the transforms
    dls.train.dataset = custom_ds_train
    dls.valid.dataset = custom_ds_valid
    dls.cuda()

    return dls

def create_dls(entry_path, source, siamese_head, batchsize = 32, size=[0,0,0,0], transform = False, balanced = False, df_filename = None, full_evaluation = False):
    '''
    returns a dataloader (siamese or not) from the desired dataset
    warning: if full evaluation is True, we need df_filename to be not None
    '''
    source.sort()
    df_reduced = create_df(entry_path + 'references', source, list_labels_cat) #inutile si df_filename is not None
    splits = create_splits(entry_path, source, df_reduced, size, balanced = balanced, df_filename = df_filename, full_evaluation = full_evaluation)

    return create_dataloader_single_image(entry_path, batchsize, df_reduced, splits, transform = transform)

def create_valid_train_test_splits(reference_path, source, list_labels_cat):
    '''
    create a test split and save it for final evaluation
    '''
    df = create_df(reference_path, source, list_labels_cat)
    array_files = df['name'].tolist()
    array_class = df['label'].tolist()
    mapping_dic = {}
    for i, file in enumerate(array_files):
        mapping_dic[file] = i

    X_train_valid, X_test, y_train_valid, y_test = train_test_split(array_files, array_class, test_size=0.1, random_state=42)
    test_index = [mapping_dic[i] for i in X_test]
    file = open('../references/variables/' + source[0] + '_test_index.obj', 'wb')
    pickle.dump(test_index, file)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_valid, y_train_valid, test_size = 0.2, random_state = 42)
    train_index = [mapping_dic[i] for i in X_train]
    print('fini train')
    valid_index = [mapping_dic[i] for i in X_valid]
    print('fini valid')
    splits = train_index, valid_index
    file = open('../references/variables/' + source[0] + '_splits.obj', 'wb')
    pickle.dump(splits, file)

def create_df_reduced(entry_path, source, size, balanced = True):
    df = create_df(entry_path + 'references', source, list_labels_cat)
    splits = create_splits('../', source, df, size, balanced = balanced)
    df = df.loc[df['index'].isin(splits[0] + splits[1])]
    # df = df.iloc[splits[0] + splits[1]]

    df = df.assign(is_valid = [i >= len(splits[0]) for i in range(len(splits[0] + splits[1]))]) # I add a column is_valid, which is true for an image in the valid set, and false if not.

    file = open('../references/variables/dataframes/reduced/' + 'df_'+ source[0] + '_0', 'wb')
    pickle.dump(df, file)
