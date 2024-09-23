import pickle
import pandas as pd
import numpy as np



def load_dataframe(reference_path, source, list_labels_cat):

    """
    Load the dataframe, filter by a specific data source and 
    update class labels to generic categories.

    Parameters:
    reference_path (str): Path to the reference directory containing the dataframe and class dictionary.
    source (list): List of data sources to filter the dataframe.
    list_labels_cat (list): List of generic label categories to filter the dataframe.

    Returns:
    pd.DataFrame: Filtered and labeled dataframe with 'name', 'label', and 'dataset' columns.
    """

    with open(reference_path + '/variables/dic_classes.obj', 'rb') as file:
        dic_classes = pickle.load(file)
    dataframe = pd.read_csv(reference_path + '/variables/dataframes/df_labeled_images.csv') 
    source_mask = dataframe.image_dataset.isin(source) 
    dataframe = dataframe.loc[source_mask,['image_path', 'image_class', 'image_dataset']]
    dataframe.image_class = [dic_classes[x] for x in dataframe.image_class]  
    label_mask = dataframe.image_class.isin(list_labels_cat) 
    dataframe = dataframe.loc[label_mask]

    dataframe = dataframe.reset_index()
    dataframe= dataframe.rename(columns={'image_path': 'name', 'image_class': 'label', 'image_dataset': 'dataset'})
    return dataframe

class SplitDataframe():

    """
    Initialize the SplitDataframe object.

    Parameters:
    source (list): List of data sources for loading the dataframe.
    size (list): List containing the desired sample sizes for each source.
    balanced (bool): Whether to balance the dataset by labels.
    reference_path (str): Path to the reference directory containing data.
    list_labels_cat (list): List of generic label categories to filter the dataframe.
    full_evaluation (bool, optional): Flag to indicate if the entire dataset should be used for evaluation. Default is False.
    fine_tune (bool, optional): Flag to indicate if the dataframe is for fine-tuning. Default is False.
    """

    def __init__(self, source, size, balanced, reference_path, list_labels_cat,  full_evaluation = False, fine_tune = False):
        self.source = source
        self.reference_path = reference_path
        self.fine_tune = fine_tune
        self.full_evaluation = full_evaluation
        self.size = size
        self.balanced = balanced
        self.list_labels_cat = list_labels_cat
        
    
    def reduce_split(self, df, splits, size):
        
        """
        Reduce the dataframe to a subset based on split indices and desired sample size.

        Parameters:
        df (pd.DataFrame): The original dataframe to split.
        splits (tuple): Tuple containing training and test split indices.
        size (int): Desired size of the subset.

        Returns:
        pd.DataFrame: A reduced dataframe with a 'is_valid' column indicating training/test split.
        """

        df_train = df.loc[splits[0]]
        df_test = df.loc[splits[1]]
        label_cat = np.unique(df_train.label)
        if size != 0: # If size is specified, reduce the dataset accordingly
            if not self.balanced:
                df_train = df_train.sample(n = min(size, len(df_train)))
                df_test = df_test.sample(n= min(size, len(df_test)))

            else: 
                df_train = (df_train.groupby('label', as_index=False)
                        .apply(lambda x: x.sample(n=size // len(label_cat), replace = True))
                        .reset_index(drop=True))
                df_test = (df_test.groupby('label', as_index=False)
                        .apply(lambda x: x.sample(n=size // len(label_cat), replace = True))
                        .reset_index(drop=True))
        
        df_combined = pd.concat([df_train, df_test], ignore_index=True)
        df_combined['is_valid'] = [i>len(df_train) for i in range(len(df_combined))]
        return df_combined
    
    
    def create_combined_dataframe(self):
        """
        Load and combine dataframes for each source, handling splits and optional fine-tuning or evaluation.

        Returns:
        pd.DataFrame: A combined dataframe from all sources with applied splits and filtering.
        """
                
        dataframes = []
        for k in range(len(self.source)): # Loop through each data source

            if self.fine_tune: # Load a reduced dataframe for fine-tuning with <=100 samples 
                df_one_source = pd.read_csv(self.reference_path + '/variables/dataframes/for_fine_tune/df_' + self.source[k] + '_0.csv')
                new_split0 = df_one_source.index[df_one_source['is_valid'] == False].tolist()
                new_split1 = df_one_source.index[df_one_source['is_valid'] == True].tolist()
                splits_one_source = new_split0, new_split1

            elif self.full_evaluation: # Load all samples not used for fine-tuning to evaluate the model
                df_one_source_train = pd.read_csv(self.reference_path + '/variables/dataframes/for_fine_tune/df_' + self.source[k] + '_0.csv')
                new_split0 = df_one_source_train.loc[df_one_source_train['is_valid'] == False]['Unnamed: 0'].tolist()

                values = {i for i in new_split0}
                new_split1 = [k for k in splits_one_source[0] + splits_one_source[1] if k not in values]
                splits_one_source = new_split0, new_split1
            
            else: # Perform a normal 80/20 split for source training
                df_one_source = load_dataframe(self.reference_path, [self.source[k]], self.list_labels_cat)
                file_splits = open(self.reference_path + '/variables/'+ self.source[k] +'_splits.obj', 'rb')
                splits_one_source = pickle.load(file_splits)
                
            # Reduce the dataframe based on the size and balance criteria        
            df_one_source = self.reduce_split(df_one_source, splits_one_source, self.size[k])
            dataframes.append(df_one_source)
        # Concatenate all dataframes from different sources    
        return pd.concat(dataframes, ignore_index=True)

