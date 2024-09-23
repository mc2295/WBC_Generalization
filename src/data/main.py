import config
from torch.utils.data import DataLoader
from visualisation.show_batch import show_batch
from data.splits_and_df import SplitDataframe
from data.dataset import MyDataset


# Initialize the SplitDataframe object which is responsible for:
# - Loading the dataset(s) from the specified source(s).
# - Splitting the dataset based on the specified configuration.
# - Handling fine-tuning or full evaluation scenarios.

splitter = SplitDataframe(config.source, config.size, config.balanced, 
                          config.reference_path, config.list_labels_cat,
                          full_evaluation = config.full_evaluation, new_dic = config.new_dic, 
                          fine_tune = config.fine_tune)
df = splitter.create_combined_dataframe()
print('dataframe : ', len(df))


train_dataset = MyDataset(df, valid = False, transform = config.transform)
valid_dataset = MyDataset(df, valid = True, transform = config.transform)

train_loader = DataLoader(train_dataset, batch_size=config.batchsize, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=config.batchsize, shuffle=False)
