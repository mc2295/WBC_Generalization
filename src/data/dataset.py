from torch.utils.data import Dataset
from torchvision import transforms
from data.transform_function import transform_color_transfer, transform_crop, transform_resolution
from fastai.vision.all import PILImage


class MyDataset(Dataset):
    def __init__(self, df, valid = False, transform = False):
        self.transform = transform
        self.df = df
        if valid:
            self.df = df.loc[df['is_valid'] == True]
        else:
            self.df = df.loc[df['is_valid'] == False]
    
    def __len__(self):
        return len(self.df)

    def _load_img_(self, img_path, dataset):
        img = transforms.ToTensor()(PILImage.create( '../data/Single_cells/' + img_path))
        img = transforms.Resize((224, 224))(img)
        if self.transform: 
            if dataset == 'barcelona': # images from barcelona are degraded 
                img = transforms.Compose([transform_resolution, transform_crop])(img)
            img = transform_color_transfer(img)
        return img
    
    def get_labels(self):
        return self.df['label'].tolist()
    
    def __getitem__(self, idx):
        
        label = self.df.iloc[idx]['label']
        dataset = self.df.iloc[idx]['dataset']
        img_path = self.df.iloc[idx]['name']
        img = self._load_img_(img_path, dataset)
        
        
        return img, label, img_path, dataset


