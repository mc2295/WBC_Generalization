from fastai.vision.all import *
from data.transform_functions import get_item
'''
This module creates
- SiameseImage: class (image1, image2, label) with label = 1 or 0
- SiameseTransform: generates pairs of images with label (SiameseImage) for train and valid sets, from splits and dic_labels
- show_batch : shows a siamese batch
'''
# import torch
def open_image(entry_path, fname, size=224):
    print(fname)
    img = PIL.Image.open(entry_path + fname).convert('RGB')
    img = img.resize((size, size))
    t = torch.Tensor(np.array(img))
    return t.permute(2,0,1).float()/255.0

class SiameseImage(fastuple):
    def show(self, ctx=None, **kwargs):
        if len(self) > 2:
            img1,img2,similarity = self
        else:
            img1,img2 = self
            similarity = 'Undetermined'
        if not isinstance(img1, Tensor):
            if img2.size != img1.size: img2 = img2.resize(img1.size)
            t1,t2 = tensor(img1),tensor(img2)
            t1,t2 = t1.permute(2,0,1),t2.permute(2,0,1)
        else: t1,t2 = img1,img2
        line = t1.new_zeros(t1.shape[0], t1.shape[1], 10)
        return show_image(torch.cat([t1,line,t2], dim=2), title=similarity, ctx=ctx, **kwargs)

class SiameseTransform(Transform):
    def __init__(self, splits, df, entry_path, transform = False):
        self.transform = transform
        self.entry_path = entry_path
        self.vocab = [[0,1]]
        self.list_labels_cat = df['label'].unique()
        self.df = df
        self.splbl2files = [{ l : [row for index, row in self.df.iloc[splits[i]].iterrows() if row['label'] == l] for l in self.list_labels_cat}
                            for i in range(2)]

        self.valid = {row['name']: self._draw(row,0) for index, row in self.df.iloc[splits[0]].iterrows()}
        self.train = {row['name']: self._draw(row,1) for index, row in self.df.iloc[splits[1]].iterrows()}
    def encodes(self, f):
        f2,same = self.valid.get(f['name'], self._draw(f, split=0))
        if self.transform:
            img1 = get_item(f, self.entry_path)
            img2 = get_item(f2, self.entry_path)
        else:
            img1,img2 = PILImage.create(self.entry_path + f['name']),PILImage.create(self.entry_path+ f2['name'])
        return SiameseImage(img1, img2, int(same))

    def _draw(self, f, split=0):
        same = random.random() < 0.5
        cls = f['label']
        if not same: cls = random.choice(L(l for l in self.list_labels_cat if l != cls))
        return random.choice(self.splbl2files[split][cls]),same

class Siamese2Label(Transform):
    def __init__(self, splits, df, row2, entry_path, transform = False):
        self.transform = transform
        self.entry_path = entry_path
        self.vocab = [[0,1]]
        self.list_labels_cat = df['label'].unique()
        self.df = df
        self.row2 = row2
        # self.splbl2files = [{ l : [row for index, row in self.df.iloc[splits[i]].iterrows() if row['label'] == l] for l in self.list_labels_cat}
        #                     for i in range(2)]

        self.valid = {row['name']: self._draw(row,0) for index, row in self.df.iloc[splits[0]].iterrows()}
        self.train = {row['name']: self._draw(row,1) for index, row in self.df.iloc[splits[1]].iterrows()}
    def encodes(self, f):
        f2 = self.row2
        if self.transform:
            img1 = get_item(f, self.entry_path)
            img2 = get_item(f2, self.entry_path)
        else:
            img1,img2 = PILImage.create(self.entry_path + f['name']),PILImage.create(self.entry_path+ f2['name'])
        return SiameseImage(img1, img2, f['label'])
    def _draw(self, f, split=0):
        return self.row2, f['label']
@typedispatch
def show_batch(x:SiameseImage, y, samples, ctxs=None, max_n=6, nrows=None, ncols=2, figsize=None, **kwargs):
    if figsize is None: figsize = (ncols*6, max_n//ncols * 3)
    if ctxs is None: ctxs = get_grid(min(x[0].shape[0], max_n), nrows=None, ncols=ncols, figsize=figsize)
    for i,ctx in enumerate(ctxs): SiameseImage(x[0][i], x[1][i], ['Not similar','Similar'][x[2][i].item()]).show(ctx=ctx)
