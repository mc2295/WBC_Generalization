from fastai.vision.all import *
# import torch
def open_image(fname, size=224):
    print(fname)
    img = PIL.Image.open('../../Documents/These/' + fname).convert('RGB')
    img = img.resize((size, size))
    t = torch.Tensor(np.array(img))
    return t.permute(2,0,1).float()/255.0


def label_func(image_path, dic_labels, list_labels_cat):
    return list.index(list_labels_cat, dic_labels[image_path])

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
    def __init__(self, files, splits, dic_labels, list_labels_cat, list_labels):
        self.dic_labels = dic_labels
        self.list_labels_cat = list_labels_cat
        self.list_labels = list_labels
        self.splbl2files = [{l: [f for f in files[splits[i]] if label_func(f, dic_labels, list_labels_cat) == l] for l in list_labels}
                          for i in range(2)]
        self.valid = {f: self._draw(f,1) for f in files[splits[1]]}
        self.train = {f: self._draw(f,1) for f in files[splits[0]]}
    def encodes(self, f):
        f2,same = self.valid.get(f, self._draw(f,0))
        img1,img2 = PILImage.create('../../Documents/These/' + f),PILImage.create('../../Documents/These/' + f2)
        return SiameseImage(img1, img2, int(same))

    def _draw(self, f, split=0):
        same = random.random() < 0.5
        cls = label_func(f, self.dic_labels, self.list_labels_cat)
        if not same: cls = random.choice(L(l for l in self.list_labels if l != cls))
        return random.choice(self.splbl2files[split][cls]),same

def show_batch(x:SiameseImage, y, samples, ctxs=None, max_n=6, nrows=None, ncols=2, figsize=None, **kwargs):
    if figsize is None: figsize = (ncols*6, max_n//ncols * 3)
    if ctxs is None: ctxs = get_grid(min(x[0].shape[0], max_n), nrows=None, ncols=ncols, figsize=figsize)
    for i,ctx in enumerate(ctxs): SiameseImage(x[0][i], x[1][i], ['Not similar','Similar'][x[2][i].item()]).show(ctx=ctx)
