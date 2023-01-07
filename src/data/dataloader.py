from fastai.vision.all import *


def label_func(image_path, df):
    return list.index(list_labels_cat, df[df['image_path'] == image_path]['image_class'].iloc[0])



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

    def __init__(self, files, splits, dataframe_source, list_labels):
        self.list_labels, self.dataframe_source = list_labels, dataframe_source
        self.splbl2files = [{l: [f for f in files[splits[i]] if label_func(f, dataframe_source) == l] for l in list_labels}
                          for i in range(2)]
        self.valid = {f: self._draw(f,1) for f in files[splits[1]]}
        self.train = {f: self._draw(f,1) for f in files[splits[0]]}
    def encodes(self, f):
        f2,same = self.valid.get(f, self._draw(f,0))
        img1,img2 = PILImage.create(f),PILImage.create(f2)
        return SiameseImage(img1, img2, int(same))

    def _draw(self, f, split=0):
        same = random.random() < 0.5
        cls = label_func(f, self.dataframe_source)
        if not same: cls = random.choice(L(l for l in self.list_labels if l != cls))
        return random.choice(self.splbl2files[split][cls]),same