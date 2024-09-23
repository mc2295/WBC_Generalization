import config
import torch
from models.create_model import create_model
from data.main import train_loader, valid_loader
from tqdm import tqdm
from torch.autograd import Variable
from sklearn import preprocessing

# model = create_model('resnet', len(config.list_labels_cat)).to(config.device)
model = torch.load(config.model_path).to(config.device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam([*model.encoder.parameters(), *model.head.parameters()], lr = config.lr)
le = preprocessing.LabelEncoder()
le.fit(config.list_labels_cat)

for epoch in range(config.epoch):
    train_loss = []
    for img, label_cat, img_path, dataset in tqdm(train_loader):
        optimizer.zero_grad()
        img = Variable(img.to(config.device))
        labels = le.transform(label_cat)
        labels = torch.as_tensor(labels).to(config.device)
        out = model(img)
        loss = criterion(out, labels)
        train_loss.append(loss.item())
        loss.backward()
        optimizer.step()
        
    print('Train Loss: ', sum(train_loss)/len(train_loss))
    for  img, label_cat, img_path, dataset in tqdm(valid_loader):
        test_loss = []
        with torch.no_grad():
            img = Variable(img.to(config.device))
            labels = le.transform(label_cat)
            labels = torch.as_tensor(labels).to(config.device)
            out = model(img)
            loss = criterion(out, labels)
            test_loss.append(loss.item())
    print('Test Loss: ', sum(test_loss)/len(test_loss))
    torch.save(model, '../models/fine_tuned/' + config.source[0])
