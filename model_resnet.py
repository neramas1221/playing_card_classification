import timm
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torchvision.transforms.functional as TV
from PIL import Image
from tqdm import tqdm
from torch.utils import data
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report


device = torch.device(0 if torch.cuda.is_available() else 'cpu')

epoches = 50


class TrainingDataSet(data.Dataset):
    def __init__(self):
        super().__init__()
        self.df = pd.read_csv("card_data_set/cards.csv")
        self.lb = LabelEncoder()
        _ = self.lb.fit(self.df["card type"])
        self.df["card_labels"] = self.lb.transform(self.df["card type"])
        self.df = self.df[~self.df["filepaths"].str.contains("output")]
        self.df_train = self.df[self.df["data set"] == "train"]
        self.df_train = self.df_train[~self.df_train["filepaths"].str.contains(".lnk")]
        self.df_train = self.df_train[self.df_train["card_labels"] != 13]
        self.transforms = T.Compose([T.Resize((256, 256)),
                                     T.Normalize(mean=0.5, std=0.5)])


    def __len__(self):
        return self.df_train.shape[0]

    
    def get_encoder(self):
        return self.lb
    

    def __getitem__(self, index):
        row = self.df_train.iloc[index]
        img = TV.to_tensor(Image.open(f"card_data_set/{row['filepaths']}"))
        img = self.transforms(img)

        label = row["card_labels"]

        return {"image": img, "caption": label}
 

class TestDataSet(data.Dataset):
    def __init__(self, lb):
        super().__init__()
        self.df = pd.read_csv("card_data_set/cards.csv")
        self.lb = lb
        _ = self.lb.fit(self.df["card type"])
        self.df["card_labels"] = self.lb.transform(self.df["card type"])
        self.df_test = self.df[self.df["data set"] == "test"]
        self.df_test = self.df_test[self.df_test["card_labels"] != 13]
        self.transforms = T.Compose([T.Resize((256, 256)),T.Normalize(mean=0.5, std=0.5)])


    def __len__(self):
        return self.df_test.shape[0]


    def __getitem__(self, index):
        row = self.df_test.iloc[index]
        img = TV.to_tensor(Image.open(f"card_data_set/{row['filepaths']}"))
        img = self.transforms(img)
        label = row["card_labels"]

        return {"image": img, "caption": label}


class ValidationData(data.Dataset):
    def __init__(self, lb):
        super().__init__()
        self.df = pd.read_csv("card_data_set/cards.csv")
        self.lb = lb
        _ = self.lb.fit(self.df["card type"])
        self.df["card_labels"] = self.lb.transform(self.df["card type"])
        self.df_test = self.df[self.df["data set"] == "valid"]
        self.df_test = self.df_test[self.df_test["card_labels"] != 13]
        self.transforms = T.Compose([T.Resize((256, 256)), T.Normalize(mean=0.5, std=0.5)])


    def __len__(self):
        return self.df_test.shape[0]


    def __getitem__(self, index):
        row = self.df_test.iloc[index]
        img = TV.to_tensor(Image.open(f"card_data_set/{row['filepaths']}"))
        img = self.transforms(img)
        label = row["card_labels"]

        return {"image": img, "caption": label}


def CNN_mdl():
    num_classes = 13 # Replace num_classes with the number of classes in your data

    # Load pre-trained model from timm
    model = timm.create_model('resnet50', pretrained=True)

    # Modify the model head for fine-tuning
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model


def create_model():
    mdl = CNN_mdl().to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(mdl.parameters(), lr=0.001)

    return mdl, loss_fn, opt


def create_datasets():
    tds = TrainingDataSet()
    testds = TestDataSet(tds.get_encoder())
    valds = ValidationData(tds.get_encoder())
    return tds, testds, valds 


def train_mdl(mdl, loss_fn, opt, trainloader, testloader):
    best_acc = 0
    for e in range(epoches):
        count = 0
        running_loss = 0.0
        for i in tqdm(trainloader):
            y_hat = mdl(i["image"].to(device))
            loss = loss_fn(y_hat, i["caption"].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            running_loss += loss.item()
            count += 1
        print(f'[{e + 1}] loss: {running_loss / count:.3f}')
        
        acc = 0
        count = 0
        for i in testloader:
            y_hat = mdl(i["image"].to(device))
            acc += (torch.argmax(y_hat, 1) == i["caption"].to(device)).float().sum()
            count += len(i["caption"])
        acc /= count
        print(f"Epoch: {e}, Model Accuracy {round(acc.item()*100,2)}")
        if round(acc.item()*100,2) > best_acc:
            check_point = {
                "epoches" : e,
                "model": mdl.state_dict(),
                "optimizer": opt.state_dict(),
                "best_acc": best_acc
            }
            torch.save(check_point, "best_mdl_suit.pt")
            best_acc = round(acc.item()*100,2)
        print(f"Best model so far: {best_acc}")


def validate_mdl(mdl, valloader, lb):
    check_point = torch.load("best_mdl_suit.pt", weights_only=True)
    mdl.load_state_dict(check_point["model"])
    mdl.eval()

    acc = 0
    count = 0
    y_real = []
    y_hats = []
    with torch.no_grad():
        for i in tqdm(valloader):
            y_hat = mdl(i["image"].to(device))
            acc += (torch.argmax(y_hat, 1) == i["caption"].to(device)).float().sum()
            count += len(i["caption"])
            y_real.append(i["caption"].item())
            y_hats.append(torch.argmax(y_hat, 1).item())
        
    acc /= count
    print(f"Model Accuracy {round(acc.item()*100,2)}")

    confmatrix = confusion_matrix(y_real, y_hats)
    cm_display = ConfusionMatrixDisplay(confusion_matrix  = confmatrix, display_labels = lb.classes_[:-1])
    cm_display.plot()
    plt.xticks(rotation = 45)
    plt.savefig("conf matrix all suits.jpg")
    print("CLASSIFICATION REPORT")
    print(classification_report(y_real, y_hats, target_names=lb))


def custom_eval(mdl, lb):
    check_point = torch.load("best_mdl_suit.pt", weights_only=True)
    mdl.load_state_dict(check_point["model"])
    mdl.eval()
    files = ["test.jpg", "test2.jpg", "test3.jpg", "test4.jpg", "test5.jpg", "test6.jpg", "test7.jpg"]
    for f in files:
        img = TV.to_tensor(Image.open(f))
        transforms = T.Compose([T.Resize((256, 256)), T.Normalize(mean=0.5, std=0.5)])
        img = transforms(img)
        img = img[None, :, :, :]
        y_hat = mdl(img.to(device))
        res = torch.argmax(y_hat, 1)
        print(f"file {f}: awnser {lb[res.item()]}")


if __name__ ==  "__main__":
    mdl, loss_fn, opt = create_model()
    tds, testds, valds = create_datasets()
    lb = tds.get_encoder()
    trainloader = DataLoader(tds, batch_size=12, shuffle=True, drop_last=True)
    testloader = DataLoader(testds, batch_size=12, drop_last=True)
    valloader = DataLoader(valds, batch_size=1, drop_last=True)
    train_mdl(mdl, loss_fn, opt, trainloader, testloader)
    validate_mdl(mdl, valloader, lb)
    custom_eval(mdl, lb)
    

