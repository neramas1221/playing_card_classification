import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils import data
from PIL import Image
import torchvision.transforms.functional as TV
import pandas as pd
import torchvision.transforms as T
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report
import matplotlib.pyplot as plt


device = torch.device(0 if torch.cuda.is_available() else 'cpu')

epoches = 100

 
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
        self.transforms = T.Compose([T.Resize((256, 256)), 
                                             T.Normalize(mean=0.5, std=0.5)])
        self.labels = self.df["labels"].unique()


    def __len__(self):
        return self.df_train.shape[0]

    
    def get_encoder(self):
        return self.lb
    

    def __getitem__(self, index):
        row = self.df_train.iloc[index]
        img = TV.to_tensor(Image.open(f"card_data_set/{row['filepaths']}"))
        img = self.transforms(img)

        label = row["class index"]

        return {"image": img, "caption": label}
 

class TestDataSet(data.Dataset):
    def __init__(self, lb):
        super().__init__()
        self.df = pd.read_csv("card_data_set/cards.csv")
        self.lb = lb
        _ = self.lb.fit(self.df["card type"])
        self.df["card_labels"] = self.lb.transform(self.df["card type"])
        self.df_test = self.df[self.df["data set"] == "test"]
        self.transforms = T.Compose([T.Resize((256, 256)), 
                                        T.Normalize(mean=0.5, std=0.5)])


    def __len__(self):
        return self.df_test.shape[0]


    def __getitem__(self, index):
        row = self.df_test.iloc[index]
        img = TV.to_tensor(Image.open(f"card_data_set/{row['filepaths']}"))
        img = self.transforms(img)
        label = row["class index"]

        return {"image": img, "caption": label}


class ValidationData(data.Dataset):
    def __init__(self, lb):
        super().__init__()
        self.df = pd.read_csv("card_data_set/cards.csv")
        self.lb = lb
        _ = self.lb.fit(self.df["card type"])
        self.df["card_labels"] = self.lb.transform(self.df["card type"])
        self.df_test = self.df[self.df["data set"] == "valid"]
        # self.df_test = self.df_test[self.df_test["card_labels"] != 20]
        self.transforms = T.Compose([T.Resize((256, 256)), 
                                        T.Normalize(mean=0.5, std=0.5)])


    def __len__(self):
        return self.df_test.shape[0]


    def __getitem__(self, index):
        row = self.df_test.iloc[index]
        img = TV.to_tensor(Image.open(f"card_data_set/{row['filepaths']}"))
        img = self.transforms(img)
        label = row["class index"]

        return {"image": img, "caption": label}


class CNN_mdl(nn.Module):
    def __init__(self):
        super().__init__()
        ##### BLOCK 1 #####
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(3,3), padding=1)
        self.act1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(64)
        self.max1 = nn.MaxPool2d(kernel_size=(2,2))
        self.drop_out1 = nn.Dropout(0.2)
        ##### BLOCK 2 #####
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(3,3), padding=1)
        self.act2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(64)
        self.max2 = nn.MaxPool2d(kernel_size=(2,2))
        self.drop_out2 = nn.Dropout(0.2)
        ##### BLOCK 3 #####
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3,3), padding=1)
        self.act3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(64)
        self.max3 = nn.MaxPool2d(kernel_size=(2,2))
        self.drop_out3 = nn.Dropout(0.2)

        self.flattern = nn.Flatten()
        self.lin1 = nn.Linear(64 * 32 * 32, 512)
        self.act4 = nn.ReLU()
        self.drop_out3 = nn.Dropout(0.2)
        self.out = nn.Linear(512,53)
    

    def forward(self, x):        
        ##### BLOCK 1 #####
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.max1(x)
        x = self.drop_out1(x)

        ##### BLOCK 2 #####
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.max2(x)
        x = self.drop_out2(x)

        ##### BLOCK 3 #####
        x = self.act3(self.bn3(self.conv3(x)))
        x = self.max3(x)
        x = self.drop_out3(x)

        x = self.flattern(x)

        x = self.act4(self.lin1(x))
        x = self.drop_out3(x)
        x = self.out(x)

        return x
    

def create_model():
    mdl = CNN_mdl().to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.AdamW(mdl.parameters(), lr=0.001, weight_decay=0.0001)

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
    cm_display = ConfusionMatrixDisplay(confusion_matrix  = confmatrix, display_labels = lb)
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
        transforms = T.Compose([T.Resize((256, 256))])
        img = transforms(img)
        img = img[None, :, :, :]
        y_hat = mdl(img.to(device))
        res = torch.argmax(y_hat, 1)
        print(f"file {f}: awnser {lb[res.item()]}")


if __name__ ==  "__main__":
    mdl, loss_fn, opt = create_model()
    tds, testds, valds = create_datasets()
    lb = tds.get_encoder()
    labels = tds.labels
    trainloader = DataLoader(tds, batch_size=12, shuffle=True, drop_last=True)
    testloader = DataLoader(testds, batch_size=12, drop_last=True)
    valloader = DataLoader(valds, batch_size=1, drop_last=True)
    train_mdl(mdl, loss_fn, opt, trainloader, testloader)
    validate_mdl(mdl, valloader, labels)
    custom_eval(mdl, labels)

    

