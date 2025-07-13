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
import copy


device = torch.device(0 if torch.cuda.is_available() else 'cpu')

epoches = 20

 
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
        self.transforms = T.Compose([T.Resize((256, 256))])


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
        self.transforms = T.Compose([T.Resize((256, 256))])


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
        self.transforms = T.Compose([T.Resize((256, 256))])


    def __len__(self):
        return self.df_test.shape[0]


    def __getitem__(self, index):
        row = self.df_test.iloc[index]
        img = TV.to_tensor(Image.open(f"card_data_set/{row['filepaths']}"))
        img = self.transforms(img)
        label = row["card_labels"]

        return {"image": img, "caption": label}


class neuralNetworkV1(nn.Module):
    # The __init__ method is used to declare the layers that will be used in the forward pass.
    def __init__(self):
        super().__init__() # required because our class inherit from nn.Module
        # First convolutional layer with 3 input channels for RGB images, 16 outputs (filters).
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        # Second convolutional layer with 16 input channels to capture features from the previous layer, 16 outputs (filters).
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        # Third and fourth convolutional layers with 16 and 10 output channels respectively.
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        # Max pooling layer to reduce feature complexity.
        self.pooling = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # ReLU activation function for introducing non-linearity.
        self.relu = nn.ReLU()
        # Flatten the 2D output from the convolutional layers for the fully connected layer.
        self.flatten = nn.Flatten()
        # Fully connected layer connecting to 1D neurons, with 3 output features for 3 classes.
        self.linear = nn.Linear(in_features=256 * 4 * 4, out_features=13)
    
    # define how each data sample will propagate in each layer of the network
    def forward(self, x: torch.Tensor):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pooling(x)
        x = self.relu(self.conv3(x))
        x = self.pooling(x)
        x = self.relu(self.conv4(x))
        x = self.flatten(x)
        try:
            x = self.linear(x)
        except Exception as e:
            print(f"Error : Linear block should take support shape of {x.shape} for in_features.")
        return x

mdl = neuralNetworkV1().to(device)


def create_model():
    mdl = neuralNetworkV1().to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.AdamW(mdl.parameters(), lr=0.001)

    return mdl, loss_fn, opt


def create_datasets():
    tds = TrainingDataSet()
    testds = TestDataSet(tds.get_encoder())
    valds = ValidationData(tds.get_encoder())
    return tds, testds, valds 


def train_mdl(mdl, loss_fn, opt, trainloader, testloader):
    es = EarlyStopping()
    best_acc = 0
    for e in range(epoches):
        mdl.train(True)
        count = 0
        count_training = 0
        running_loss = 0.0
        acc = 0
        for i in tqdm(trainloader):
            y_hat = mdl(i["image"].to(device))
            loss = loss_fn(y_hat, i["caption"].to(device))
            acc += (torch.argmax(y_hat, 1) == i["caption"].to(device)).float().sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            running_loss += loss.item()
            count += 1
            count_training += len(i["caption"])
        acc /= count_training
        print(f'[{e + 1}], Model Train Accuracy {round(acc.item()*100,2)}, training loss: {running_loss / count:.3f}')

        mdl.eval()
        acc = 0
        count = 0
        test_loss = 0
        test_loss_lst = []
        count_loss = 0
        with torch.no_grad():
            for i in testloader:
                y_hat = mdl(i["image"].to(device))
                loss = loss_fn(y_hat, i["caption"].to(device))
                test_loss += loss.item()
                test_loss_lst.append(loss.item())
                acc += (torch.argmax(y_hat, 1) == i["caption"].to(device)).float().sum()
                count += len(i["caption"])
                count_loss += 1
        acc /= count
        print(f"Epoch: {e+1}, Model Test Accuracy {round(acc.item()*100,2)}, Model Test Loss {test_loss / count_loss:.3f}")
        if round(acc.item()*100,2) > best_acc:
            check_point = {
                "epoches" : e,
                "model": mdl.state_dict(),
                "optimizer": opt.state_dict(),
                "best_acc": best_acc
            }
            torch.save(check_point, "best_mdl.pt")
            best_acc = round(acc.item()*100,2)
        print(f"Best model so far: {best_acc}")
        if es(mdl, test_loss):
            print(f"ES STATUS: {es.status}")
            break


def validate_mdl(mdl, valloader, lb):

    check_point = torch.load("best_mdl.pt", weights_only=True)
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
    plt.savefig("conf matrix.jpg")
    print("SCORES")
    print("CLASSIFICATION REPORT")
    print(classification_report(y_real, y_hats, target_names=lb.classes_[:-1]))


def custom_eval(mdl, lb):
    files = ["test.jpg", "test2.jpg", "test3.jpg", "test4.jpg", "test5.jpg", "test6.jpg", "test7.jpg"]
    mdl.eval()
    for f in files:
        img = TV.to_tensor(Image.open(f))
        transforms = T.Compose([T.Resize((256, 256))])
        img = transforms(img)
        img = img[None, :, :, :]
        y_hat = mdl(img.to(device))
        res = torch.argmax(y_hat, 1)
        print(f"file {f}: awnser {lb.inverse_transform([res.item()])[0]}")


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_model = None
        self.best_loss = None
        self.counter = 0
        self.status = ""

    def __call__(self, model, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model.state_dict())
        elif self.best_loss - val_loss >= self.min_delta:
            self.best_model = copy.deepcopy(model.state_dict())
            self.best_loss = val_loss
            self.counter = 0
            self.status = f"Improvement found, counter reset to {self.counter}"
        else:
            self.counter += 1
            self.status = f"No improvement in the last {self.counter} epochs"
            if self.counter >= self.patience:
                self.status = f"Early stopping triggered after {self.counter} epochs."
                if self.restore_best_weights:
                    model.load_state_dict(self.best_model)
                return True
        return False


def get_feature_maps(mdl, valds):
    conv_weights = []
    conv_layers = []
    total_conv_layers = 0
    for module in mdl.children():
        if isinstance(module, nn.Conv2d):
            total_conv_layers += 1
            conv_weights.append(module.weight)
            conv_layers.append(module)

    print(f"Total convolution layers: {total_conv_layers}")
    transforms = T.Compose([T.Resize((256, 256))])
    img = None
    for i in valds:
        img = transforms(i["image"]).to(device)
    
    # Extract feature maps
    feature_maps = []  # List to store feature maps
    layer_names = []  # List to store layer names
    for layer in conv_layers:
        img = layer(img)
        feature_maps.append(img)
        layer_names.append(str(layer))

    # Display feature maps shapes
    print("\nFeature maps shape")
    for feature_map in feature_maps:
        print(feature_map.shape)

    # Process and visualize feature maps
    processed_feature_maps = []  # List to store processed feature maps
    for feature_map in feature_maps:
        feature_map = feature_map.squeeze(0)  # Remove the batch dimension
        mean_feature_map = torch.sum(feature_map, 0) / feature_map.shape[0]  # Compute mean across channels
        processed_feature_maps.append(mean_feature_map.data.cpu().numpy())

    print("\n Processed feature maps shape")
    for fm in processed_feature_maps:
        print(fm.shape)

    # Plot the feature maps
    fig = plt.figure(figsize=(12, 12))
    for i in range(len(processed_feature_maps)):
        ax = fig.add_subplot(5, 4, i + 1)
        ax.imshow(processed_feature_maps[i])
        ax.axis("off")
        ax.set_title(layer_names[i].split('(')[0], fontsize=30)
    plt.show()


if __name__ ==  "__main__":
    mdl, loss_fn, opt = create_model()
    tds, testds, valds = create_datasets()
    lb = tds.get_encoder()
    trainloader = DataLoader(tds, batch_size=32, shuffle=True, drop_last=True)
    testloader = DataLoader(testds, batch_size=32, drop_last=True)
    valloader = DataLoader(valds, batch_size=1, drop_last=True)
    train_mdl(mdl, loss_fn, opt, trainloader, testloader)
    validate_mdl(mdl, valloader, lb)
    custom_eval(mdl, lb)
    get_feature_maps(mdl, valloader)
