from copy import deepcopy
from pathlib import Path
from time import time

import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch import stack, nn, max, cuda, device, set_grad_enabled, sum, cat, save
from torch.optim import Adam
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader
from torchvision.models import resnet34, ResNet34_Weights
from torchvision.transforms import Compose, Resize, ToTensor

class_id_map = {'bathtub': 0, 'bed': 1, 'chair': 2, 'desk': 3, 'dresser': 4, 'monitor': 5, 'nightstand': 6, 'sofa': 7,
                'table': 8, 'toilet': 9}


class ModelDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.nb_views = 2
        self.model_label_map = self._get_model_label_map()
        self.model_names = self._get_model_names()
        self.label_encoder = self._get_label_encoder()

    def _get_model_names(self):
        return list(self.model_label_map.keys())

    def _get_model_label_map(self):
        model_label_map = {}
        for item in Path(self.root).iterdir():
            label, model_no, view_id = item.stem.split('_')
            model_label_map[f'{label}_{model_no}'] = label
        return model_label_map

    def __len__(self):
        return len(self.model_names)

    def _transform(self, image):
        transform = Compose([Resize((224, 224)), ToTensor()])
        return transform(image)

    def __getitem__(self, index):
        model_name = self.model_names[index]
        model_view_files = Path(self.root).glob(f'{model_name}*')
        model = stack([self._transform(Image.open(view).convert('RGB')) for view in model_view_files])
        label = self.label_encoder[self.model_label_map[model_name]]
        return model, label

    def _get_label_encoder(self):
        return {label: idx for idx, label in enumerate(sorted(set(self.model_label_map.values())))}


# MULTI-VIEW CONVOLUTIONAL NEURAL NETWORK (MVCNN) ARCHITECTURE
class MVCNN(nn.Module):
    def __init__(self, num_classes=1000, weights=ResNet34_Weights.DEFAULT):
        super(MVCNN, self).__init__()
        resnet = resnet34(weights=weights)
        fc_in_features = resnet.fc.in_features
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(fc_in_features, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes)
        )

    def forward(self, inputs):  # inputs.shape = samples x views x height x width x channels
        inputs = inputs.transpose(0, 1)
        view_features = []
        for view_batch in inputs:
            view_batch = self.features(view_batch)
            view_batch = view_batch.view(view_batch.shape[0], view_batch.shape[1:].numel())
            view_features.append(view_batch)

        pooled_views, _ = max(stack(view_features), 0)
        outputs = self.classifier(pooled_views)
        return outputs


# DEFINE A FUNCTION TO TRAIN THE MODEL
def train_model(model, dataloaders, criterion, optimizer, _device, num_epochs=25):
    since = time()

    val_acc_history = []

    best_model_wts = deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            all_preds = []
            all_labels = []
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(_device)
                labels = labels.to(_device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    # Get model predictions
                    _, preds = max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += sum(preds == labels.data)
                all_preds.append(preds)
                all_labels.append(labels)

            epoch_loss = running_loss / len(dataloaders[phase].sampler.indices)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].sampler.indices)
            all_labels = cat(all_labels, 0)
            all_preds = cat(all_preds, 0)
            epoch_weighted_acc = accuracy_score(all_labels.cpu().numpy(), all_preds.cpu().numpy())

            print('{} Loss: {:.4f} - Acc: {:.4f} - Weighted Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc,
                                                                                epoch_weighted_acc))

            # deep copy the model
            if phase == 'val' and epoch_weighted_acc > best_acc:
                best_acc = epoch_weighted_acc
                best_model_wts = deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_weighted_acc)

        print()

    time_elapsed = time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


# FUNCTION TO GET PREDICTIONS
def mvcnn_pred(model_name, data_dir, model, _device):
    transform = Compose([Resize((224, 224)), ToTensor()])
    model_view_files = Path(data_dir).glob(f'{model_name}*')
    model_3d = stack([transform(Image.open(fname).convert('RGB')) for fname in model_view_files]).unsqueeze(0)
    model_3d = model_3d.to(_device)
    pred = nn.functional.softmax(model(model_3d), dim=1).argmax().item()
    return pred, {v: k for k, v in class_id_map.items()}[pred]


def main():
    # CREATE DATASET
    root = 'output/'
    dataset = ModelDataset(root)

    # CREATE STRATIFIED TRAIN-VALIDATION SPLIT INDICES
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
    y = [dataset.model_label_map[model] for model in dataset.model_names]
    train_indices, val_indices = next(sss.split(np.zeros(len(y)), y))

    # CREATE TRAIN AND VALIDATION DATA LOADERS
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=16, sampler=train_sampler, num_workers=8)
    val_loader = DataLoader(dataset, batch_size=16, sampler=val_sampler, num_workers=8)
    data_loaders = {'train': train_loader, 'val': val_loader}

    # BUILD AND VISUALIZE THE MODEL
    model = MVCNN(num_classes=10)

    # DEFINE THE DEVICE
    _device = device("cuda:0" if cuda.is_available() else "cpu")

    # FREEZE THE WEIGHTS IN THE FEATURE EXTRACTION BLOCK OF THE NETWORK (I.E. RESNET BASE)
    for param in model.features.parameters():
        param.requires_grad = False

    # TRAIN THE CLASSIFIER BLOCK OF THE MODEL (I.E TOP DENSE LAYERS)
    model.to(_device)
    EPOCHS = 40
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.classifier.parameters(), lr=0.0005)
    model, val_acc_history = train_model(model=model, dataloaders=data_loaders, criterion=criterion,
                                         optimizer=optimizer, _device=_device, num_epochs=EPOCHS)

    # SAVE CURRENT WEIGHTS OF THE MODEL (STAGE 1: FEATURE EXTRACTION)
    save(model.state_dict(), 'mvcnn_stage_1.pkl')

    # UNFREEZE ALL THE WEIGHTS OF THE NETWORK
    for param in model.parameters():
        param.requires_grad = True

    # FINE-TUNE THE ENTIRE MODEL (I.E FEATURE EXTRACTOR + CLASSIFIER BLOCKS) USING A VERY SMALL LEARNING RATE
    EPOCHS = 20
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.00005)  # We use a smaller learning rate

    model, val_acc_history = train_model(model=model, dataloaders=data_loaders, criterion=criterion,
                                         optimizer=optimizer, _device=_device, num_epochs=EPOCHS)

    # SAVE CURRENT WEIGHTS OF THE MODEL (STAGE 2: FINE-TUNING)
    save(model.state_dict(), 'mvcnn_stage_2.pkl')

    # _device = device("cuda:0" if cuda.is_available() else "cpu")
    # model = MVCNN(num_classes=10)
    # model.load_state_dict(load('mvcnn_stage_2.pkl'))
    # model.to(_device)

    # GET NEW PREDICTIONS FROM RAW IMAGES
    pred = mvcnn_pred('bed_0001', 'output/', model, _device)
    print(pred)


if __name__ == '__main__':
    main()
