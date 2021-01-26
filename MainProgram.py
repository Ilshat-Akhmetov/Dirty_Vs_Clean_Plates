from Train import train_algorithm
from Parameters import *
import torchvision.datasets
from torch.utils.data import DataLoader
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import shutil
from tqdm import tqdm
import numpy as np
import torchvision
import zipfile
from Transforms import *
import torch
import subprocess




def clear_train_val_directories():
    bashCommandSecond = 'rm -rf train val'
    process = subprocess.Popen(bashCommandSecond.split(), stdout=subprocess.PIPE)
    process.communicate()  # run bash script

def get_data():
    clear_train_val_directories()
    # Input data files are available in the "../input/" directory.
    # For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

    # Any results you write to the current directory are saved as output.


    with zipfile.ZipFile('plates.zip', 'r') as zip_obj:
        # Extract all the contents of zip file in current directory
        zip_obj.extractall('kaggle/working/')

    print('After zip extraction:')
    print(os.listdir("kaggle/working/"))


    print(os.listdir(data_root))

    train_dir = 'train'
    val_dir = 'val'

    class_names = ['cleaned', 'dirty']

    for dir_name in [train_dir, val_dir]:
        for class_name in class_names:
            os.makedirs(os.path.join(dir_name, class_name), exist_ok=True)

    for class_name in class_names:
        source_dir = os.path.join(data_root, 'train', class_name)
        for i, file_name in enumerate(tqdm(os.listdir(source_dir))):
            if i % EachNthVal != 0:
                dest_dir = os.path.join(train_dir, class_name)
            else:
                dest_dir = os.path.join(val_dir, class_name)
            shutil.copy(os.path.join(source_dir, file_name), os.path.join(dest_dir, file_name))




    # train_dataset = torch.utils.data.ConcatDataset([
    #     torchvision.datasets.ImageFolder(train_dir, train_transforms),
    #     torchvision.datasets.ImageFolder(train_dir, train_transforms),
    #     torchvision.datasets.ImageFolder(train_dir, train_transforms),
    #     torchvision.datasets.ImageFolder(train_dir, train_transforms),
    #     torchvision.datasets.ImageFolder(train_dir, train_transforms),
    #     torchvision.datasets.ImageFolder(train_dir, train_transforms),
    #     torchvision.datasets.ImageFolder(train_dir, train_transforms),
    #     torchvision.datasets.ImageFolder(train_dir, train_transforms),
    #     torchvision.datasets.ImageFolder(train_dir, train_transforms),
    #     torchvision.datasets.ImageFolder(train_dir, train_transforms),
    #     torchvision.datasets.ImageFolder(train_dir, train_transforms),
    #     torchvision.datasets.ImageFolder(train_dir, train_transforms),
    #     torchvision.datasets.ImageFolder(train_dir, train_transforms),
    #     torchvision.datasets.ImageFolder(train_dir, train_transforms)
    # ])
    train_dataset = torch.utils.data.ConcatDataset([torchvision.datasets.ImageFolder(train_dir,train_transforms)
                                                    for i in range(TrainNumberGain)])
    val_dataset = torchvision.datasets.ImageFolder(val_dir, val_transforms)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=batch_size)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=batch_size)

    return train_dataloader, val_dataloader

def load_model(Model_Path):
    TheModel = model
    TheModel.load_state_dict(torch.load(Model_Path))
    TheModel = TheModel.to(device)
    TheModel.eval()
    return TheModel

def train_model(Model_Path_For_Extra_Training=None):
    train_autoloader, val_autoloader = get_data()
    The_Model=model
    if (Model_Path_For_Extra_Training!=None):
        The_Model = load_model(Model_Path_For_Extra_Training)
    train_algorithm(The_Model, loss, optimizer, scheduler, num_epochs, train_autoloader, val_autoloader, device)
    clear_train_val_directories()

def check_model(Model_Path):

    model = load_model(Model_Path)
    test_dir = 'test'
    if not os.path.isdir(os.path.join(test_dir, 'unknown')):
        shutil.copytree(os.path.join(data_root, 'test'), os.path.join(test_dir, 'unknown'))

    class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
        def __getitem__(self, index):
            original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
            path = self.imgs[index][0]
            tuple_with_path = (original_tuple + (path,))
            return tuple_with_path

    test_dataset = ImageFolderWithPaths('test', val_transforms)

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    test_predictions = []
    test_img_paths = []
    for inputs, labels, paths in tqdm(test_dataloader):
        inputs = inputs.to(device)
        #labels = labels.to(device)
        with torch.set_grad_enabled(False):
            preds = model(inputs)
        test_predictions.append(
            torch.nn.functional.softmax(preds, dim=1)[:, 1].data.cpu().numpy())
        test_img_paths.extend(paths)

    test_predictions = np.concatenate(test_predictions)

    submission_df = pd.DataFrame.from_dict({'id': test_img_paths, 'label': test_predictions})
    submission_df['label'] = submission_df['label'].map(lambda pred: 'dirty' if pred > 0.5 else 'cleaned')
    submission_df['id'] = submission_df['id'].str.replace('test/unknown/', '')
    submission_df['id'] = submission_df['id'].str.replace('.jpg', '')
    submission_df.set_index('id', inplace=True)
    submission_df.head(n=6)
    submission_df.to_csv('submission.csv')


if __name__ == "__main__":

    import random
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


    # train_model()
    # PATH = os.path.join("Weights", ("model_{}th_epoch.pth").format(num_epochs-1))
    # check_model(PATH)
    PATH = os.path.join("Weights", ("model_{}th_epoch.pth").format(15))
    check_model(PATH)
