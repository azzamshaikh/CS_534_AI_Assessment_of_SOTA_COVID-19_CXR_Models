# Import necessary libraries and modules

# Magic command to display matplotlib plots inline in Jupyter Notebook

import os             # Module for interacting with the operating system
import shutil         # Module for file operations
import random         # Module for generating random numbers
import torch          # PyTorch library for deep learning
import torchvision    # PyTorch library for computer vision tasks
import numpy as np    # Library for numerical computing

from PIL import Image          # Python Imaging Library for image processing
from matplotlib import pyplot as plt  # Library for creating plots and visualizations
from torch.utils.data import Dataset  # Base class for custom datasets in PyTorch
from torchvision import transforms  # Module for image transformations in PyTorch
from torch.utils.data import DataLoader  # DataLoader class for managing datasets and data loading in PyTorch
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt


import torchvision                    # PyTorch library for computer vision tasks
import torch                          # PyTorch library for deep learning
import torch.nn as nn                 # PyTorch module for building neural networks
import torch.optim as optim           # PyTorch module for optimization algorithms
from torch.utils.data import DataLoader, Dataset, random_split  # DataLoader class for managing datasets and data loading in PyTorch, random_split for splitting datasets
from torchvision import models, transforms            # Module for pre-trained models and image transformations in PyTorch
from torchvision.datasets import ImageFolder           # Dataset class for loading image folders in PyTorch
from sklearn.metrics import classification_report, confusion_matrix  # Module for generating classification reports and confusion matrices

from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix, matthews_corrcoef
import seaborn as sns


# Set a manual seed for reproducibility of results
torch.manual_seed(0)

# Print the PyTorch version being used
print('Using PyTorch version', torch.__version__)

"""Mount Google Drive to Dataset Directory"""

# Mount the Google drive
from google.colab import drive
drive.mount('/content/drive')



class ChestXRayDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Initialize the ChestXRayDataset.

        Args:
            root_dir (str): The root directory where the dataset is located.
            transform (callable, optional): A function/transform to apply to the images. Defaults to None.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.class_names = ['NORMAL', 'Viral Pneumonia', 'COVID-19']

        # Lists to store image paths and corresponding labels
        self.images = []
        self.labels = []

        # Loop through each class directory and store image paths with their labels
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(self.root_dir, class_name)
            for image_name in os.listdir(class_dir):
                self.images.append(os.path.join(class_dir, image_name))
                self.labels.append(class_idx)

    def __len__(self):
        """
        Get the total number of images in the dataset.

        Returns:
            int: Total number of images.
        """
        return len(self.images)

    def __getitem__(self, index):
        """
        Get an item (image and its label) from the dataset.

        Args:
            index (int): Index of the item to retrieve.

        Returns:
            tuple: Tuple containing the image and its corresponding label.
        """
        image_path = self.images[index]
        image = Image.open(image_path).convert('RGB')

        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)

        label = self.labels[index]
        return image, label

# Define transformations for image preprocessing
transform_common = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define transformations for image preprocessing for InceptionV3 Model
transform_inception = transforms.Compose([
    transforms.Resize(size=(299, 299)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define the root directory where the dataset is located
root_dir = '/content/drive/My Drive/COVID-19 Radiography Database'

# Create an instance of ChestXRayDataset
chest_xray_dataset = ChestXRayDataset(root_dir=root_dir, transform=transform_common)

# Create an instance of ChestXRayDataset for inceptionV3 Model
chest_xray_dataset_inception = ChestXRayDataset(root_dir=root_dir, transform=transform_inception)

# Define the sizes for your train, validation, and test sets
total_size = len(chest_xray_dataset)
train_size = int(0.7 * total_size)    # 70% for training
valid_size = int(0.15 * total_size)   # 15% for validation
test_size = total_size - train_size - valid_size  # Remaining 15% for testing

# Split the dataset
train_dataset, valid_dataset, test_dataset = random_split(chest_xray_dataset, [train_size, valid_size, test_size])

# Split the dataset
train_dataset_inception, valid_dataset_inception, test_dataset_inception = random_split(chest_xray_dataset_inception, [train_size, valid_size, test_size])

# Define the batch size
batch_size = 6

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Create data loaders for InceptionV3
train_loader_inception = DataLoader(train_dataset_inception, batch_size=batch_size, shuffle=True)
valid_loader_inception = DataLoader(valid_dataset_inception, batch_size=batch_size, shuffle=True)
test_loader_inception = DataLoader(test_dataset_inception, batch_size=batch_size, shuffle=False)


''' CHECK THE TRANSFORMED DATASET '''

# Check the length of the training dataset
chest_xray_dataset_length = len(chest_xray_dataset)
print("Number of images in the chest_xray dataset:", chest_xray_dataset_length)

# Check the length of the training dataset
train_dataset_length = len(train_dataset)
print("Number of images in the training dataset:", train_dataset_length)

# Check the length of the test dataset
test_dataset_length = len(test_dataset)
print("Number of images in the test dataset:", test_dataset_length)


# Check the length of the validation dataset
valid_dataset_length = len(valid_dataset)
print("Number of images in the valid dataset:", valid_dataset_length)


def visualize_batch(images, labels, class_names):
    plt.figure(figsize=(10, 5))
    for i in range(len(images)):
        # Denormalize the image
        image = images[i].numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)  # Clip values to ensure they are within [0, 1]

        plt.subplot(2, 3, i + 1)
        plt.imshow(image)
        plt.title(class_names[labels[i]])
        plt.axis('off')
    plt.show()

# Explore train dataset
print("Train Dataset:")
for images, labels in train_loader:
    visualize_batch(images, labels, chest_xray_dataset.class_names)
    break  # Show only the first batch for demonstration

# Explore test dataset
print("Test Dataset:")
for images, labels in test_loader:
    visualize_batch(images, labels, chest_xray_dataset.class_names)
    break  # Show only the first batch for demonstration


def show_images(images, labels, preds, class_names):
    num_images = len(images)
    num_cols = min(6, num_images)  # Ensure no more than 6 columns
    num_rows = (num_images + num_cols - 1) // num_cols  # Calculate number of rows

    plt.figure(figsize=(4 * num_cols, 4 * num_rows))  # Adjust figure size based on grid size
    for i, image in enumerate(images):
        plt.subplot(num_rows, num_cols, i + 1, xticks=[], yticks=[])
        image = image.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = image * std + mean
        image = np.clip(image, 0., 1.)
        plt.imshow(image)
        col = 'green'
        if preds[i] != labels[i]:
            col = 'red'

        plt.xlabel(f'{class_names[int(labels[i].numpy())]}')
        plt.ylabel(f'{class_names[int(preds[i].numpy())]}', color=col)
    plt.tight_layout()
    plt.show()

class_names = ['NORMAL', 'Viral Pneumonia', 'COVID-19']
images, labels = next(iter(train_loader))
show_images(images, labels, labels,class_names)

# Print structure of a batch from train_loader
for images, labels in train_loader:
    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)
    print("Example image tensor:")
    print(images[0])  # Print the first image tensor in the batch
    print("Example label:")
    print(labels[0])  # Print the first label in the batch
    break  # Stop after printing the first batch for demonstration purposes


''' self.class_names = ['NORMAL', 'Viral Pneumonia', 'COVID-19']
Each label corresponds to one of these classes. Therefore, tensor(2) corresponds to the class 'COVID-19' in this case.
So, whenever you see a label in the form of tensor(x), you can interpret it as the index x of the class name in the class_names list. '''


"""CREATING MODELS TO TRAIN THE DATSET ON"""

def get_pretrained_model(model_name, num_classes):

    # Define the loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    #If the DNN Model is VGG19
    if model_name == 'vgg':
        model = models.vgg16(pretrained=True)
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_features, num_classes)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)

    #If the DNN Model is Resnet50
    elif model_name == 'resnet':
        model = models.resnet50(pretrained=True)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)

     #If the DNN Model is Incepption
    elif model_name == 'inception':
        model = models.inception_v3(pretrained=True, aux_logits=True)  # Make sure to enable auxiliary logits)
        # Handle the auxilary net
        num_features_aux = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(num_features_aux, num_classes)
        # Handle the primary net
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
    else:
        raise ValueError("Model not supported")

    return model



""" Duplicating their exact approach in their Research Paper"""
def get_pretrained_model_rp(model_name, num_classes):
    if model_name == 'vgg':
        model = models.vgg16(pretrained=True)
        # Replace the classifier with a new one
        model.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
            nn.Softmax(dim=1)  # Adding Softmax for the classification output
        )
    elif model_name == 'resnet':
        model = models.resnet50(pretrained=True)
        num_features = model.fc.in_features
        # Modify the fully connected layer
        model.fc = nn.Sequential(
            nn.Linear(num_features, num_classes),
            nn.Softmax(dim=1)  # Adding Softmax for the classification output
        )
    elif model_name == 'inception':
        model = models.inception_v3(pretrained=True, aux_logits=True)
        num_features_aux = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(num_features_aux, num_classes)  # Modifying auxiliary classifier
        num_features = model.fc.in_features
        # Modify the primary classifier
        model.fc = nn.Sequential(
            nn.Linear(num_features, num_classes),
            nn.Softmax(dim=1)  # Adding Softmax for the classification output
        )
    else:
        raise ValueError("Model not supported")

    return model


''' GET MODELS FOR TRAINING THE NETWORK ARCHITECTURE '''

resnet50_Model = get_pretrained_model(model_name = 'resnet',num_classes= 3)
vgg19_model = get_pretrained_model(model_name = 'vgg',num_classes= 3)
inception_model =get_pretrained_model(model_name = 'inception',num_classes= 3)

resnet50_Model_rp = get_pretrained_model_rp(model_name = 'resnet',num_classes= 3)
vgg19_model_rp = get_pretrained_model_rp(model_name = 'vgg',num_classes= 3)
inception_model_rp =get_pretrained_model_rp(model_name = 'inception',num_classes= 3)

def show_preds(model):
  model.eval() # set to evaluation mode
  images, labels = next(iter(test_loader))
  outputs = model(images)
  _, preds = torch.max(outputs, 1)
  show_images(images, labels, preds,class_names)


""" CALL MODELS TO SEE THE EVALUATION """
#Calling Resnet50
show_preds(resnet50_Model)
#Calling Vgg19
show_preds(vgg19_model)
#Calling Inception
show_preds(inception_model)




""" Train the Model for One Epoch and Save it  """
def train_and_save_model(model, model_name, epochs):
    print('Starting training..')
    best_accuracy = 0.0  # Initialize best accuracy to 0

    # Initialise loss function and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
    loss_fn = torch.nn.CrossEntropyLoss()

    for e in range(0, epochs):
        print('=' * 20)
        print(f'Starting epoch {e + 1}/{epochs}')
        print('=' * 20)

        train_loss = 0.
        val_loss = 0.

        model.train()  # Set model to training phase

        for train_step, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if train_step % 20 == 0:
                print('Evaluating at step', train_step)
                accuracy = 0
                model.eval()  # Set model to eval phase
                for val_step, (images, labels) in enumerate(valid_loader):
                    outputs = model(images)
                    loss = loss_fn(outputs, labels)
                    val_loss += loss.item()
                    _, preds = torch.max(outputs, 1)
                    accuracy += (preds == labels).sum().item()

                val_loss /= (val_step + 1)
                accuracy = accuracy / len(valid_dataset)  # Calculate accuracy over the entire validation dataset
                print(f'Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}')
                #show_preds(model)

                if accuracy >= best_accuracy:
                    best_accuracy = accuracy
                    print('Performance condition satisfied, saving model..')
                    torch.save(model.state_dict(), f"{model_name}_best_weights.pth")
                    filename = f"{model_name}_best_weights.pth"
                    print(f"New best model saved with accuracy: {accuracy:.4f}")
                    print(f"Saved model weights as: {filename}")
                    best_accuracy = accuracy  # Update best accuracy
                    #return

                model.train()

        train_loss /= (train_step + 1)
        print(f'Training Loss: {train_loss:.4f}')
    print('Training complete..')

current_directory = os.getcwd()
print("Current directory:", current_directory)
new_directory = '/content/drive/My Drive/COVID-19 Radiography Database'

# Change the current directory to the new directory
os.chdir(new_directory)

# Print the new current directory to verify the change
print("Current directory:", os.getcwd())

"""TRAIN AND SAVE MODELS THE OLD WAY"""

#Train and Save ResNet-50 model
print("\nTraining and Saving ResNet-50 model")
train_and_save_model(resnet50_Model, 'resnet50_Model', epochs=1)

#Train and Save Vgg-19 model
print("\nTraining and Saving Vgg19 model")
train_and_save_model(vgg19_model, 'vgg19_model', epochs=1)


'''
def get_pretrained_model(model_name, num_classes):
    if model_name == 'vgg':
        model = models.vgg16(pretrained=True)
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_features, num_classes)
    elif model_name == 'resnet':
        model = models.resnet50(pretrained=True)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    elif model_name == 'inception':
        model = models.inception_v3(pretrained=True, aux_logits=True)  # Make sure to enable auxiliary logits
        num_features_aux = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(num_features_aux, num_classes)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    else:
        raise ValueError("Model not supported")

    return model
'''

def train_and_save_model_inception(model, model_name, epochs, train_loader, valid_loader, valid_dataset):
    print('Starting training..')
    best_accuracy = 0.0  # Initialize best accuracy to 0

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)

    for e in range(epochs):
        print('=' * 20)
        print(f'Starting epoch {e + 1}/{epochs}')
        print('=' * 20)

        train_loss = 0.

        model.train()  # Set model to training phase

        for train_step, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs, aux_outputs = model(images)  # Assuming the model returns both logits and auxiliary outputs

            #logits = outputs.logits  # Extract primary logits from the outputs tensor

            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if train_step % 20 == 0:
                print('Evaluating at step', train_step)
                accuracy = 0
                model.eval()  # Set model to eval phase
                with torch.no_grad():
                    for val_step, (images, labels) in enumerate(valid_loader):
                        outputs = model(images)  # Assuming the model returns both logits and auxiliary outputs
                        #logits = outputs.logits  # Extract primary logits from the outputs tensor
                        preds = torch.max(outputs, 1)
                        accuracy += (preds[1] == labels).sum().item()

                accuracy /= len(valid_dataset)  # Calculate accuracy over the entire validation dataset
                print(f'Validation Accuracy: {accuracy:.4f}')

                if accuracy >= best_accuracy:
                    best_accuracy = accuracy
                    print('Performance condition satisfied, saving model..')
                    torch.save(model.state_dict(), f"{model_name}_best_weights.pth")
                    print(f"New best model saved with accuracy: {accuracy:.4f}")

                model.train()

        train_loss /= (train_step + 1)
        print(f'Training Loss: {train_loss:.4f}')

    print('Training complete..')

inception_model =get_pretrained_model(model_name = 'inception',num_classes= 3)

#Train and Save InceptionV3 model
print("\nTraining and Saving InceptionV3-50 model")
train_and_save_model_inception(inception_model, 'inception_model', 1, train_loader_inception, valid_loader_inception, valid_dataset_inception)


"""RESEARCH PAPER TRAIN AND SAVE MODEL"""
#Train and Save ResNet-50 model
print("\nTraining and Saving ResNet-50 model")
train_and_save_model(resnet50_Model_rp, 'resnet50_Model_rc', epochs=1)



"""EVALUATE MODEL TRAINING"""

def evaluate_model_accuracy(model, model_name, test_loader, device):
    # Correct the path to load the best model weights
    model.load_state_dict(torch.load(f"{model_name}_best_weights_with accuracy.pth"))

    model.eval()  # Ensure model is in evaluation mode and on the correct device

    correct_predictions = 0
    total_predictions = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():  # No gradients needed
        for images, labels in test_loader:
            images, labels = images, labels

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            correct_predictions += (preds == labels).sum().item()
            total_predictions += labels.size(0)

            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_accuracy = correct_predictions / total_predictions
    print(f"Test Accuracy using best model: {test_accuracy:.4f}")
    print("Testing Complete.")

    # Compute multiclass AUC-ROC curve
    n_classes = len(np.unique(all_labels))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve((np.array(all_labels) == i).astype(int), np.eye(n_classes)[np.array(all_predictions)][:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute macro-average AUC-ROC score
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    roc_auc["macro"] = auc(all_fpr, mean_tpr)

    # Plot multiclass AUC-ROC curve
    plt.figure()
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f'ROC curve (class {i}) (area = {roc_auc[i]:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve - Multiclass')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(os.getcwd(), f"{model_name}_multiclass_roc_curve.png"))  # Save ROC curve
    plt.show()

    # Print macro-average AUC-ROC score
    print(f"Macro-average AUC-ROC score: {roc_auc['macro']:.4f}")

    # Generate classification report
    print("Classification Report:")
    classification_rep = classification_report(all_labels, all_predictions, target_names=chest_xray_dataset.class_names)
    print(classification_rep)

    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_predictions,normalize='true')

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt=".2f", xticklabels=chest_xray_dataset.class_names, yticklabels=chest_xray_dataset.class_names)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix (Normalized)')
    plt.savefig(os.path.join(os.getcwd(), f"{model_name}_confusion_matrix.png"))  # Save confusion matrix
    plt.show()

    # Compute Matthews correlation coefficient (MCC)
    mcc = matthews_corrcoef(all_labels, all_predictions)
    print(f"Matthew's Correlation Coefficient (MCC): {mcc:.4f}")

    # Save classification report and MCC to a text file
    with open(os.path.join(os.getcwd(), f"{model_name}_evaluation_metrics.txt"), "w") as file:
        file.write("Classification Report:\n")
        file.write(classification_rep)
        file.write("\n\n")
        file.write(f"Matthew's Correlation Coefficient (MCC): {mcc:.4f}")


""" Test the Model Training """
def test_model(model, model_name, test_loader, device):
    # Correct the path to load the best model weights
    model.load_state_dict(torch.load(f"{model_name}_best_weights_with accuracy.pth"))

    model.eval()  # Ensure model is in evaluation mode and on the correct device

    correct_predictions = 0
    total_predictions = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():  # No gradients needed
        for images, labels in test_loader:
            images, labels = images, labels

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            correct_predictions += (preds == labels).sum().item()
            total_predictions += labels.size(0)

            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_accuracy = correct_predictions / total_predictions
    print(f"Test Accuracy using best model: {test_accuracy:.4f}")
    print("Testing Complete.")

    # Compute multiclass AUC-ROC curve
    n_classes = len(np.unique(all_labels))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve((np.array(all_labels) == i).astype(int), np.eye(n_classes)[np.array(all_predictions)][:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute macro-average AUC-ROC score
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    roc_auc["macro"] = auc(all_fpr, mean_tpr)

    # Plot multiclass AUC-ROC curve
    plt.figure()
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f'ROC curve (class {i}) (area = {roc_auc[i]:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve - Multiclass')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(os.getcwd(), f"{model_name}_multiclass_roc_curve.png"))  # Save ROC curve
    plt.show()

    # Print macro-average AUC-ROC score
    print(f"Macro-average AUC-ROC score: {roc_auc['macro']:.4f}")

    # Generate classification report
    print("Classification Report:")
    classification_rep = classification_report(all_labels, all_predictions, target_names=chest_xray_dataset.class_names)
    print(classification_rep)

    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_predictions,normalize='true')

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt=".2f", xticklabels=chest_xray_dataset.class_names, yticklabels=chest_xray_dataset.class_names)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix (Normalized)')
    plt.savefig(os.path.join(os.getcwd(), f"{model_name}_confusion_matrix.png"))  # Save confusion matrix
    plt.show()


    # Save classification report to a text file
    with open(os.path.join(os.getcwd(), f"{model_name}_classification_report.txt"), "w") as file:
        file.write("Classification Report:\n")
        file.write(classification_rep)

evaluate_model_accuracy(resnet50_Model,'resnet50_Model', test_loader,torch.device)

evaluate_model_accuracy(vgg19_model,'vgg19_model', test_loader,torch.device)

evaluate_model_accuracy(inception_model,'inception_model', test_loader_inception,torch.device)
