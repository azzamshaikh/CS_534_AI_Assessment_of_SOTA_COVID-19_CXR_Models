import os                           # Module for interacting with the operating system
import numpy as np                  # Library for numerical computing
import seaborn as sns
import copy
import matplotlib.pyplot as plt
import torch                        # PyTorch library for deep learning
import torch.nn as nn               # PyTorch module for building neural networks

# Python Imaging Library for image processing
from PIL import Image
# DataLoader class for managing datasets and data loading in PyTorch, random_split for splitting datasets
from torch.utils.data import DataLoader, Dataset, random_split
# Module for pre-trained models and image transformations in PyTorch
from torchvision import models, transforms

from sklearn.metrics import accuracy_score, roc_curve, auc, classification_report, confusion_matrix, matthews_corrcoef
from tqdm import tqdm


class ChestXRayDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Initialize the ChestXRayDataset class to load custom dataset

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


def visualize_batch(images, labels, class_names):
    """Function to visualize a batch"""
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


def show_images(images, labels, preds, class_names):
    """Function to show images"""
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


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device):
    """
    Helper function to run a training step
    :param model: An AlexNet model object to be passed
    :param dataloader: A dataloader object to be passed
    :param loss_fn: A loss function object to be passed
    :param optimizer: A optimizer object to be passed
    :param device: A device name to be passed
    :return: Return train loss and accuracy
    """
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

        # print('so far trained',batch,'out of',len(dataloader))

    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


def run_test_step(model: torch.nn.Module,
                  dataloader: torch.utils.data.DataLoader,
                  loss_fn: torch.nn.Module,
                  device):
    """
    Helper function to run a test step
    :param model: AlexNet model object to be passed
    :param dataloader: A dataloader object to be passed
    :param loss_fn: A loss function object to be passed
    :param device: A device name to be passed
    :return: Return test loss and accuracy
    """
    # Put model in eval mode
    model.eval()

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item() / len(test_pred_labels))

            # print('so far tested',batch,'out of',len(dataloader))

    # Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc


def train(device,
          model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5):
    """
    Function to train the model
    :param device: A device name to be passed
    :param model: A specific AlexNet model object to be trained
    :param train_dataloader: A training dataset dataloader object to be passed
    :param test_dataloader: A testing dataset dataloader object to be passed
    :param optimizer: A optimizer object to be passed
    :param loss_fn: A loss function object to be passed
    :param epochs: The number of epochs to be run. Default is 5
    :return: Returns the training results and a break flag if testing accuracy is greater than 95%
    """

    # Create empty results dictionary
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []
               }

    break_flag = False
    print("=" * 40)
    print('Beginning training')
    print("=" * 40)
    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        print('Running epoch:', epoch)
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device)
        test_loss, test_acc = run_test_step(model=model, dataloader=test_dataloader, loss_fn=loss_fn, device=device)

        # Print out what's happening
        print(
            f"\nEpoch: {epoch + 1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    # Return the filled results at the end of the training and the break flag status
    return results


def evaluate_model(model, test_set_loader, device, class_names):
    """
    Function to evaluate model accuracy
    :param model: A specific AlexNet model object to be evaluated
    :param test_set_loader: A test dataset dataloader object to be passed
    :param device: A device name to be passed
    :param dataset: The original dataset to be passed (used for classification report
    :param class_names: Classification names to add to a plot
    :return: Return the model accuracy
    """
    # Put model in eval mode
    model.eval()

    # Initialize lists and counters
    correct_predictions = 0
    total_predictions = 0
    all_predictions = []
    all_labels = []

    # No gradients needed
    with torch.no_grad():
        # Loop through test dataset dataloader and obtain predictions from the model
        for images, labels in test_set_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            pred_prob = torch.softmax(outputs, dim=1)
            preds = torch.argmax(pred_prob, dim=1)

            correct_predictions += (preds == labels).sum().item()
            total_predictions += labels.size(0)

            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Print accuracy score
    print(f"Test Accuracy with scikit function: {accuracy_score(all_labels, all_predictions)}")
    print("Testing Complete.")
    # Generate classification report
    print("Classification Report:")
    classification_rep = classification_report(all_labels, all_predictions, target_names=class_names)
    print(classification_rep)

    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_predictions, normalize='true')
    print(cm)
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt=".2f", xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix (Normalized)')
    plt.savefig(os.path.join(os.getcwd(), "confusion_matrix.png"))  # Save confusion matrix
    plt.show()

    accuracy = accuracy_score(all_labels, all_predictions)
    print('Accuracy score:', accuracy)

    mcc = matthews_corrcoef(all_labels, all_predictions)
    print('MCC:', mcc)

    # Compute multiclass AUC-ROC curve
    n_classes = len(np.unique(all_labels))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve((np.array(all_labels) == i).astype(int),
                                      np.eye(n_classes)[np.array(all_predictions)][:, i])
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
    plt.savefig(os.path.join(os.getcwd(), "multiclass_roc_curve.png"))  # Save ROC curve
    plt.show()

    # Print macro-average AUC-ROC score
    print(f"Macro-average AUC-ROC score: {roc_auc['macro']:.4f}")

    return accuracy_score(all_labels, all_predictions)


def main():
    # Print the PyTorch version being used
    print('Using PyTorch version:', torch.__version__)

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print('Using device:', device)

    ''' Create transformations for the dataloaders '''

    transform_128 = transforms.Compose([
        transforms.Resize(size=(128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_128_training = transforms.Compose([
        transforms.Resize(size=(128, 128)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_224 = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_224_training = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_229 = transforms.Compose([
        transforms.Resize(size=(229, 229)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_229_training = transforms.Compose([
        transforms.Resize(size=(229, 229)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    root_dir = "COVID-19_Radiography_Dataset"

    # Create an instance of ChestXRayDataset
    chest_xray_dataset = ChestXRayDataset(root_dir=root_dir, transform=None)

    # Define the sizes for your train, validation, and test sets
    total_size = len(chest_xray_dataset)
    train_size = int(0.7 * total_size)    # 70% for training
    valid_size = int(0.15 * total_size)   # 15% for validation
    test_size = total_size - train_size - valid_size  # Remaining 15% for testing

    train_dataset, valid_dataset, test_dataset = random_split(chest_xray_dataset, [train_size, valid_size, test_size])

    ''' CHECK THE TRANSFORMED DATASET '''

    # Check the length of the training dataset
    chest_xray_dataset_length = len(chest_xray_dataset)
    print("Number of images in the chest_xray dataset:", chest_xray_dataset_length)

    # Check the length of the training dataset
    train_dataset_length = len(train_dataset)
    print("Number of images in the training dataset:", train_dataset_length)

    valid_dataset_length = len(valid_dataset)
    print("Number of images in the validation dataset:", valid_dataset_length)

    # Check the length of the test dataset
    test_dataset_length = len(test_dataset)
    print("Number of images in the test dataset:", test_dataset_length)

    ''' CREATE THE TRANSFORMED DATASET FOR EACH STAGE '''

    train_dataset_128 = copy.deepcopy(train_dataset)
    valid_dataset_128 = copy.deepcopy(valid_dataset)
    test_dataset_128 = copy.deepcopy(test_dataset)

    train_dataset_128.dataset.transform = transform_128_training
    valid_dataset_128.dataset.transform = transform_128
    test_dataset_128.dataset.transform = transform_128

    train_dataset_224 = copy.deepcopy(train_dataset)
    valid_dataset_224 = copy.deepcopy(valid_dataset)
    test_dataset_224 = copy.deepcopy(test_dataset)

    train_dataset_224.dataset.transform = transform_224_training
    valid_dataset_224.dataset.transform = transform_224
    test_dataset_224.dataset.transform = transform_224

    train_dataset_229 = copy.deepcopy(train_dataset)
    valid_dataset_229 = copy.deepcopy(valid_dataset)
    test_dataset_229 = copy.deepcopy(test_dataset)

    train_dataset_229.dataset.transform = transform_229_training
    valid_dataset_229.dataset.transform = transform_229
    test_dataset_229.dataset.transform = transform_229

    batch_size = 6

    ''' CREATE THE DATALOADERS '''

    train_loader_128 = DataLoader(train_dataset_128, batch_size=batch_size, shuffle=True)
    valid_loader_128 = DataLoader(valid_dataset_128, batch_size=batch_size, shuffle=True)
    test_loader_128 = DataLoader(test_dataset_128, batch_size=batch_size, shuffle=False)

    train_loader_224 = DataLoader(train_dataset_224, batch_size=batch_size, shuffle=True)
    valid_loader_224 = DataLoader(valid_dataset_224, batch_size=batch_size, shuffle=True)
    test_loader_224 = DataLoader(test_dataset_224, batch_size=batch_size, shuffle=False)

    train_loader_229 = DataLoader(train_dataset_229, batch_size=batch_size, shuffle=True)
    valid_loader_229 = DataLoader(valid_dataset_229, batch_size=batch_size, shuffle=True)
    test_loader_229 = DataLoader(test_dataset_229, batch_size=batch_size, shuffle=False)

    # Explore train dataset
    print("Train Dataset:")
    for images, labels in train_loader_128:
        visualize_batch(images, labels, chest_xray_dataset.class_names)
        break  # Show only the first batch for demonstration

    # Explore test dataset
    print("Test Dataset:")
    for images, labels in test_loader_128:
        visualize_batch(images, labels, chest_xray_dataset.class_names)
        break  # Show only the first batch for demonstration

    class_names = ['NORMAL', 'Viral Pneumonia', 'COVID-19']
    images, labels = next(iter(train_loader_128))
    show_images(images, labels, labels,class_names)

    for images, labels in train_loader_128:
        print("Image batch shape:", images.shape)
        print("Label batch shape:", labels.shape)
        print("Example image tensor:")
        print(images[0])  # Print the first image tensor in the batch
        print("Example label:")
        print(labels[0])  # Print the first label in the batch
        break  # Stop after printing the first batch for demonstration purposes

    # Set random seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    '''RUNNING STAGE 1'''

    # Set number of epochs
    NUM_EPOCHS = 3

    pretrained_resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    for params in pretrained_resnet.parameters():
        params.requires_grad = False
    pretrained_resnet.fc = nn.Linear(2048, 3)
    pretrained_resnet.to(device)

    # Setup loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=pretrained_resnet.parameters(), lr=1e-3)

    # Start the timer
    from timeit import default_timer as timer
    start_time = timer()

    # Train model_0
    model_results_stage1_step1 = train(device=device,
                                       model=pretrained_resnet,
                                       train_dataloader=train_loader_128,
                                       test_dataloader=valid_loader_128,
                                       optimizer=optimizer,
                                       loss_fn=loss_fn,
                                       epochs=NUM_EPOCHS)

    # Set number of epochs
    NUM_EPOCHS = 5

    for params in pretrained_resnet.parameters():
        params.requires_grad = True

    # Setup loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=pretrained_resnet.parameters(), lr=1e-3)

    model_results_stage1_step2 = train(device=device,
                                       model=pretrained_resnet,
                                       train_dataloader=train_loader_128,
                                       test_dataloader=valid_loader_128,
                                       optimizer=optimizer,
                                       loss_fn=loss_fn,
                                       epochs=NUM_EPOCHS)

    # End the timer and print out how long it took
    end_time = timer()
    print(f"Total training time: {end_time - start_time:.3f} seconds")

    '''RUNNING STAGE 2'''

    # Turn off training on the network and leave only the classifier
    for name, params in pretrained_resnet.named_parameters():
        if name == 'fc.weight' or name == 'fc.bias': 
            #print('found')
            params.requires_grad = True
        else:
            params.requires_grad = False
        print(name, params.requires_grad)

    # Setup loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=pretrained_resnet.parameters(), lr=1e-4)

    # Start the timer
    from timeit import default_timer as timer
    start_time = timer()

    NUM_EPOCHS = 3

    # Train model_0
    model_results_stage2_step1 = train(device=device,
                                       model=pretrained_resnet,
                                       train_dataloader=train_loader_224,
                                       test_dataloader=valid_loader_224,
                                       optimizer=optimizer,
                                       loss_fn=loss_fn,
                                       epochs=NUM_EPOCHS)

    # Set number of epochs
    NUM_EPOCHS = 5

    for params in pretrained_resnet.parameters():
        params.requires_grad = True

    # Setup loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=pretrained_resnet.parameters(), lr=1e-4)

    model_results_stage2_step2 = train(device=device,
                                       model=pretrained_resnet,
                                       train_dataloader=train_loader_224,
                                       test_dataloader=valid_loader_224,
                                       optimizer=optimizer,
                                       loss_fn=loss_fn,
                                       epochs=NUM_EPOCHS)

    # End the timer and print out how long it took
    end_time = timer()
    print(f"Total training time: {end_time - start_time:.3f} seconds")

    '''RUNNING STAGE 3'''

    # Set number of epochs
    NUM_EPOCHS = 25

    for params in pretrained_resnet.parameters():
        params.requires_grad = True

    # Setup loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=pretrained_resnet.parameters(), lr=1e-6)

    start_time = timer()

    model_results_stage3 = train(device=device,
                                 model=pretrained_resnet,
                                 train_dataloader=train_loader_229,
                                 test_dataloader=valid_loader_229,
                                 optimizer=optimizer,
                                 loss_fn=loss_fn,
                                 epochs=NUM_EPOCHS)

    # End the timer and print out how long it took
    end_time = timer()
    print(f"Total training time: {end_time - start_time:.3f} seconds")

    torch.save(pretrained_resnet.state_dict(), 'pretrained_resnet_model.pt')

    test_acc = evaluate_model(pretrained_resnet, test_loader_224, device, class_names)


if __name__ == '__main__':
    main()
