# Import necessary libraries
from glob import glob  # For file path pattern matching
import cv2  # OpenCV library for image processing
import numpy as np  # For numerical operations
from sklearn.model_selection import train_test_split, cross_validate, KFold  # For data splitting
from keras.applications.densenet import DenseNet169  # Pre-trained model
from xgboost import XGBClassifier  # XGBoost classifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_curve, auc, classification_report, matthews_corrcoef  # Evaluation metrics
from sklearn import preprocessing  # For data preprocessing
import seaborn as sns  # For visualization
import matplotlib.pyplot as plt  # For plotting

def load_data():
    # Data loading
    images_path = {
        "Covid": glob("/content/drive/MyDrive/COVID-19 Radiography Database/COVID-19/*.png"),
        "NoFindings": glob("/content/drive/MyDrive/COVID-19 Radiography Database/NORMAL/*.png"),
        "Pneumonia": glob("/content/drive/MyDrive/COVID-19 Radiography Database/Viral Pneumonia/*.png")
    }

    images_class = {
        "Covid": 0,
        "Pneumonia": 1,
        "NoFindings": 2
    }

    X = []
    Y = []

    # Load images and corresponding labels
    for label in images_path:
        for image_path in images_path[label]:
            image = cv2.imread(image_path)
            image = cv2.resize(image, (224, 224)) # Resize image to (224, 224)
            X.append(image)
            Y.append(images_class[label])

    return np.array(X), np.array(Y)

def preprocess_data(X):
    # Preprocessing of data (if needed)
    return X

def build_model(input_shape):
    # Build model
    pre_trained_models = {}
    pre_trained_models["DenseNet169"] = DenseNet169(include_top=False, input_shape=input_shape, pooling="avg")
    return pre_trained_models["DenseNet169"]

def train_evaluate_model(X_train, X_test, Y_train, Y_test, pre_trained_model):
    # Extract features
    X_train_features = pre_trained_model.predict(X_train)
    X_test_features = pre_trained_model.predict(X_test)

    # Initialize and train XGBoost Classifier
    XGB_Classifier = XGBClassifier(learning_rate=0.44, n_estimators=100, random_state=0, seed=0, gamma=0)
    XGB_Classifier.fit(X_train_features, Y_train)

    # Evaluate model
    Y_pred = XGB_Classifier.predict(X_test_features)

    return XGB_Classifier, Y_pred

def plot_roc_curve(Y_test, Y_pred):
    # Convert labels to binary format for ROC curve
    y_test_bin = preprocessing.label_binarize(Y_test, classes=[0, 1, 2])
    y_pred_bin = preprocessing.label_binarize(Y_pred, classes=[0, 1, 2])

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(3):  # Three classes
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_bin[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_pred_bin.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot multiclass ROC curve
    plt.figure()
    lw = 2
    plt.plot(fpr["micro"], tpr["micro"], color='darkorange',
             lw=lw, label='micro-average ROC curve (AUC = %0.2f)' % roc_auc["micro"])
    for i in range(3):
        plt.plot(fpr[i], tpr[i], lw=lw, label='ROC curve (class %d, AUC = %0.2f)' % (i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multiclass ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

def evaluate_model_performance(Y_test, Y_pred, class_names):
    # Calculate accuracy score
    accuracy = accuracy_score(Y_test, Y_pred)

    # Calculate MCC score
    mcc_score = matthews_corrcoef(Y_test, Y_pred)

    # Compute multiclass AU ROC curve
    n_classes = len(np.unique(Y_test))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve((Y_test == i).astype(int), (Y_pred == i).astype(int))
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Calculate macro-average ROC score
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    roc_auc_macro = auc(all_fpr, mean_tpr)

    # Calculate F1 score
    f1 = f1_score(Y_test, Y_pred, average="macro")

    # Calculate precision score
    precision = precision_score(Y_test, Y_pred, average="macro")

    # Calculate recall score
    recall = recall_score(Y_test, Y_pred, average="macro")

    # Calculate sensitivity and specificity
    cm = confusion_matrix(Y_test, Y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    # Generate classification report
    classification_rep = classification_report(Y_test, Y_pred, target_names=class_names)

    # Generate confusion matrix
    cm = confusion_matrix(Y_test, Y_pred, normalize='true')

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt=".2f", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix (Normalized)')
    plt.show()

    # Print results
    print(f"\n============")
    print(f"\nAccuracy score : {accuracy:.4f}")
    print(f"\nMacro-average ROC Score: {roc_auc_macro:.4f}")
    print(f"\nF1 Score: {f1:.4f}")
    print(f"\nPrecision Score: {precision:.4f}")
    print(f"\nRecall Score: {recall:.4f}")
    print(f"\nSensitivity : {sensitivity:.4f}")
    print(f"\nSpecificity : {specificity:.4f}")
    print("\nClassification Report:")
    print(classification_rep)
    print(f"Matthew's Correlation Coefficient (MCC): {mcc_score:.4f}")


def main():
    # Main function
    X, Y = load_data()
    X = preprocess_data(X)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=80)
    pre_trained_model = build_model(input_shape=(224, 224, 3))
    XGB_Classifier, Y_pred = train_evaluate_model(X_train, X_test, Y_train, Y_test, pre_trained_model)
    plot_roc_curve(Y_test, Y_pred)
    class_names = ["Covid", "Pneumonia", "No Findings"]
    evaluate_model_performance(Y_test, Y_pred, class_names)

if __name__ == "__main__":
    main()
