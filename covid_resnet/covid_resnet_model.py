import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import itertools

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from load_data import *

NUM_CLASSES = 3
DENSE_LAYER_ACTIVATION = 'softmax'


def create_image_data_generators(train_df, dummy_df, valid_df, test_df):
    training_gen = ImageDataGenerator(rescale=1. / 255, rotation_range=15, vertical_flip=True)

    train_128_gen = training_gen.flow_from_dataframe(train_df, x_col='path', y_col='Classification',
                                                     target_size=(128, 128), class_mode='categorical', color_mode='rgb',
                                                     shuffle=True, batch_size=32)

    train_224_gen = training_gen.flow_from_dataframe(train_df, x_col='path', y_col='Classification',
                                                     target_size=(224, 224), class_mode='categorical', color_mode='rgb',
                                                     shuffle=True, batch_size=32)

    train_229_gen = training_gen.flow_from_dataframe(train_df, x_col='path', y_col='Classification',
                                                     target_size=(229, 229), class_mode='categorical', color_mode='rgb',
                                                     shuffle=True, batch_size=32)

    test_generator = ImageDataGenerator(rescale=1. / 255)

    test_gen = test_generator.flow_from_dataframe(dummy_df, x_col='path', y_col='Classification',
                                                  target_size=(128, 128), class_mode='categorical', color_mode='rgb',
                                                  shuffle=False, batch_size=32)

    return train_128_gen, train_224_gen, train_229_gen, test_gen


def first_keras_model():
    model1 = Sequential()

    # 1st layer as the lump sum weights from resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
    # NOTE that this layer will be set below as NOT TRAINABLE, i.e., use it as is
    model1.add(ResNet50(include_top=False, input_shape=(128, 128, 3), pooling='avg', weights='imagenet'))
    model1.add(BatchNormalization())

    # 2nd layer as Dense for 2-class classification, i.e., dog or cat using SoftMax activation
    model1.add(Dense(NUM_CLASSES, activation=DENSE_LAYER_ACTIVATION))

    # Say not to train first layer (ResNet) model as it is already trained

    return model1


def runStage1(xtrain128):
    # TRAINING ON 128*128 WITH FREEZED RESNET LAYER
    model1 = first_keras_model()
    model = model1
    model.layers[0].trainable = False
    print(model.summary())
    opt = tf.keras.optimizers.Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    hist = model.fit(xtrain128, batch_size=32, epochs=3, verbose=1)
    model.save('model1_freezed.h5')

    # FINE TUNING ON 128*128 WITH UNFREEZED RESNET LAYER
    model.layers[0].trainable = True
    print(model.summary())
    opt = tf.keras.optimizers.Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    hist = model.fit(xtrain128, batch_size=32, epochs=5, verbose=1)
    model.save('model1_unfreezed.h5')


def Second_keras_model():
    model2 = Sequential()

    # 1st layer as the lump sum weights from resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
    # NOTE that this layer will be set below as NOT TRAINABLE, i.e., use it as is
    model2.add(ResNet50(include_top=False, input_shape=(224, 224, 3), pooling='avg', weights='imagenet'))
    model2.add(BatchNormalization())

    # 2nd layer as Dense for 2-class classification, i.e., dog or cat using SoftMax activation
    model2.add(Dense(NUM_CLASSES, activation=DENSE_LAYER_ACTIVATION))
    model2.load_weights("model1_unfreezed.h5")

    # Say not to train first layer (ResNet) model as it is already trained

    return model2


def runStage2(xtrain224):
    model2 = Second_keras_model()
    model2.layers[0].trainable = False
    print(model2.summary())
    opt = tf.keras.optimizers.Adam(lr=0.0001)
    model2.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    hist = model2.fit(xtrain224, batch_size=32, epochs=3, verbose=1)
    model2.save('model2_freezed.h5')

    # FINE TUNING ON 224 WITH UNFREEZED RESNET LAYER
    model2 = model2
    model2.layers[0].trainable = True
    print(model2.summary())
    opt = tf.keras.optimizers.Adam(lr=0.0001)
    model2.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    hist = model2.fit(xtrain224, batch_size=32, epochs=5, verbose=1)
    model2.save('model2_unfreezed.h5')


def Third_keras_model():
    model3 = Sequential()

    # 1st layer as the lump sum weights from resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
    # NOTE that this layer will be set below as NOT TRAINABLE, i.e., use it as is
    model3.add(ResNet50(include_top=False, input_shape=(229, 229, 3), pooling='avg', weights='imagenet'))
    model3.add(BatchNormalization())

    # 2nd layer as Dense for 2-class classification, i.e., dog or cat using SoftMax activation
    model3.add(Dense(NUM_CLASSES, activation=DENSE_LAYER_ACTIVATION))
    model3.load_weights('model2_unfreezed.h5')

    # Say not to train first layer (ResNet) model as it is already trained

    return model3


def runStage3(xtrain229):
    # TRAINING ON 229*229 WITH FREEZED RESNET LAYER
    model3 = Third_keras_model()
    # model3=model3(training=False)
    model3.layers[0].trainable = True
    print(model3.summary())
    opt = tf.keras.optimizers.Adam(lr=0.00001)
    model3.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    hist = model3.fit(xtrain229, batch_size=32, epochs=25, verbose=1)
    model3.save('Final_model.h5')
    return model3


def get_scores(model3, test_gen):
    scores = model3.evaluate(test_gen)
    print(scores)
    print("%s%s: %.2f%%" % ("evaluate ", model3.metrics_names[1], scores[1] * 100))


def get_predictions(model3, test_gen):
    predictions = model3.predict(test_gen)
    # print(predictions)
    predicted_class = np.argmax(predictions, axis=1)
    l = dict((v, k) for k, v in test_gen.class_indices.items())
    prednames = [l[k] for k in predicted_class]
    filenames = test_gen.filenames
    image_names = []

    for file in filenames:
        output = file.split("\\")[-1]
        image = output.split('/')[-1]
        image_names.append(image)

    final_df = pd.DataFrame({'Filename': image_names, 'Prediction': prednames})

    print(final_df)
    return predictions


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    """This function plot confusion matrix method from sklearn package."""

    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Normalized Confusion Matrix')

    else:
        print('Confusion Matrix, Without Normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment='center', color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')


def get_confusion_matrix(predictions, test_gen):
    y_pred = np.argmax(predictions, axis=1)
    # print(y_pred)
    g_dict = test_gen.class_indices
    classes = list(g_dict.keys())
    cm = confusion_matrix(test_gen.classes, y_pred)
    print(cm)
    plot_confusion_matrix(cm=cm, classes=classes, title='Confusion Matrix')
    print(classification_report(test_gen.classes, y_pred, target_names=classes))


def get_metrics(predictions, test_gen):
    accuracy_score(test_gen.classes, predictions)
    matthews_corrcoef(test_gen.classes, predictions)
    classes = ['COVID-19', 'NORMAL', 'Viral Pneumonia']
    print(classification_report(test_gen.classes, predictions, target_names=classes))


def main():
    data = prep_data()
    train_df, dummy_df, valid_df, test_df = split_data(data)
    train_128_gen, train_224_gen, train_229_gen, test_gen = create_image_data_generators(train_df, dummy_df,
                                                                                         valid_df, test_df)

    runStage1(train_128_gen)
    runStage2(train_224_gen)
    model = runStage3(train_229_gen)

    get_scores(model, test_gen)
    predictions = get_predictions(model, test_gen)
    get_confusion_matrix(predictions, test_gen)
    get_metrics(predictions, test_gen)


if __name__ == "__main__":
    main()
