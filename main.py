import os
import random
import time
import numpy as np
import matplotlib.pyplot as plt
import itertools
import tensorflow as tf

from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import LearningRateScheduler


def main():
    lab_num = None
    while lab_num not in [1, 2, 3]:
        print("Выбирете номер лабораторной работы:")
        print("1 - Лабораторная №1 'Простой классификатор' 1 семестр.")
        print("2 - Лабораторная №2 'Полносвязная Нейронная Сеть' 2 семестр.")
        print("2 - Лабораторная №3 'Свёрточная Нейронная Сеть' 2 семестр.")
        lab_num = input("Номер: ")
        lab_num = int(lab_num)
    
    current_dir = os.getcwd()
    path_to_pictures = f"{current_dir}/notMNIST_large"
    if not check_class_balance(path=path_to_pictures):
        print("Classes are not balanced!")
        return

    data = load_images_and_labels(path=path_to_pictures)

    show_images(num_img_to_display=3, data=data)

    train_samples, test_samples, validation_samples = prepare_sampling(data)
    if lab_num == 1:
        model_obj = LogisticRegression
        accuracies = []
        sample_sizes = [50, 100, 1000, 10_000, 50_000]
        for sample_size in sample_sizes:
            model = train_model(model_obj, *train_samples, sample_size)
            accuracy_score = get_accuracy_score(model, *test_samples)
            accuracies.append(accuracy_score)
        create_plot_accuracy_on_train_size(sample_sizes=sample_sizes, accuracies=accuracies)
    elif lab_num == 2:
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(train_samples[1])
        y_test_encoded = label_encoder.transform(test_samples[1])
        y_validation_encoded = label_encoder.transform(validation_samples[1])

        train_samples = (train_samples[0], y_train_encoded)
        test_samples = (test_samples[0], y_test_encoded)
        validation_samples = (validation_samples[0], y_validation_encoded)
        # необходимо преобразовать данные в нужную форму
        train_samples = (train_samples[0].reshape(-1, 28, 28), train_samples[1])
        test_samples = (test_samples[0].reshape(-1, 28, 28), test_samples[1])
        validation_samples = (validation_samples[0].reshape(-1, 28, 28), validation_samples[1])
        accuracies = []
        lr_callback = LearningRateScheduler(lr_scheduler)
        with open("reports/results.txt", 'w') as file:
            for hidden_layers, neurons_per_layer, activation in generate_parameters_for_model_training():
                print(f"Training model: hidden_layers={hidden_layers}, neurons_per_layer={neurons_per_layer}, activation={activation}")
                model = create_model(hidden_layers, neurons_per_layer, activation)

                #train_model_fully_connected_nn(model, *train_samples, optimizer="sgd")
                # для случае регулировки скорости обучения:
                train_model_fully_connected_nn(model, *train_samples, optimizer="sgd", callbacks=[lr_callback])
                accuracy = evaluate_model(model, *test_samples)
                file.write(f"hidden_layers={hidden_layers}, neurons_per_layer={neurons_per_layer}, activation={activation}, accuracy={accuracy}\n")
                accuracies.append(accuracy)

        print(accuracies)
    elif lab_num == 3:
        train_samples, test_samples, validation_samples = prepare_sampling(data)
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(train_samples[1])
        y_test_encoded = label_encoder.transform(test_samples[1])
        y_validation_encoded = label_encoder.transform(validation_samples[1])

        train_samples = (train_samples[0].reshape(-1, 28, 28, 1), y_train_encoded)
        test_samples = (test_samples[0].reshape(-1, 28, 28, 1), y_test_encoded)
        validation_samples = (validation_samples[0].reshape(-1, 28, 28, 1), y_validation_encoded)

        #model = create_cnn_model()
        #model = create_cnn_model_with_pooling()
        model = create_lenet5_model()
        with open("reports/cnn_results.txt", 'w') as file:
            #train_cnn_model(model, *train_samples, optimizer="sgd")
            train_lenet5_model(model, *train_samples, optimizer="sgd")
            accuracy = evaluate_model(model, *test_samples)
            print(f"Accuracy of the CNN model: {accuracy}")
            file.write(f"Accuracy of the CNN model: {accuracy}")


def check_class_balance(path):
    classes_subdirs = [subdir_path for subdir_path in os.listdir(path) if os.path.isdir(os.path.join(path, subdir_path))]
    num_of_files_in_dirs = {}
    for dir_name in classes_subdirs:
        path_to_files = os.path.join(path, dir_name)
        files_list = [file for file in os.listdir(path_to_files) if os.path.isfile(os.path.join(path_to_files, file))] 
        num_of_images_in_dir = len(files_list)
        num_of_files_in_dirs[dir_name] = num_of_images_in_dir

    avarage_num_of_files = sum(num_of_files_in_dirs.values()) / len(num_of_files_in_dirs)

    limit = 10

    for dir_name, num_files in num_of_files_in_dirs.items():
        if abs(num_files - avarage_num_of_files) > limit:
            return False
        return True


def load_images_and_labels(path):
    data = {"images": [], "labels": []}

    for label in os.listdir(path):
        if label.startswith("."):
            continue
        dir_with_labels = os.path.join(path, label)

        for file in os.listdir(dir_with_labels):
            if file.startswith("."):
                continue
            path_to_image = os.path.join(dir_with_labels, file)
            image = Image.open(path_to_image)
            image = image.resize((28, 28))
            image = np.array(image)

            data["images"].append(image)
            data["labels"].append(label)

    data["images"] = np.array(data["images"])
    data["labels"] = np.array(data["labels"])
    return data


def prepare_sampling(data):
    total_train_samples = 200_000
    total_validation_samples = 10_000
    total_test_samples = 19_000

    permutation = np.random.permutation(len(data["images"]))
    shuffled_images = data["images"][permutation]
    shuffled_labels = data["labels"][permutation]

    x_train, x_temp, y_train, y_temp = train_test_split(shuffled_images, shuffled_labels, test_size=(1 - total_train_samples / len(data["images"])), random_state=33, stratify=shuffled_labels)

    x_validation, x_test, y_validation, y_test = train_test_split(x_temp, y_temp, test_size=(total_test_samples / (total_validation_samples + total_test_samples)), random_state=33, stratify=y_temp)

    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)
    x_validation = x_validation.reshape(x_validation.shape[0], -1)

    return ((x_train, y_train), (x_test, y_test), (x_validation, y_validation))



def train_model(model_class, x_train, y_train, sample_size):
    model = model_class()
    model.fit(x_train[:sample_size], y_train[:sample_size])
    return model


def get_accuracy_score(model, x_test, y_test):
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy of the model {model.__class__.__name__}: {accuracy}")
    return accuracy


def create_plot_accuracy_on_train_size(sample_sizes, accuracies):
    plt.plot(sample_sizes, accuracies, marker="o")
    plt.title("Dependence of accuracy on training sample size")
    plt.xlabel("Training sample size")
    plt.ylabel("Accuracy")
    plt.show()


def show_images(num_img_to_display, data):
    selected_indices = random.sample(range(len(data["images"])), num_img_to_display)
    selected_images = data["images"][selected_indices]
    selected_labels = data["labels"][selected_indices]
    for i in range(num_img_to_display):
        plt.subplot(1, num_img_to_display, i + 1)
        plt.imshow(selected_images[i], cmap='gray')
        plt.title(selected_labels[i])
        plt.axis('off')

    plt.show()


def train_model_fully_connected_nn(model, train_images, train_labels, optimizer, callbacks=None):
    model.compile(
        optimizer=optimizer, 
        loss="sparse_categorical_crossentropy", 
        metrics=["accuracy"]
        )
    model.fit(train_images, train_labels, epochs=15, batch_size=64, validation_split=0.2, callbacks=callbacks)


def generate_parameters_for_model_training():
    hidden_layers_list = [1, 2, 3, 4, 5]
    neurons_per_layer_list = [64, 128, 256, 512]
    activations = ["relu", "sigmoid", "tanh"]

    # generate combinations of hidden layers, neurons per layer, activation functions
    param_combinations = itertools.product(hidden_layers_list, neurons_per_layer_list, activations)
    return param_combinations


# def create_model(hidden_layers, neurons_per_layer, activation):
#     model = models.Sequential()
#     model.add(layers.Flatten(input_shape=(28, 28)))
#     for _ in range(hidden_layers):
#         model.add(layers.Dense(neurons_per_layer, activation=activation))
#     model.add(layers.Dense(10, activation='softmax'))
#     return model

def create_model(hidden_layers, neurons_per_layer, activation):
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=(28, 28)))
    for _ in range(hidden_layers):
        model.add(
            layers.Dense(
                neurons_per_layer, 
                activation=activation, 
                kernel_regularizer=regularizers.l2(0.001), 
                bias_regularizer=regularizers.l2(0.001)
                )
            )  # регуляризацию L2 коэффициентом 0.001 к каждому полносвязанному слою 
        model.add(layers.Dropout(0.5))  # слой сброса нейронов dropout с коэффициентом 0.5
    model.add(layers.Dense(10, activation="softmax"))
    return model


def create_cnn_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(10, activation="softmax"))
    return model


def create_cnn_model_with_pooling(pooling_type="max"):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)))
    if pooling_type == "max":
        model.add(layers.MaxPooling2D((2, 2)))
    elif pooling_type == "average":
        model.add(layers.AveragePooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    if pooling_type == "max":
        model.add(layers.MaxPooling2D((2, 2)))
    elif pooling_type == "average":
        model.add(layers.AveragePooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(10, activation="softmax"))
    return model


def create_lenet5_model():
    model = models.Sequential()

    # Сверточные слои
    model.add(layers.Conv2D(6, kernel_size=(5, 5), activation="relu", input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(layers.Conv2D(16, kernel_size=(5, 5), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))

    # Полносвязанные слои
    model.add(layers.Flatten())
    model.add(layers.Dense(120, activation="relu"))
    model.add(layers.Dense(84, activation="relu"))
    model.add(layers.Dense(10, activation="softmax"))

    return model


def train_cnn_model(model, train_images, train_labels, optimizer, callbacks=None):
    model.compile(
        optimizer=optimizer, 
        loss="sparse_categorical_crossentropy", 
        metrics=["accuracy"]
        )
    model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.2, callbacks=callbacks)


def train_lenet5_model(model, train_images, train_labels, optimizer, callbacks=None):
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.2, callbacks=callbacks)


def evaluate_model(model, test_images, test_labels):
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print("Test accuracy:", test_acc)
    return test_acc


def lr_scheduler(epoch):
    """Функция динамически изменяющая скорость обучения"""
    if epoch < 10:
        return 0.01
    elif epoch < 20:
        return 0.001
    else:
        return 0.0001


if __name__ == "__main__":
    start_time = time.time()
    print("Started")
    main()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Working time: {execution_time} seconds.")
