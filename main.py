import os
import random
import time
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def main():
    current_dir = os.getcwd()
    path_to_pictures = f"{current_dir}/notMNIST_large"
    if not check_class_balance(path=path_to_pictures):
        print("Classes are not balanced!")
        return

    data = load_images_and_labels(path=path_to_pictures)

    show_images(num_img_to_display=3, data=data)

    train_samples, test_samples, validation_samples = prepare_sampling(data)

    model_obj = LogisticRegression
    accuracies = []
    sample_sizes = [50, 100, 1000, 10_000, 50_000]
    for sample_size in sample_sizes:
        model = train_model(model_obj, *train_samples, sample_size)
        accuracy_score = get_accuracy_score(model, *test_samples)
        accuracies.append(accuracy_score)

    create_plot_accuracy_on_train_size(sample_sizes=sample_sizes, accuracies=accuracies)



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



if __name__ == "__main__":
    start_time = time.time()
    print("Started")
    main()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Working time: {execution_time} seconds.")
