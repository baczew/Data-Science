# ### Import packages


import os
import urllib.request 
import gzip
import numpy as np
from collections import Counter
import time
import multiprocessing
import concurrent.futures
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import random



# ### Specify files needed for modeling

def main():
    file_names = {"train_images": "train-images-idx3-ubyte.gz",
                "train_labels": "train-labels-idx1-ubyte.gz",
                "test_images": "t10k-images-idx3-ubyte.gz",
                "test_labels": "t10k-labels-idx1-ubyte.gz"}

    download_files(file_names)
    train_labels, test_labels = load_labels(file_names)
    train_images, test_images = load_images(file_names)

    #verify_labels(train_labels)
    #verify_labels(test_labels)
    #verify_image(train_images)
    #verify_image(test_images)
    
    n = 10_000 #full test set
    
    train_repr_list = []
    train_repr_label = []

    start_time = time.time()

    for label in set(train_labels):
        reps, labels = get_representants(label, train_images, train_labels)
        train_repr_list.extend(reps)
        train_repr_label.extend(labels)
    print(f"--- Clustering took {(time.time() - start_time)/60} minutes ({time.time() - start_time} seconds) ---")

    result = run(test_images[:n], train_repr_list, train_repr_label)
    print(f"--- Whole process took {(time.time() - start_time)/60} minutes ({time.time() - start_time} seconds) ---")

    acc = np.sum(np.array(result) == test_labels[:n])/n
    print(f"Accuracy: {acc}")


def get_representants(label, image_set, label_set):

    n_clusters = 25
    take = 10

    representants = []
    images_with_indicated_label = [img.reshape(28*28,) for img, leb in zip(image_set, label_set) if leb == label]
    kmeans = KMeans(init="random", n_clusters=n_clusters, n_init=1, max_iter=300, random_state=123)
    kmeans.fit(images_with_indicated_label)

    for i in set(kmeans.labels_):
        
        class_repr = [image for image, km_label in zip(images_with_indicated_label, kmeans.labels_) if km_label == i]

        if len(class_repr) > take:
            sub_kmeans = KMeans(init="random", n_clusters=take, n_init=1, max_iter=300, random_state=321)
            sub_kmeans.fit(class_repr)

            for j in set(sub_kmeans.labels_):
                
                class_sub_repr = [image.reshape(28, 28) for image, km_label in zip(class_repr, sub_kmeans.labels_) if km_label == j]
                pick = random.sample(class_sub_repr, 1)
                representants.extend(pick)

        else:
            representants.extend([i.reshape(28, 28) for i in class_repr])
    
    return representants, [label] * len(representants)


def download_files(file_names):

    url = "http://yann.lecun.com/exdb/mnist/"

    for name, file in file_names.items():
        if file in os.listdir():
            continue
        else:
            urllib.request.urlretrieve(url + file, file)

    check = set(file_names.values()).issubset(set(os.listdir()))

    assert check == True


def read_labels(file_name):
    
    with gzip.open(file_name, "rb") as f:
        labels = f.read()
        f.close()
    
    return [int(label) for label in labels][8:]


def load_labels(file_names):

    train_labels = read_labels(file_names['train_labels'])
    test_labels = read_labels(file_names['test_labels'])
    return train_labels, test_labels


def read_images(file_name):
    
    with gzip.open(file_name, "rb") as f:
        img = bytearray(f.read())[16:]
        images = np.reshape(np.array(img), (-1, 28, 28))
        images = images/255
        #print(images.shape)
        f.close()
        
    return images


def load_images(file_names):

    train_images = read_images(file_names['train_images'])
    test_images = read_images(file_names['test_images'])
    return train_images, test_images


def verify_labels(labels):

    print(dict(Counter(labels)))


def verify_image(images):

    print(images[0].shape)
    plt.imshow(images[0])
    plt.show()


def predict_1nn(img, train_images, train_labels):

    cur_distance = [(np.linalg.norm(img - train_image), train_image_index) for train_image_index, train_image in enumerate(train_images)]
    _, best_image_index = min(cur_distance)
    
    return train_labels[best_image_index]


def predict_1nn_list(images, train_images, train_labels):
    return [predict_1nn(image, train_images, train_labels) for image in images]


def run(images, train_images, train_labels):

    image_count = images.shape[0]
    batch = int(image_count / 4)

    with concurrent.futures.ProcessPoolExecutor() as executor:

        processes = [executor.submit(predict_1nn_list, *[images[offset : offset + batch], train_images, train_labels]) for offset in range(0, image_count, batch)]
        concurrent.futures.wait(processes)
        result_list = [f.result() for f in processes]
        results = [item for sublist in result_list for item in sublist]
    
    return results
        

if __name__ == '__main__':
    main()