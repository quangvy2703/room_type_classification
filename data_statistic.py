import os
from matplotlib import pyplot as plt

def statistics(path="/disk_local/vypham/dataset/room_type/aug_images/train"):
    labels = os.listdir(path)

    categories = {}
    for label in labels:
        n_samples = len(os.listdir(os.path.join(path, label)))
        categories[label] = n_samples
    

    sorted_keys = sorted(categories, key=categories.get)
    values = [categories[key] for key in sorted_keys]


    plt.barh(sorted_keys, values)

    for index, value in enumerate(values):
        plt.text(value, index, str(value))

    plt.ylabel("Label name")
    plt.xlabel("No. of bboxes")
    # plt.show()
    plt.savefig('status.png')

statistics()