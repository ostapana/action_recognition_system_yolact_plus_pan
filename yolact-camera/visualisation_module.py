import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def dist(x0, x1, y0, y1):
    return np.sqrt((x0-x1)**2 + (y0-y1)**2)


class Visualiser:
    def __init__(self, helper):
        self.filt_images_path = "testset/dataset.txt"
        self.helper = helper
        self.num_to_yol_object = {0: 'kuka', 1: 'car_roof', 2: 'cube_holes', 3: 'ex_bucket', 4: 'hammer', 5: 'nut',
                                  6: 'peg_screw', 7: 'pliers', 8: 'screw_round', 9:'screwdriver',
                                  10: 'sphere_holes', 11: 'wafer', 12: 'wheel', 13: 'wrench'}

    def save_histogram(self, data, title, stat='count', bins=20):
        plt.figure()
        sp = sns.histplot(data=data, bins=bins)
        sp.set_title(title)
        sp.figure.savefig('visualisation/plot_' + title)


    def get_filtered_images(self):
        with open(self.filt_images_path) as f:
            content = f.readlines()
        return [n.split(' ')[0] for n in content]

    def create_moving_histogram(self, annotation, images, folder, title, stat='count', one_object=None, bins=10):
        move_distance = []
        for key in images:
            try:
                objects = annotation[folder + key]
                if one_object:
                    o_keys = [one_object]
                else:
                    o_keys = objects.keys()
                for key2 in o_keys:
                    for key3 in annotation[folder + key][key2].keys():

                        i_object = annotation[folder + key][key2][key3]
                        distance = i_object[7] / i_object[6]
                        if distance < 10:
                            move_distance.append(distance)
            except KeyError:
                pass
        self.save_histogram(move_distance, title, stat=stat, bins=bins)

    def create_obj_histogram(self, annotation, images, folder, title):
        object_classes = {key:0 for key in self.num_to_yol_object.values() if key not in ["kuka", 'car_roof']}
        for key in images:
            try:
                objects = annotation[folder + key]
                o_keys = objects.keys()

                for key2 in o_keys:
                    object_classes[key2] += len(objects[key2].keys())

            except KeyError:
                pass

        names = list(object_classes.keys())
        values = list(object_classes.values())
        fig = plt.figure(figsize=(15, 6))
        plt.bar(range(len(object_classes)), values, tick_label=names, color="#88c0d7")
        plt.savefig('visualisation/plot_hist_' + title)

    def save_box_plot(self, move_distances, title):
        fig = plt.figure(figsize=(14, 8))
        labels, data = move_distances.keys(), move_distances.values()
        fig = plt.figure(figsize=(15, 7))
        ax = fig.add_subplot(111)
        bp = ax.boxplot(data, patch_artist=True)

        for patch in bp['boxes']:
            patch.set_facecolor("#88c0d7")

        for median in bp['medians']:
            median.set(color='red', linewidth=2)


        plt.xticks(range(1, len(labels) + 1), labels)
        plt.savefig('visualisation/plot_box_' + title)

    def get_moved_distances(self, annotation, images, folder):
        classes = self.num_to_yol_object.values()
        move_distances = {key: [] for key in classes if key not in ['kuka', 'car_roof']}
        for key in images:
            try:
                objects = annotation[folder + key]
                for key2 in objects.keys():
                    for key3 in objects[key2].keys():  # can we just not talk about it
                        i_object = objects[key2][key3]
                        distance  = i_object[7]/i_object[6]
                        if distance < 2:
                            continue
                        elif distance > 100:
                            print(key, distance)
                        if key2 in move_distances.keys():
                            move_distances[key2].append(distance)
            except KeyError:
                pass
        return move_distances

    def create_boxes_moving_hist(self, annotation, images, folder, title):
        move_distances = self.get_moved_distances(annotation, images, folder)
        self.save_box_plot(move_distances, title)


    def get_categories(self):
        with open('data/categories/categories8.txt') as f:
            categories = f.readlines()
        return [f.strip() for f in categories]

    # box plot for one particular action recognised by pan
    def create_boxes_moving_one_action(self, annotation, images, folder, title, req_action):
        classes = self.num_to_yol_object.values()
        categories = self.get_categories()
        move_distances = {key: [] for key in classes if key not in ['kuka', 'car_roof']}
        with open("testset/dataset.txt") as f:
            for line in f:
                img = line.split(" ")[0]
                index = int(line.split(" ")[2])
                true_action = categories[index-1]
                if true_action == req_action:
                    objects = annotation[folder + img]
                    for key2 in objects.keys():
                        for key3 in objects[key2].keys():
                            i_object = objects[key2][key3]
                            distance = i_object[7]/i_object[6]
                            if distance > 100:
                                print(distance, img)
                            if key2 in move_distances.keys():
                                move_distances[key2].append(distance)

        self.save_box_plot(move_distances, title)


