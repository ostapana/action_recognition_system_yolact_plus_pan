import argparse
import json
import os
import glob
import time
import csv
import pyrealsense2 as rs
import cv2
import numpy as np

from sklearn import preprocessing
from scipy import stats as st

import help_module
from visualisation_module import *
from final_module import *
from prepare_dataset import *


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotate", default=False, action="store_true",
                        help="create video annotation, also use --basic, --move")
    parser.add_argument("--basic", default=False, action="store_true",
                        help="creates basic video annotation")
    parser.add_argument("--move", default=False, action="store_true",
                        help="tracks movements in video")
    parser.add_argument("--path", default="", type=str,
                        help="path for images which should be annotated")
    parser.add_argument("--input_file", default="", type=str,
                        help="path where evaluation should be saved")
    parser.add_argument("--output_file", default="", type=str,
                        help="path where true actions should be saved")
    parser.add_argument("--show", default=False, action="store_true",
                        help="if to show the images during annotation")
    parser.add_argument("--video", default=False, action="store_true",
                        help="annotate one video (info is not saved)")
    parser.add_argument("--evaluate", default=False, action="store_true",
                        help="")
    parser.add_argument("--visualise", default=False, action="store_true",
                        help="creates a histogram/boxing plot")
    parser.add_argument("--prep_dataset", default=False, action="store_true",
                        help="separates data to 2 datasets")

    args = parser.parse_args(argv)
    return args


class MainModule():
    def __init__(self):
        self.classesFile = "classes.json"
        self.helper = help_module.Helper()
        self.annotations = {}
        self.num_to_yol_object = {0: "kuka", 1: "car_roof", 2: "cube_holes", 3: "ex_bucket", 4: "hammer", 5: "nut",
                                  6: "peg_screw", 7: "pliers", 8: "screw_round",
                                  9: "screwdriver", 10: "sphere_holes", 11: "wafer", 12: "wheel", 13: "wrench"}
        self.statistics = {0: 0, 1: 0, 2: 0, 3: 0}
        self.file_name = "annotation.json"

    def frame2image(self, frames):
        align_to = rs.stream.color
        align = rs.align(align_to)
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        return color_image[..., ::-1]

    def runLocallyRS(self, pipe):
        while True:
            frames = pipe.wait_for_frames()
            im = self.frame2image(frames)
            im = np.array(im, np.int16)
            yolact_im = self.helper.applyYolact(im, False)
            cv2.imshow("img_yolact", yolact_im)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.clean("classes.json")
                break

    def get_frame_size(self, frames):
        im = self.frame2image(frames)
        return im.shape[:2]

    def runOnRealSense(self):
        try:
            pipe = rs.pipeline()
            profile = pipe.start()
            self.runLocallyRS(pipe)
        finally:
            pipe.stop()

    def annotateImages(self, path, isShow):
        filelist = glob.glob(os.path.join(path, "*.jpg"))
        for filename in sorted(filelist):
            filename = str(filename)
            img = cv2.imread(filename)
            img = np.array(img, np.int16)
            yolact_im = self.helper.applyYolactImage(img, True,
                                                     path + "/" + self.file_name, filename, self.annotations)
            if isShow:
                cv2.imshow("img_yolact", yolact_im)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                time.sleep(0.3)

    def update_content(self, file, dict_to_update):
        content = self.helper.getData(file)
        content.update(dict_to_update)
        self.helper.dumpData(content, file)

    def update_annotation(self, start_time, index, length, annotation, movements_dict, basic, move):
        print("--- %s seconds ---" % round((time.time() - start_time), 2),
              str(round((index * 100 / length), 2)) + "% done")
        if basic:
            self.update_content("data/annotation.json", annotation)
        if move:
            self.update_content("data/objects_movements.json", movements_dict)
        return {}, {}

    def update_objects_dict(self, objects_dict, o, score):
        if str(o) in objects_dict.keys():
            objects_dict[str(o)][0] += 1
            objects_dict[str(o)][1] += score
        else:
            objects_dict[str(o)] = [1, score]
        return objects_dict

    def update_movements_dict(self, movements_dict, object, centroids, scores):
        if len(centroids) == 1:  # it"s a [[]] dont know why
            centroids = centroids[0]
        x = round(centroids[0], 1)
        y = round(centroids[1], 1)
        object = self.num_to_yol_object[object]
        long_dist = ["hammer", "pliers"]
        mid_dist = ["screw_round", "wafer", "screwdriver"]
        sh_dist = [o for o in self.num_to_yol_object.values() if o not in long_dist and o not in mid_dist]

        if object in movements_dict.keys():
            # we should go through all inner keys and see if a new object is near some of the old objects
            # (so it"s old) or if it is in completely different place it"s new
            found = False
            for key in movements_dict[object].keys():
                old_x = movements_dict[object][key][0]
                old_y = movements_dict[object][key][1]
                moved_dist = dist(old_x, x, old_y, y)
                if (object in long_dist and moved_dist < 90) or \
                        (object in mid_dist and moved_dist < 60) or \
                        (object in sh_dist and moved_dist < 40):  # its old object
                    min_x = min(x, movements_dict[object][key][
                        2])  # we need this to count the maximum moved distance later
                    min_y = min(y, movements_dict[object][key][3])
                    max_x = max(x, movements_dict[object][key][4])
                    max_y = max(y, movements_dict[object][key][5])
                    frames = movements_dict[object][key][6]
                    total_moved_dist = movements_dict[object][key][7]
                    total_scores = movements_dict[object][key][8]
                    movements_dict[object][key] = [x, y, min_x, min_y, max_x, max_y, frames + 1,
                                                   total_moved_dist + moved_dist, total_scores + scores]
                    found = True
                    break
            if not found:
                movements_dict[object][object + str(len(movements_dict[object]))] = [x, y, x, y, x, y, 1, 0, 0]
        else:
            movements_dict[object] = {object + "0": [x, y, x, y, x, y, 1, 0,
                                                     0]}  # {hammer: {"hammer0": [x, y, (minx miny, max, maxy), frames, mean_dis, tot_prob]}}
            movements_dict[object][object + "0"][8] += scores
        return movements_dict

    def annotateVideo(self, name, basic, movements, show=False):
        objects_dict = {}
        movements_dict = {}
        filelist = glob.glob(os.path.join(name, "*.jpg"))
        for filename in filelist:
            filename = str(filename)
            img = cv2.imread(filename)
            img = np.array(img, np.int16)
            objects, object_names, scores, centroids, img = self.helper.applyYolact4Video(img)
            if show:
                cv2.imshow(name, img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            for i in range(0, len(objects)):
                if basic:
                    objects_dict = self.update_objects_dict(objects_dict, objects[i], scores[i])
                if movements:
                    movements_dict = self.update_movements_dict(movements_dict, objects[i], centroids[i], scores[i])
        for k in objects_dict.keys():
            objects_dict[k] = [objects_dict[k][0], objects_dict[k][1] / objects_dict[k][0]]

        return objects_dict, movements_dict

    def create_annotation(self, main_folder, basic=False, move=False):
        annotation = self.helper.getData("data/" + self.file_name)
        objects_movements = self.helper.getData("data/objects_movements.json")
        folders = glob.glob(os.path.join(main_folder, "*"))
        start_time = time.time()
        flag = True
        length = len(folders)
        for i in range(0, length):
            f = folders[i]
            if flag:  # we don"t want to re-evaluate already annotated videos
                if (basic and f not in annotation.keys()) or (move and f not in objects_movements.keys()):
                    flag = False
                    annotation = {}
                    objects_movements = {}
            if not flag:  # cant use else cause the flag could be changed
                video_annotation, movements_annotation = self.annotateVideo(f, basic, move)
                if basic:
                    annotation[f] = video_annotation
                if move:
                    objects_movements[f] = movements_annotation
                if i % 10 == 0 or (i == length - 1):
                    annotation, objects_movements = self.update_annotation(start_time, i, length, annotation,
                                                                           objects_movements, basic, move)

    def get_yolo_action(self, obj_movements):
        obj_to_action = {"hammer": 1, "pliers": 3, "screw_round": 5, "cube_holes": 7}
        needed_objects = obj_to_action.keys()
        info = dict(filter(lambda elem: elem[0] in list(needed_objects), obj_movements.items()))
        if len(info) == 0:
            return (9, 1.0, 0, 0)  # 7 is code for no movement
        else:
            action = -1
            max_prob = 0
            max_distance = -1
            max_mean_dist = -1
            for key in obj_movements.keys():
                if key in obj_to_action.keys():
                    object = obj_movements[key]
                    for key2 in object.keys():
                        object_inside = object[key2]
                        frames_seen = object_inside[6]
                        mean_dist = object_inside[7] / frames_seen
                        if mean_dist > max_mean_dist:
                            max_distance = dist(object_inside[2], object_inside[4], object_inside[3], object_inside[5])
                            action = obj_to_action[key]
                            max_prob = object_inside[8] / frames_seen
                            max_mean_dist = mean_dist
            return (action, round(max_prob, 3), max_distance, max_mean_dist)

    def get_result(self, true_action, pan_action, yol_action):
        if true_action == pan_action and true_action == yol_action:
            return 0
        elif true_action == pan_action and true_action != yol_action:
            return 1
        elif true_action != pan_action and true_action == yol_action:
            return 2
        else:
            return 3

    def evaluate(self, pan_file, dataset, input_file, output_file):
        ismoving_stat = 0

        objects_movements = self.helper.getData("data/objects_movements.json")

        with open(dataset) as f:
            data = f.readlines()

        true_actions = {n.split(" ")[0]: int(n.split(" ")[2]) for n in data}

        with open("testset/banned.txt") as f:
            data = f.readlines()
        banned = np.array([n.split(" ")[0][0:6] for n in data])
        images_opened = 0

        # --------------------------------------------------------------------------

        with open(pan_file, "r") as f1, open(input_file, "w") as f2, open(output_file, "w") as f3:

            header = ["img", "pan_prob", "pan_action", "mean_dist", "yol_prob", "yol_action"]
            writer = csv.writer(f2)
            writer_y = csv.writer(f3)
            writer.writerow(header)

            for line in f1:
                line = line.split(" ")
                img = line[0]
                if img in banned:
                    continue

                tot_frames = int(line[1])
                pan_action = int(line[2][0:1])
                pan_prob = float(line[3][0:5])
                try:
                    obj_movement = objects_movements["testset/rgb/" + img]
                    yol_action, yol_prob, moved_dist, mean_dist = self.get_yolo_action(obj_movement)
                    if mean_dist > 0: ismoving_stat += 1
                    result = self.get_result(true_actions[img], pan_action, yol_action)
                    images_opened += 1
                    self.statistics[result] += 1
                    data = [str(img), str(pan_prob), str(pan_action), str(round(mean_dist, 5)), str(yol_prob),
                            str(yol_action)]
                    writer.writerow(data)
                    writer_y.writerow([str(img), true_actions[img]])
                except KeyError:
                    pass

        # -----------------------------------------------------

        print("0: both, 1: only pan, 2: only yolact, 3:both wrong")
        for k in self.statistics.keys():
            print(k, ":", self.statistics[k] * 100 / images_opened, "%")
        print("is moving % " + str(ismoving_stat * 100 / images_opened))
        self.statistics = {0: 0, 1: 0, 2: 0, 3: 0}

    def evaluate_yolact(self, dataset):
        objects_movements = self.helper.getData("data/objects_movements.json")

        with open(dataset) as f:
            data = f.readlines()

        true_actions = {n.split(" ")[0]: int(n.split(" ")[2]) for n in data}

        print("-" * 20 + "\n")
        categories = {1:"Hammering", 3:"Pliering", 5:"Screw-driving", 7:"Wrenching", 9:"Fake"}
        true_labels = categories.keys()
        fake_labels = [2, 4, 6, 8]
        TN, FP, FN, TP = 0, 1, 2, 3
        classes = {i: [0, 0, 0, 0] for i in true_labels}  # TN, FP, FN, TP
        score = 0

        for image in true_actions.keys():
            obj_movement = objects_movements["testset/rgb/" + image]
            yol_action, _, _, _ = self.get_yolo_action(obj_movement)
            true_action = true_actions[image]
            if true_action in fake_labels:
                true_action = 9
            if yol_action == true_action:
                classes[yol_action][TP] += 1
                for i in classes.keys():
                    if i != yol_action:
                        classes[i][TN] += 1
            else:
                classes[true_action][FN] += 1
                classes[yol_action][FP] += 1
                for i in classes.keys():
                    if i != yol_action and i != true_action:
                        classes[i][TN] += 1

        # ---------------------------------------------------------------
        recall_total = 0
        prec_total = 0
        acc_total = 0
        for i in true_labels:
            recall = classes[i][TP] / (classes[i][TP] + classes[i][FN])
            precision = classes[i][TP] / (classes[i][TP] + classes[i][FP])
            accuracy = (classes[i][TP] + classes[i][TN]) / (
                    classes[i][TP] + classes[i][FP] + classes[i][TN] + classes[i][FN])
            recall_total += recall
            prec_total += precision
            acc_total += accuracy
            print(f"\nAccuracy for {categories[i]}: {accuracy * 100}")
            print(f"Recall for  {categories[i]}: {recall * 100}")
            print(f"Precision for {categories[i]}: {precision * 100}\n")

        print("-" * 20 + "\n")

        print(f"Accuracy total: {acc_total * 100 / len(true_labels)}")
        print(f"Recall total: {recall_total * 100 / len(true_labels)}")
        print(f"Precision total: {prec_total * 100 / len(true_labels)}")

    def clean(self, filename):
        data = {}
        self.helper.dumpData(data, filename)

    def visualisation(self):
        visualiser = Visualiser(helper=help_module.Helper())
        filteredImages = visualiser.get_filtered_images()
        annotation = self.helper.getData("data/objects_movements.json")
        visualiser.create_moving_histogram(annotation, images=filteredImages,
                                           folder="testset/rgb/", title="test", bins=30)
        visualiser.create_obj_histogram(annotation, images=filteredImages,
                                        folder="testset/rgb/", title="hist")
        visualiser.create_boxes_moving_hist(annotation, filteredImages, "testset/rgb/", "test2")
        visualiser.create_boxes_moving_one_action(annotation, filteredImages,
                                                  "testset/rgb/", "hammering_mean_dist", "hammering")


if __name__ == "__main__":
    helper = help_module.Helper()
    # write_to_dataset()

    args = parse_args()
    mainModule = MainModule()

    mainModule.evaluate_yolact("testset/dataset0.txt")
    try:
        if args.evaluate:
            mainModule.evaluate()
        elif args.visualise:
            mainModule.visualisation()
        elif args.annotate:
            mainModule.create_annotation(args.path, args.basic, args.move)
        elif args.video:
            mainModule.annotateVideo(args.path, args.basic, args.move, args.show)
        elif args.prep_dataset:
            clean_dataset(helper)
            create_dataset(args.path)
            separate()


    except KeyboardInterrupt:
        exit()
