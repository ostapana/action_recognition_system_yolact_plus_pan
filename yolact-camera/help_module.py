import cv2
import os.path
import json
from copy import copy

from yolact.utils.functions import SavePath
from yolact.inference_tool import InfTool

class Helper:
    def __init__(self):
        self.frames = None
        weights = "yolact/weights/weights_yolact_kuka_30/crow_base_25_133333.pth"
        model_path = SavePath.from_str(weights)
        config = model_path.model_name + '_config'

        self.cnn = InfTool(weights=weights, config=config, score_threshold=0.35)

    def dumpData(self, data, file):
        with open(file, 'w') as f:
            try:
                json.dump(data, f, indent=4)
            except:
                return
                self.dumpData(data, file)

    def getData(self, file):
        with open(file) as f:
            try:
                return json.load(f)
            except:
                return self.getData(file)

    def assignCentroids(self, class_names, centroids):
        if not centroids:
            return {}
        assignedCentroids = {}
        for i in range(0, len(centroids)):
            assignedCentroids[class_names[i]] = str(centroids[i])
        return assignedCentroids

    def extract_centroids(self, centroids):
        centroids = centroids[1:len(centroids) - 1]
        centroids = centroids.split()
        x = int(float(centroids[0]))  # string to float then to int
        y = int(float(centroids[1]))
        return x, y

    def extractDepth(self, coordinates, class_names):
        for name in class_names:
            x, y = self.extract_centroids(coordinates[name])
            depth = self.frames.get_depth_frame()
            dist = depth.get_distance(round(x), round(y))
            print("Depth of " + name + " is " + str(dist))

    def updateFrames(self, frames):
        self.frames = frames

    def assignAndDumpData(self, class_names, centroids):
        data = {'class_names': class_names,
                'centroids': centroids}
        self.dumpData(data, 'classes.json')

    def updateInfo(self, class_names, assigned_centroids):
        data = self.getData('classes.json')
        if data is None:
            return
        self.assignAndDumpData(class_names, assigned_centroids)

    def annotate(self, filename, annotation_f, centroids, annotations):
        data = self.getData(annotation_f)
        data[filename] = centroids
        self.dumpData(data, annotation_f)

    def applyYolactImage(self, img, annotation, annotation_f=None, filename=None, annotations=None):
        preds, frame = self.cnn.process_batch(img)
        classes, class_names, scores, boxes, masks, centroids = self.cnn.raw_inference(img, preds=preds, frame=frame)
        c = self.assignCentroids(class_names, centroids)
        img_numpy = self.cnn.label_image(img, preds=preds)
        if annotation:
            self.annotate(filename, annotation_f, c, annotations)
        else:
            self.updateInfo(class_names, self.assignCentroids(class_names, centroids))
        return img_numpy

    def applyYolact4Video(self, img):
        preds, frame = self.cnn.process_batch(img)
        classes, class_names, scores, boxes, masks, centroids = self.cnn.raw_inference(img, preds=preds, frame=frame)
        img_numpy = self.cnn.label_image(img, preds=preds)
        return classes, class_names, scores, centroids, img_numpy

    def for_test(self, img):
        preds, frame = self.cnn.process_batch(img)
        classes, class_names, scores, boxes, masks, centroids = self.cnn.raw_inference(img, preds=preds, frame=frame)
        img_numpy = self.cnn.label_image(img, preds=preds)
        from PIL import Image
        im = Image.fromarray(img_numpy)
        im.save("camera.jpg")
