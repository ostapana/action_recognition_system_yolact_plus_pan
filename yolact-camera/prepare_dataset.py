import glob
import os
import numpy as np

def separate(classes=8):
    with open("testset/banned.txt") as f:
        data = f.readlines()
    banned = np.array([n.split(" ")[0][0:6] for n in data])
    valid_v = {"1":0, "2":0, "3":0, "4":0, "5":0, "6":0, "7":0, "8":0}
    actions_required = round(200 * 0.15)

    with open("testset/dataset0.txt", "r") as file, \
            open("testset/valid" + str(classes) + ".txt", "w") as valid, \
            open("testset/dataset" + str(classes) + ".txt", "w") as dataset:
        for f in file:
            video_name = f[0:6]
            action = f[len(f)-2]
            if video_name not in banned and valid_v[action] < actions_required:
                valid.write(f)
                valid_v[action] += 1
            elif video_name not in banned:
                dataset.write(f)


def create_dataset(path, pan=True):
    folders = glob.glob(os.path.join(path, "*"))
    with open("testset/dataset0.txt", "w") as f:
        for folder in folders:
            name_len = len(folder)
            video_name = folder[name_len - 6:name_len]
            row = video_name + " "
            if pan:
                frames = len(os.listdir(path + "/" + video_name)) - 1  # dir is your directory path
                if frames < 3:
                    continue
                row += str(frames) + " "
            video_name = int(video_name)
            if video_name < 200 or video_name == 410 or video_name == 411:
                row += "1\n"
            elif 200 <= video_name < 409:
                row += "2\n"
            elif (800 <= video_name < 1000) or (1672 <= video_name < 1757):
                 row += "3\n"
            elif 1000 <= video_name < 1200:
                row += "4\n"
            elif (1200 <= video_name < 1300) or (1406 < video_name < 1532):
                row += "5\n"
            elif (1300 <= video_name < 1406) or (1531 < video_name <= 1631):
                row += "6\n"
            elif (411 < video_name < 602) or (1631 < video_name <= 1671):
                 row += "7\n"
            elif 602 <= video_name < 800:
                 row += "8\n"
            else:
                continue
            f.write(row)


#delete videos where yolact didn"t recognise any objects
def clean_dataset(helper):
    objects_movements = helper.getData("data/objects_movements.json")
    object_trash = []
    for o in objects_movements:
        im = int(o[len("testset/rgb/"):])
        objects = objects_movements[o]
        if "hammer" not in objects.keys() and im < 10:
            object_trash.append("00000" + str(im))
        elif "hammer" not in objects.keys() and im < 100:
            object_trash.append("0000" + str(im))
        elif "hammer" not in objects.keys() and im < 200:
            object_trash.append("000" + str(im))
        elif "cube_holes" not in objects.keys() and "wafer" not in objects.keys() and (411 < im < 602):
            object_trash.append("000" + str(im))
        elif "screw_round" not in objects.keys() and "pliers" not in objects.keys() and (800 <= im < 1000):
            object_trash.append("000" + str(im))
        elif "nut" not in objects.keys() and "screw_round" not in objects.keys() \
                and ((im >= 1199 and im < 1302) or (1405 < im < 1532)):
            object_trash.append("00" + str(im))
        elif "cube_holes" not in objects.keys() and "wafer" not in objects.keys() and (1631 < im < 1671):
            object_trash.append("00" + str(im))
        elif "screw_round" not in objects.keys() and "pliers" not in objects.keys() and (1672 <= im < 1757):
            object_trash.append("00" + str(im))

    with open("testset/banned.txt", "w") as f:
        for row in sorted(object_trash):
            f.write(row + "\n")