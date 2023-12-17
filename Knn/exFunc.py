import pandas as pd
import numpy as np
import os
import cv2

f_name = "data.csv"
# Tạo file csv
def write(name, data):
    if os.path.isfile(f_name):
        df = pd.read_csv(f_name, index_col = 0)
        latest = pd.DataFrame(data, columns = map(str, range(10000)))
        latest["name"] = name
        df = pd.concat((df, latest), ignore_index = True, sort = False)
    else:
        df = pd.DataFrame(data, columns = map(str, range(10000)))
        df["name"] = name
    df.to_csv(f_name)

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation = inter)
    return resized
    
def save_image(image, name, count):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(BASE_DIR, "Train_Data")
    path = image_dir + "\\" + name + "\\"
    if (not os.path.exists(path)): os.makedirs(path)
    cv2.imwrite(path + name + "_" + str(count) + ".jpg", image)