import cv2
import os
import re
import numpy as np
import pandas as pd
from joblib import load
from collections import Counter
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Bước 1: Load model từ file train_custom.joblib
class CustomKNN:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def set_n_neighbors(self, n_neighbors):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X):
        predictions = []
        for i in range(len(X)):
            distances = [self.euclidean_distance(X[i], x_train) for x_train in self.X_train]
            k_indices = np.argsort(distances)[:self.n_neighbors]
            k_nearest_labels = [self.y_train[j] for j in k_indices]
            print("\n1 > k_nearest_labels: ", k_nearest_labels)
            most_common = Counter(k_nearest_labels).most_common(1)
            print("2 >> most_common: ", most_common)
            predictions.append(most_common[0][0])
            print("3 >>> predictions: ", most_common[0][0])
        return predictions

current_file_path = os.path.abspath(__file__) 

model_path = os.path.join(os.path.dirname(__file__), 'train_custom.joblib')
knn_model = load(model_path)

# Lựa chọn tham số K
knn_model.set_n_neighbors(3)

# Bước 2: Load thư mục test
## 2.1: Load ảnh từ thư mục test

# Hàm thay đổi kích thước của ảnh
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


# Hàm lấy ảnh và gãn nhãn cho ảnh
def load_images_from_folder(folder):
    classifier = cv2.CascadeClassifier(cv2.data.haarcascades +"haarcascade_frontalface_default.xml")
    images = []
    labels = []
    paths = []
    soluonganh=0
    for face_class in os.listdir(folder):
        face_path = os.path.join(folder, face_class)
        if os.path.isdir(face_path):
            for filename in os.listdir(face_path):
                img_path = os.path.join(face_path, filename)
                soluonganh+=1
                img = cv2.imread(img_path)
                if img is not None:
                    img = image_resize(img, height = 600)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces = classifier.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors = 0)
                    print("\n 73 >> faces:", faces)
                    faces = sorted(faces, key = lambda x: x[2] * x[3], reverse = True)
                    faces = faces[:1]
                    print("\n 76 >> faces:", faces)
                    if len(faces) >= 1:
                        face = faces[0]
                        x, y, w, h = face
                        im_face = img[y : y + h, x : x + w]
                        if len(faces) == 1:
                            gray_face = cv2.cvtColor(im_face, cv2.COLOR_BGR2GRAY)
                            gray_face = cv2.resize(gray_face, (100, 100))
                            data = gray_face.reshape(-1)
                            images.append(data)
                            labels.append(face_class)
                            paths.append(img_path)
                            print("\ndata: ", data)
                        else:
                            print("Không tìm thấy khuôn mặt")
                    print("\n 92. face_class: ", face_class)
    return images, labels, paths, soluonganh

# Đường dẫn các thư mục cần
test_folder_path = os.path.join(os.path.dirname(current_file_path), 'test')
test_images, test_labels, test_paths, soluonganh = load_images_from_folder(test_folder_path)

# Kiểm tra số lượng ảnh test
print("Số lượng ảnh trong tập test:", len(test_images))
print("Số lượng nhãn trong tập test:", len(test_labels))

## Bước 2.2: Xử lý ảnh
# Chuyển đổi ma trận ảnh test thành ma trận mảng 1D
test_data = [image.flatten() for image in test_images]

# Bước 3: Sử dụng mô hình để nhận diện ảnh
# Nhận diện tập ảnh test
tested_labels = knn_model.predict(test_data)

# Đánh giá độ chính xác của thuật toán trên tập test
accuracy_test = accuracy_score(test_labels, tested_labels)
print("\n\n >>> Độ chính xác trên tập test: ", accuracy_test)

# Xuất file excel đánh giá cho tập test
data_with_accuracy = {'Label': test_labels, 'Detect': tested_labels, 'Accuracy': [accuracy_test] * len(test_labels), "Paths": test_paths, "soluonganh": soluonganh}
df_test_accuracy = pd.DataFrame(data_with_accuracy)
excel_file_path = os.path.join(os.path.dirname(current_file_path), 'test.xlsx')
df_test_accuracy.to_excel(excel_file_path, index=False)


