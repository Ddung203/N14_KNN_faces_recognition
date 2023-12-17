import cv2
import os
import numpy as np
from collections import Counter
from joblib import load
import matplotlib.pyplot as plt

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
            print("1 > k_nearest_labels: ", k_nearest_labels)
            most_common = Counter(k_nearest_labels).most_common(1)
            print("2 >> most_common: ", most_common)
            predictions.append(most_common[0][0])
            print("3 >>> predictions: ", predictions)
        return predictions


current_file_path = os.path.abspath(__file__) 

model_path = os.path.join(os.path.dirname(__file__), 'train_custom.joblib')
knn_model = load(model_path)

# Set k 
knn_model.set_n_neighbors(3)

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

def display_faces_grid(images, labels):
    num_faces = len(images)
    rows = 7  # Số hàng trong lưới
    cols = 7  # Số cột trong lưới
    total_plots = rows * cols
    
    if num_faces < total_plots:
        total_plots = num_faces
    
    plt.figure(figsize=(10, 10))
    for i in range(total_plots):
        plt.subplot(rows, cols, i + 1)
        img = images[i].reshape(100, 100)  
        plt.imshow(img, cmap='gray')
        plt.title(f'Face {i+1}\nLabel: {labels[i]}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

classifier = cv2.CascadeClassifier(cv2.data.haarcascades +"haarcascade_frontalface_default.xml")
# image_path = os.path.join(os.path.dirname(current_file_path), 'Dung.jpg')
# image_path = os.path.join(os.path.dirname(current_file_path), 'Thang.jpg')
image_path = os.path.join(os.path.dirname(current_file_path), 'Thanh.jpg')
img = cv2.imread(image_path)
if img is not None:
                    img = image_resize(img, height = 600)
                    print("\n img sau thay đổi:", img)
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
                            flattened_data = data.flatten()
                            predicted_person = knn_model.predict([flattened_data])
                            print("Ảnh đơn của  - ", predicted_person)
                        else:
                            print("Không tìm thấy khuôn mặt")
