# Bước 1: Cài các thư viện cần thiết
import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import dump
from collections import Counter
from sklearn.decomposition import PCA

# classifier = cv2.CascadeClassifier(cv2.data.haarcascades +"haarcascade_frontalface_default.xml")
classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

class CustomKNN:
    def __init__(self, n_neighbors=3):
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
            print("k_nearest_labels: ", k_nearest_labels)
            most_common = Counter(k_nearest_labels).most_common(1)
            print("most_common: ", most_common)
            predictions.append(most_common[0][0])
        return predictions


# Bước 2: Tiền xử lý dữ liệu
## 2.1: Load ảnh từ thư mục
current_file_path = os.path.abspath(__file__) 

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

# Hiển thị tập ảnh khuôn mặt trong tập ảnh train
def display_faces_grid(images, labels):
    num_faces = len(images)
    rows = 6
    cols = 6  
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

# Hàm lấy ảnh và gãn nhãn cho ảnh
def load_images_from_folder(folder):
    classifier = cv2.CascadeClassifier(cv2.data.haarcascades +"haarcascade_frontalface_default.xml")
    images = []
    labels = []
    paths = []
    soluonganh=0
    print(" \n List:: ", os.listdir(folder))
    for face_class in os.listdir(folder):
        face_path = os.path.join(folder, face_class)
        for filename in os.listdir(face_path):
            img_path = os.path.join(face_path, filename)
            soluonganh+=1
            img = cv2.imread(img_path)
            if img is not None:
                img = image_resize(img, height = 600)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = classifier.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors = 1)
                print("\n 73 >> faces:", faces)
                faces = sorted(faces, key = lambda x: x[2] * x[3], reverse = True)
                faces = faces[:1]
                print("\n 98 >> faces:", faces)
                if len(faces) >= 1:
                    face = faces[0]
                    print("\n 99 >> faces[0]:", faces[0])
                    x, y, w, h = face
                    im_face = img[y : y + h, x : x + w]
                    if len(faces) == 1:
                        gray_face = cv2.cvtColor(im_face, cv2.COLOR_BGR2GRAY)
                        gray_face = cv2.resize(gray_face, (100, 100))
                        data = gray_face.reshape(-1)
                        paths.append(img_path)
                        images.append(data)
                        labels.append(face_class)
                        print("\ndata: ", data)
                    else:
                        print("Không tìm thấy khuôn mặt")
                else:
                        print("Không tìm thấy khuôn mặt")
                print("\n 92. face_class: ", face_class)
    print(f"Đã đọc thư mục {folder}")
    return images, labels, paths, soluonganh

# Đường dẫn các thư mục cần 
train_folder_path = os.path.join(os.path.dirname(current_file_path), 'train')
train_images, train_labels, train_paths, soluonganh = load_images_from_folder(train_folder_path)

# Kiểm tra số lượng ảnh
print("Số ảnh train:", len(train_images))
print("Số nhãn train:", len(train_labels))

# Hiển thị các khuôn mặt
display_faces_grid(train_images, train_labels)

## Bước 2.2: Xử lý ảnh
# Chuyển đổi ma trận ảnh thành ma trận mảng 1D
train_data = [image.flatten() for image in train_images]

# Bước 3: Khởi tạo mô hình KNN
knn_model = CustomKNN(n_neighbors=3)
knn_model.fit(train_data, train_labels)

## 3.1: Lưu lại mô hình KNN để tái sử dụng
model_path = os.path.join(os.path.dirname(current_file_path), 'train_custom.joblib')
dump(knn_model, model_path)
print("Đã lưu mô hình KNN vào:", model_path)

# Hiển thị biểu đồ PCA cho dữ liệu train
pca = PCA(n_components=2)

# Fit và transform dữ liệu train_data
train_data_pca = pca.fit_transform(train_data)

# Tạo DataFrame từ dữ liệu PCA và labels
df = pd.DataFrame(data=train_data_pca, columns=['Component 1', 'Component 2'])
df['Label'] = train_labels

# Vẽ biểu đồ PCA
plt.figure(figsize=(8, 6))
targets = df['Label'].unique()
colors = ['r', 'g', 'b', 'y', 'm', 'c']
for target, color in zip(targets, colors):
    indices_to_keep = df['Label'] == target
    plt.scatter(df.loc[indices_to_keep, 'Component 1'],
                df.loc[indices_to_keep, 'Component 2'],
                c=color,
                s=50)
plt.legend(targets)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Visualization of Face Labels')
plt.grid()
plt.show()

# Xuất file excel đánh giá cho tập train
data_with_accuracy = {'Label': train_labels, "Paths": train_paths, "soluonganh": soluonganh}
df_test_accuracy = pd.DataFrame(data_with_accuracy)
excel_file_path = os.path.join(os.path.dirname(current_file_path), 'train.xlsx')
df_test_accuracy.to_excel(excel_file_path, index=False)

print("\nHoàn thành training")
