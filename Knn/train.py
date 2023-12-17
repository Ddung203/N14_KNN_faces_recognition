import cv2
import numpy as np
import exFunc
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

f_name = "data.csv"
classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def display_pca_plot(X_train, Y_train):
    unique_labels = np.unique(Y_train)

    for label in unique_labels:
        label_indices = np.where(Y_train == label)[0]
        label_images = X_train[label_indices]

        # Apply PCA to reduce dimensionality to 2 for visualization
        pca = PCA(n_components=2)
        label_images_pca = pca.fit_transform(label_images)

        plt.scatter(label_images_pca[:, 0], label_images_pca[:, 1], label=str(label))

    plt.title("PCA Plot for Training Images")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.show()

def train_by_camera(save_img = False):
    name = input("Nhập tên của bạn: ")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    face = (0,0,0,0)
    f_list = []
    auto_capture = False
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = classifier.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors = 5)
        faces = sorted(faces, key = lambda x: x[2] * x[3], reverse = True)
        faces = faces[:1]
        if len(faces) >= 1:
            face = faces[0]
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)
        x, y, w, h = face
        im_face = frame[y : y + h, x : x + w]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
        if not ret:
            continue
        cv2.imshow("Face-Recognition", frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord("q"):
            break
        elif key & 0xFF == ord("c") or auto_capture:
            if len(faces) == 1:
                gray_face = cv2.cvtColor(im_face, cv2.COLOR_BGR2GRAY)
                gray_face = cv2.resize(gray_face, (100, 100))
                print(len(f_list), type(gray_face), gray_face.shape)
                f_list.append(gray_face.reshape(-1))
                if (save_img):
                    exFunc.save_image(gray_face, name, len(f_list))
            else:
                print("Không tìm thấy khuôn mặt")
            if len(f_list) == 20:
                if (f_list): exFunc.write(name, np.array(f_list))
                f_list = []
                auto_capture = False
        elif key & 0xFF == ord("s"):
            auto_capture = True
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("1. Huấn luyện bằng camera")
    sel = input("Nhập lựa chọn (yes/no): ")

    if sel.lower() == "yes" or sel.lower() == "y":
        train_by_camera(save_img=True)

        # Read data from the CSV file
        data = pd.read_csv(f_name).values
        X_train, Y_train = data[:, 1:-1], data[:, -1]

        # Display PCA plot for each training image
        display_pca_plot(X_train, Y_train)

        # Continue with the face recognition code...
    elif sel.lower() == "no" or sel.lower() == "n":
        print("Bạn đã chọn không huấn luyện bằng camera.")
    else:
        print("Lựa chọn không hợp lệ.")

