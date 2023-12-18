# Nhận diện khuôn mặt dựa trên dữ liệu
# đã được lưu trữ trước đó
# và phân tích bằng thuật toán KNN
import cv2
import numpy as np
import pandas as pd
import operator
import os
import exFunc
from operator import itemgetter
from exFunc import f_name

# Hàm tính khoảng cách Euclid
def euc_dist(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

class KNN:
    def __init__(self, K=3):
        self.K = K

    def fit(self, x_train, y_train):
        self.X_train = x_train
        self.Y_train = y_train

    def predict(self, X_test):
        predictions = []
        for i in range(len(X_test)):
            dist = np.array([euc_dist(X_test[i], x_t) for x_t in self.X_train])
            dist_sorted = dist.argsort()[: self.K]
            neigh_count = {}
            for idx in dist_sorted:
                if self.Y_train[idx] in neigh_count:
                    neigh_count[self.Y_train[idx]] += 1
                else:
                    neigh_count[self.Y_train[idx]] = 1
            sorted_neigh_count = sorted(neigh_count.items(), key = operator.itemgetter(1), reverse = True)
            predictions.append(sorted_neigh_count[0][0])
        return predictions
if (os.path.exists(f_name) == False):
    print("Chưa có dữ liệu huấn luyện. Hãy chạy chương trình huấn luyện trước.")
    exit()

# Đọc dữ liệu hình ảnh trong tập tin csv
data = pd.read_csv(f_name).values
# Phân loại dữ liệu
X, Y = data[:, 1:-1], data[:, -1]
# print(X, Y)
# Sử dụng KNN với K = 5
model = KNN(K=5)
# Mẫu huấn luyện
model.fit(X, Y)

print("Kiểm thử bằng camera")
sel = input("Nhập lựa chọn (yes/no): ")
if sel.lower() == "y":
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
elif sel.lower() == "n":
    print("Bạn đã chọn không huấn luyện bằng camera.")
else:
    print("Lựa chọn không hợp lệ.")
# if (sel == "1"):
#     cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

classifier = cv2.CascadeClassifier(cv2.data.haarcascades +"haarcascade_frontalface_default.xml")
f_list = []
x, y, w, h = (0,0,0,0)
response_name = "Unknown"
count_err = 0

while True:
    if (sel.lower() == "y"):
        ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray, 1.5, 5)
    X_test = []
    # Kiểm tra dữ liệu
    for face in faces:
        x, y, w, h = face
        im_face = gray[y : y + h, x : x + w]
        im_face = cv2.resize(im_face, (100, 100))
        X_test.append(im_face.reshape(-1))
    if len(faces) > 0:
        count_err = 0
        # Dự đoán bằng KNN
        response = model.predict(np.array(X_test))
        for i, face in enumerate(faces):
            x, y, w, h = face
            response_name = response[i]
            print(response_name, x, y, w, h)
    else:
        count_err += 1
    if count_err > 50:
        x, y, w, h = (0,0,0,0)
        response_name = "Unknown"
    # Vẽ khung nhận diện khuôn mặt
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 1)
    # Hiển thị tên của khuôn mặt nhận diện được
    ((response_w, response_h), _) = cv2.getTextSize(response_name, cv2.FONT_HERSHEY_SIMPLEX, 1, 0)
    if w < response_w + 20:
        x_w = x + response_w + 20
    else:
        x_w = x + w
    cv2.rectangle(frame, (x, y), (x_w, y - 50), (0, 0, 0), -1)
    cv2.putText(frame, response_name, (x + 10, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 0,)
    cv2.imshow("Face-Recognition", frame)
    key = cv2.waitKey(1)
    if key & 0xFF == ord("q"):
        break
if (sel.lower() == "y"):
    cap.release()
cv2.destroyAllWindows()
