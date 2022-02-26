import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
assert tf.__version__.startswith('2')

from tflite_model_maker import model_spec
from tflite_model_maker import image_classifier
from tflite_model_maker.image_classifier import DataLoader


#화면에 25개의 샘플 이미지를 보여주는 함수를 만듬.
def showSample(data):

    plt.figure(figsize=(10,10))
    for i, (image, label) in enumerate(data.gen_dataset().unbatch().take(25)):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image.numpy(), cmap=plt.cm.gray)
        plt.xlabel(data.index_to_label[label.numpy()])
    plt.show()

#원래의 이미지 라벨과 다른 라벨이 나왔다면 라벨의 색을 red로 지정
def get_label_color(val1, val2):
    if val1 == val2:
        return 'black'
    else:
        return 'red'


#100개의 이미지를 예측하는 함수를 만듬. 여기서 아까 만든 get_label_color함수를 사용함으로써 잘못된 학습 구분.
def showPredicted(data):

    plt.figure(figsize=(20, 20))
    predicts = model.predict_top_k(data)
    for i, (image, label) in enumerate(data.gen_dataset().unbatch().take(100)):
        ax = plt.subplot(10, 10, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image.numpy(), cmap=plt.cm.gray)

        predict_label = predicts[i][0][0]
        color = get_label_color(
            predict_label,
            data.index_to_label[label.numpy()]
            )
        ax.xaxis.label.set_color(color)
        plt.xlabel('Predicted: %s' % predict_label)
    plt.show()


# 학습에 필요한 이미지를 image_path라는 변수에 저장.
image_path = tf.keras.utils.get_file(
      'flower_photos',
      'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
      untar=True)

# 밑의 print함수를 이용해서 image_path 경로를 알 수 있다. 그 경로로 들어가면 다운로드 된 이미지를 확인할 수 있다.
print("======================================")
print("image_path=",image_path)
print("======================================")

#data에 image들을 넣고 train_data와 test_data변수에 9대1 비율로 이미지를 담는다.
data = DataLoader.from_folder(image_path)
train_data, test_data = data.split(0.9)

showSample(data)

print("======================================")
print("The number of Data =", len(data))
print("The number of Train data =", len(train_data))
print("The number of Test Data =", len(test_data))
print("======================================")

#image_classifier API를 사용함으로써 이미지를 분류하는 학습을 함. 여기서 epoch는 기본 5로 설정되어 있음.
model = image_classifier.create(train_data)  

#학습된 모델을 평가하는 코드
loss, accuracy = model.evaluate(test_data)
showPredicted(test_data)

#tensorflow lite파일을 현재 디렉터리에 만드는 함수.
model.export(export_dir='.', with_metadata=False)