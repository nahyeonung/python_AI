import numpy as np
import cv2
import utils

FILE_NAME = "trained.npz"

#각 글자의 (1X400) 데이터와 정답 (0~9,+,-,*)
with np.load(FILE_NAME) as data:
    train = data['train']
    train_labels = data['train_labels']

knn = cv2.ml.KNearest_create()
knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)

def check(test, train, train_labels):
    ret, result, neighbours, dist = knn.findNearest(test, k=1)
    return result

def get_result(file_name):
    image = cv2.imread(file_name)
    chars = utils.extract_chars(image) #순서대로 정렬해놓는 함수를 여기서 들고옴
    result_string = ""
    for char in chars:
        matched = check(utils.resize20(char[1]), train, train_labels) #knn으로 학습한 내용을 여기서 사용.
        if matched < 10:
            result_string += str(int(matched))
            continue
        if matched == 10:
            matched = '+'
        elif matched == 11:
            matched = '-'
        elif matched == 12:
            matched = '*'
        result_string += matched
    return result_string
print(get_result("1.png"))