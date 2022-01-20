import cv2
import numpy as np
import re
#색에 대해서 알아야 할 것.
#파란색: RGB에서 B 값이 항상 FF(255)입니다.
#초록색: RGB에서 G 값이 항상 FF(255)입니다.
#빨간색: RGB에서 R 값이 항상 FF(255)입니다.
#파&초: RGB에서 R 값이 항상 AA 이하 입니다.
#파&빨: RGB에서 G 값이 항상 AA 이하 입니다.
#초&빨: RGB에서 B 값이 항상 AA 이하 입니다.

#이미지에서 숫자의 색은 항상 옆과 다르다....

BLUE = 0
GREEN = 1
RED = 2

#특정한 색상의 모든 단어가 포함된 이미지를 추출합니다.
#색상을 확인할 때 img[row, column, channel]문법이다.
#channel은 R,G,B,순서대로 0~1까지 주어진다. 하지만 OpenCV에선 B,G,R순서대로 0~1까지!
def get_chars(image, color):
    other_1 = (color + 1) % 3
    other_2 = (color + 2) % 3
    c = image[:, :, other_1] == 255 #Green [:,:,1]은 녹색채널!!!!
    image[c] = [0,0,0] # 검은색 
    c = image[:, :, other_2] == 255 #Red [:,:,2]는 레드채널!!!
    image[c] = [0,0,0]
    c = image[:,:,color] < 170 # G/R 여기까지 거를 것들 다 거르는 역할. 그리고 밑에서 남은 것들 다 하얀색 만들어 줌.
    image[c] = [0,0,0]
    c = image[:,:,color] != 0
    image[c] = [255, 255, 255] #하얀색 
    return image

# 전체 이미지에서 왼쪽부터 단어별로 이미지를 추출한다.
def extract_chars(image):
    chars = []
    colors = [BLUE, GREEN, RED]
    #대충 thresholding으로 흑백화된 그림의 한계점을 나타내고 그 한계점을 이용해서 contour하면 원본 이미지에서 그 한계점을 선으로 그림.
    #contour: 동일한 색 또는 동일한 색상 강도를 가진 부분의 가장자리 경계를 연결한 선.
    for color in colors:
        image_from_one_color = get_chars(image.copy(), color) # 색깔별로 값들이 여기에 담길 것 같다.
        image_gray = cv2.cvtColor(image_from_one_color, cv2.COLOR_BGR2GRAY) #색깔별로 담긴 것을 여기서 흑백화.
        ret, thresh = cv2.threshold(image_gray, 127, 255, 0) #(img, threshold_value, value, flag)
        #RETR_EXTERNAL 옵션으로 숫자의 외각을 기준으로 분리
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            #추출된 이미지 크기가 50이상인 경우만 실제 문자 데이터인 것으로 파악
            area = cv2.contourArea(contour)
            if area > 50:
                x, y, width, height = cv2.boundingRect(contour)
                roi = image_gray[y:y + height, x:x + width]
                chars.append((x, roi))

    #람다식은 그냥 단순한 수학 식. char[0]이면 리스트의 0번째 값들을 기준(x축을 기준)으로 정렬.(그래야 왼쪽->오른쪽으로 정렬됨)
    chars = sorted(chars, key=lambda char: char[0]) #sorted(iterable, key=<function>, reverse=<True|False>)
    return chars

    # 특정한 이미지를 (20x20) 크기로 Scaling 한다.
def resize20(image):
    resized = cv2.resize(image, (20,20))
    return resized.reshape(-1,400).astype(np.float32)

def remove_firest_0(string): #불필요한 0을 제거하는 함수
    temp = []
    for i in string:
        if i == '+' or i == '-' or i == '*':
            temp.append(i)
    split = re.split('\*|\+|-', string)
    i = 0
    temp_count = 0
    result = ""
    for a in split:
        a = a.lstrip('0')
        if a == '':
            a = '0'
        result += a
        if i < len(split) - 1:
            result += temp[temp_count]
            temp_count = temp_count + 1
        i = i + 1
    return result        