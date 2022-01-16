import cv2
import numpy as np
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
    for color in colors:
        image_from_one_color = get_chars(image.copy(), color)
        image_gray = cv2.cvtColor(image_from_one_color, cv2.COLOR_BGR2GRAY)
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


    chars = sorted(chars, key=lambda char: char[0])
    return chars

    # 특정한 이미지를 (20x20) 크기로 Scaling 한다.
def resize20(image):
    resized = cv2.resize(image, (20,20))
    return resized.reshape(-1,400).astype(np.float32)
