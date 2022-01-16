import cv2
import utils

image = cv2.imread('1.png', cv2.IMREAD_COLOR)
blue = utils.get_chars(image.copy(), utils.BLUE) #원본 이미지에 영향을 주지 않기 위해 copy()함수를 사용하는 듯!
green = utils.get_chars(image.copy(), utils.GREEN)
red = utils.get_chars(image.copy(), utils.RED)

cv2.imshow('Image Gray', blue)
cv2.waitKey(0)
cv2.imshow('Image Gray', green)
cv2.waitKey(0)
cv2.imshow('Image Gray', red)
cv2.waitKey(0)
