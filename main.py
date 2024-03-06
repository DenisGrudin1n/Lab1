import cv2
import numpy as np

# Зчитування вихідного зображення
image_path = 'Pictures/4.jpg'
image = cv2.imread(image_path)

# Зміна розміру зображення
new_width = int(input("Введіть нову ширину зображення: "))
new_height = int(input("Введіть нову висоту зображення: "))
resized_image = cv2.resize(image, (new_width, new_height))

# Перетворення в градації сірого
gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

# Обчислення градієнтів по X та Y за допомогою Sobel фільтрів
gradient_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
gradient_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)

# Обчислення структурного тензору за допомогою матриці Харріса
window_size = 3
k = 0.04
corner_response = cv2.cornerHarris(gray_image, blockSize=3, ksize=3, k=0.04)

# Відбір кутів за допомогою порогового значення
threshold = 0.01 * corner_response.max()
corners = np.argwhere(corner_response > threshold)

# Копіювання оригінального розміру зображення для відображення кутів
image_with_corners = resized_image.copy()

# Відмалювання кутів на зображенні
for corner in corners:
    x, y = corner[1], corner[0]
    cv2.circle(image_with_corners, (x, y), 5, (0, 255, 0), 2)

# Відображення оригінального та відзначеного зображення
cv2.imshow('Original Image', resized_image)
cv2.imshow('Corners Detected', image_with_corners)

cv2.waitKey(0)
cv2.destroyAllWindows()
