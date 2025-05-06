# Функции общего назначения

import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
import cv2
from PIL import Image
import PIL.ImageOps 


def show(im):
    plt.axis('off')
    plt.imshow(im)


def save_image(image, filename):
    """Сохраняет изображение по указанному имени файла."""
    cv2.imwrite(filename, image)
    print(f"Изображение сохранено как {filename}")


def preprocess_image(image):
    """Применяет Гауссов фильтра для уменьшения шума"""
    # Применение Гауссового фильтра для уменьшения шума
    return cv2.GaussianBlur(image, (9, 9), 0)


# преместить по осям изображение фрагмента
def translate_image(image, dx, dy):
    # Получение размеров изображения
    height, width = image.shape[:2]
    # Создание матрицы перевода
    translation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    # Применение аффинного преобразования к изображению
    translated_image = cv2.warpAffine(image, translation_matrix, (width, height))
    return translated_image


def remove_background_gray(gray):
    # Применение пороговой бинаризации для выделения переднего плана
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    # Применение инверсии, чтобы передний план был белым, а фон черным
    binary = cv2.bitwise_not(binary)
    # Применение морфологических операций для удаления шума
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    # Применение маски к оригинальному изображению
    result = cv2.bitwise_and(gray, gray, mask=binary)
    return result