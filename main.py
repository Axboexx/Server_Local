import cv2
import numpy as np

# 加载图像
image = cv2.imread('123.jpg')

# 图像去噪
denoised_image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

# 图像锐化
sharp_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
sharpened_image = cv2.filter2D(denoised_image, -1, sharp_kernel)

# 图像增强
enhanced_image = cv2.convertScaleAbs(sharpened_image, alpha=1.2, beta=0)

# 图像对比度增强

cv2.imwrite('new.jpg',enhanced_image)