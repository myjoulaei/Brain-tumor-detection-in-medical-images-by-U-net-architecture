import numpy as np
import cv2
import os

# بارگذاری داده‌های واقعی
def load_data(image_dir, mask_dir, image_size=(128, 128)):
    images, masks = [], []
    
    for filename in os.listdir(image_dir):
        # بارگذاری تصویر
        img_path = os.path.join(image_dir, filename)
        img = cv2.imread(img_path)
        
        if img is None:
            continue  # اگر تصویر بارگذاری نشد، آن را نادیده بگیریم
        
        img = cv2.resize(img, image_size)  # تغییر اندازه به 128x128
        images.append(img / 255.0)  # نرمال‌سازی به بازه [0, 1]
        
        # بارگذاری ماسک
        mask_path = os.path.join(mask_dir, filename)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            continue  # اگر ماسک بارگذاری نشد، آن را نادیده بگیریم
        
        mask = cv2.resize(mask, image_size)  # تغییر اندازه به 128x128
        masks.append(mask // 255)  # دودویی کردن ماسک‌ها (0 یا 1)
    
    # تبدیل لیست‌ها به آرایه‌های numpy و بازسازی ابعاد
    X = np.array(images)
    y = np.array(masks)
    
    # اطمینان از اینکه ابعاد صحیح برای مدل استفاده می‌شود
    y = y.reshape(-1, image_size[0], image_size[1], 1)  # تبدیل به (n_samples, height, width, 1)
    
    return X, y

# مسیرهای تصاویر و ماسک‌ها
image_dir = r'C:\Users\Danial\Downloads\archive\images'
mask_dir = r'C:\Users\Danial\Downloads\archive\masks'

# بارگذاری داده‌ها
X_train, y_train = load_data(image_dir, mask_dir)
