import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow.keras import layers, models

# بارگذاری داده‌ها (کد قبلی که نوشتید)
def load_data(image_dir, mask_dir, image_size=(128, 128)):
    images, masks = [], []
    
    for filename in os.listdir(image_dir):
        img_path = os.path.join(image_dir, filename)
        img = cv2.imread(img_path)
        
        if img is None:
            continue
        
        img = cv2.resize(img, image_size)
        images.append(img / 255.0)
        
        mask_path = os.path.join(mask_dir, filename)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            continue
        
        mask = cv2.resize(mask, image_size)
        mask = mask // 255
        masks.append(mask)
    
    X = np.array(images)
    y = np.array(masks)
    
    y = y.reshape(-1, image_size[0], image_size[1], 1)
    
    return X, y

# مسیرهای تصاویر و ماسک‌ها
image_dir = r'C:\Users\Danial\Downloads\archive\images'
mask_dir = r'C:\Users\Danial\Downloads\archive\masks'

# بارگذاری داده‌ها
X_train, y_train = load_data(image_dir, mask_dir)

# ساخت مدل UNet ساده
def build_unet(input_size=(128, 128, 3)):
    inputs = layers.Input(input_size)
    
    # Encoder
    conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)
    
    # Decoder
    up1 = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(pool1)
    concat1 = layers.concatenate([up1, conv1], axis=3)
    conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(concat1)
    conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    
    # خروجی
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(conv2)
    
    model = models.Model(inputs, outputs)
    return model

# ساخت مدل
model = build_unet()

# کامپایل مدل
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# آموزش مدل
model.fit(X_train, y_train, batch_size=8, epochs=10, validation_split=0.1)

# ذخیره مدل پس از آموزش
model.save('unet_model.h5')
print("مدل ذخیره شد!")

