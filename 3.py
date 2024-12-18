import tensorflow as tf
import numpy as np
import cv2
import os
from tkinter import Tk, filedialog, Label, Button, Canvas
from PIL import Image, ImageTk

# بارگذاری مدل ذخیره‌شده
model = tf.keras.models.load_model('unet_model.h5')

# تابع برای بارگذاری و پیش‌بینی تصاویر
def predict_and_display():
    # باز کردن پنجره انتخاب فایل
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if not file_path:
        return  # اگر کاربر هیچ فایلی انتخاب نکرد
    
    # بارگذاری و پردازش تصویر
    img = cv2.imread(file_path)
    original_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # تبدیل به RGB برای نمایش
    img_resized = cv2.resize(img, (128, 128)) / 255.0  # تغییر اندازه و نرمال‌سازی
    img_array = np.expand_dims(img_resized, axis=0)  # افزودن بعد برای پیش‌بینی

    # پیش‌بینی ماسک
    prediction = model.predict(img_array)[0]
    pred_mask = (prediction > 0.5).astype(np.uint8) * 255  # دودویی‌سازی ماسک
    
    # تبدیل ماسک به تصویر
    pred_mask_resized = cv2.resize(pred_mask, (original_img.shape[1], original_img.shape[0]))  # تغییر اندازه ماسک به اندازه تصویر اصلی

    # رسم دایره‌ها روی تصویر اصلی
    result_img = draw_circles_on_tumors(original_img, pred_mask_resized)

    # نمایش تصویر اصلی با دایره‌ها
    display_image(result_img, canvas_original, "Detected Tumors")
    # نمایش ماسک پیش‌بینی‌شده
    display_image(pred_mask_resized, canvas_prediction, "Predicted Mask")

# تابع برای رسم دایره دور نواحی تومور
def draw_circles_on_tumors(image, mask):
    # شناسایی نواحی تومور در ماسک
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # رسم دایره دور هر ناحیه
    for contour in contours:
        # محاسبه دایره حداقل شامل ناحیه
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)
        
        # رسم دایره روی تصویر
        cv2.circle(image, center, radius, (255, 0, 0), 2)  # آبی با ضخامت 2
    
    return image

# تابع کمکی برای نمایش تصاویر
def display_image(image_array, canvas, title):
    # تنظیم اندازه بوم
    h, w = image_array.shape[:2]
    canvas.config(width=w, height=h)
    
    # تبدیل تصویر به PhotoImage و نمایش
    image = Image.fromarray(image_array)
    image_tk = ImageTk.PhotoImage(image)
    canvas.create_image(0, 0, anchor="nw", image=image_tk)
    canvas.image = image_tk  # نگه‌داری مرجع تصویر برای جلوگیری از حذف شدن

# ایجاد محیط GUI
root = Tk()
root.title("Brain Tumor Detection")

# برچسب‌ها
Label(root, text="Detected Tumors").grid(row=0, column=0)
Label(root, text="Predicted Mask").grid(row=0, column=1)

# دکمه بارگذاری و پیش‌بینی
Button(root, text="Load and Predict", command=predict_and_display).grid(row=2, column=0, columnspan=2)

# ایجاد بوم‌ها برای نمایش تصاویر
canvas_original = Canvas(root, bg="gray")
canvas_original.grid(row=1, column=0)

canvas_prediction = Canvas(root, bg="gray")
canvas_prediction.grid(row=1, column=1)

# اجرای محیط گرافیکی
root.mainloop()
