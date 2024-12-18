در این پروژه، سیستمی مبتنی بر یادگیری عمیق برای تشخیص تومورهای مغزی در تصاویر MRI طراحی و پیاده‌سازی شده است. مدل پیشنهادی از معماری U-Net استفاده می‌کند که به‌طور خاص برای مسائل بخش‌بندی تصویر طراحی شده است. تصاویر MRI و ماسک‌های مرتبط پس از پیش‌پردازش شامل تغییر اندازه و نرمال‌سازی، برای آموزش مدل استفاده شده‌اند.
پس از آموزش، مدل توانایی تشخیص نواحی تومور را با دقت بالا به دست می‌آورد. برای افزایش کارایی و سهولت استفاده، یک رابط گرافیکی کاربرپسند (GUI) با استفاده از کتابخانه Tkinter طراحی شده است که به کاربران امکان بارگذاری تصویر MRI، پیش‌بینی ماسک تومور، و مشاهده نواحی شناسایی‌شده را می‌دهد.
این سیستم می‌تواند به عنوان یک ابزار کمکی در تحلیل تصاویر پزشکی و تشخیص سریع‌تر تومورها مورد استفاده قرار گیرد و پایه‌ای برای توسعه ابزارهای مشابه برای سایر انواع تصاویر پزشکی باشد.


In this project, a deep learning-based system has been developed for detecting brain tumors in MRI images. The proposed model utilizes the U-Net architecture, specifically designed for image segmentation tasks. MRI images and their corresponding masks are preprocessed, including resizing and normalization, and then used for training the model.
After training, the model demonstrates high accuracy in identifying tumor regions. To enhance usability, a user-friendly graphical interface (GUI) is designed using the Tkinter library, allowing users to upload MRI images, predict tumor masks, and visualize detected regions.
This system serves as an assistive tool for analyzing medical images and accelerating tumor detection, laying a foundation for developing similar tools for other types of medical imaging.
