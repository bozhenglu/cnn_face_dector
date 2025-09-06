from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image  # 影像載入與轉換
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np         


model = load_model("my_model.h5")
train_dir = "image/train"
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
train_data = train_gen.flow_from_directory(train_dir, target_size=(128,128), batch_size=16, class_mode='categorical')

pic_path = "ugly1.png"  # 改成你要測試的照片

img = image.load_img(pic_path, target_size=(128,128))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

pred = model.predict(img_array)
pred_class = np.argmax(pred, axis=1)[0]

# 對應類別名稱
class_labels = {v: k for k, v in train_data.class_indices.items()}
print("預測類別:", class_labels[pred_class])


