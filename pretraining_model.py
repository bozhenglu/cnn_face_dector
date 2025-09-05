from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import numpy as np

# -------------------------
# 1️⃣ 訓練資料設定
# -------------------------
train_dir = "image/train"
val_dir = "image/val"
test_dir = "image/test"
num_classes = 2

# 資料增強
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)
val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(train_dir, target_size=(128,128), batch_size=16, class_mode='categorical')
val_data = val_gen.flow_from_directory(val_dir, target_size=(128,128), batch_size=16, class_mode='categorical')

# -------------------------
# 2️⃣ 建立模型
# -------------------------
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128,128,3))
base_model.trainable = False  # 冷凍卷積層

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# -------------------------
# 3️⃣ 訓練模型
# -------------------------
model.fit(train_data, validation_data=val_data, epochs=10)
model.save("my_model.h5")   
# -------------------------
# 4️⃣ 測試資料準確率
# -------------------------
test_gen = ImageDataGenerator(rescale=1./255)
test_data = test_gen.flow_from_directory(test_dir, target_size=(128,128), batch_size=16, class_mode='categorical')
loss, acc = model.evaluate(test_data)
print("測試準確率:", acc)

# -------------------------
# 5️⃣ 單張圖片預測
# -------------------------
pic_path = "cute.jpg"  # 改成你要測試的照片

img = image.load_img(pic_path, target_size=(128,128))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

pred = model.predict(img_array)
pred_class = np.argmax(pred, axis=1)[0]

# 對應類別名稱
class_labels = {v: k for k, v in train_data.class_indices.items()}
print("預測類別:", class_labels[pred_class])
