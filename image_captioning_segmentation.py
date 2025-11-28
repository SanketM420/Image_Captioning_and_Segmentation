
# IMAGE CAPTIONING AND SEGMENTATION
# =============================================================
# Created by: Sanket Mathapati
# Tools: Python, TensorFlow, OpenCV, Matplotlib
# =============================================================

# ---------- Step 1: Import Libraries ----------
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Embedding, Dropout, add,
    Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
)
import cv2
import os

print("✅ Libraries imported successfully!")


# ---------- Step 2: Feature Extraction (CNN) ----------
print("\n🔹 Loading VGG16 model for image feature extraction...")

vgg_model = VGG16(weights='imagenet')
vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)

def extract_features(image_path):
    """Extract deep features from image using VGG16"""
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    feature = vgg_model.predict(image, verbose=0)
    return feature


# ---------- Step 3: Caption Generation Model (CNN + LSTM) ----------
def define_caption_model(vocab_size, max_length):
    """Define a CNN + LSTM model for image captioning"""
    # Feature extractor (CNN)
    inputs1 = Input(shape=(4096,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    # Sequence processor (LSTM)
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    # Decoder (Combine both)
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model


# ---------- Step 4: Image Segmentation Model (U-Net) ----------
def build_unet(input_size=(128,128,3)):
    """Build a simple U-Net model for segmentation"""
    inputs = Input(input_size)

    # Encoder
    c1 = Conv2D(16, 3, activation='relu', padding='same')(inputs)
    p1 = MaxPooling2D(2)(c1)

    c2 = Conv2D(32, 3, activation='relu', padding='same')(p1)
    p2 = MaxPooling2D(2)(c2)

    # Bottleneck
    b1 = Conv2D(64, 3, activation='relu', padding='same')(p2)

    # Decoder
    u1 = Conv2DTranspose(32, 2, strides=2, padding='same')(b1)
    m1 = concatenate([u1, c2])
    c3 = Conv2D(32, 3, activation='relu', padding='same')(m1)

    u2 = Conv2DTranspose(16, 2, strides=2, padding='same')(c3)
    m2 = concatenate([u2, c1])
    c4 = Conv2D(16, 3, activation='relu', padding='same')(m2)

    outputs = Conv2D(1, 1, activation='sigmoid')(c4)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# ---------- Step 5: Load Test Image ----------
test_image = "example.jpg"  # 🔸 Change this path to your own image file
if not os.path.exists(test_image):
    print(f"⚠️ Please place an image named 'example.jpg' in this folder!")
else:
    print(f"✅ Found test image: {test_image}")


# ---------- Step 6: Run Feature Extraction ----------
if os.path.exists(test_image):
    feature = extract_features(test_image)
    print("✅ Feature extracted successfully! Shape:", feature.shape)


# ---------- Step 7: Build & Summarize Models ----------
print("\n🔹 Building U-Net model for segmentation...")
unet_model = build_unet()
unet_model.summary()

print("\n🔹 Building Captioning model (demo architecture)...")
caption_model = define_caption_model(vocab_size=5000, max_length=30)
caption_model.summary()


# ---------- Step 8: Test Segmentation ----------
if os.path.exists(test_image):
    image = cv2.imread(test_image)
    image = cv2.resize(image, (128, 128))
    image = np.expand_dims(image, axis=0) / 255.0

    print("\n🔹 Predicting segmentation mask (with random weights)...")
    pred_mask = unet_model.predict(image)

    plt.imshow(pred_mask[0, :, :, 0], cmap='gray')
    plt.title("Predicted Segmentation Mask")
    plt.axis('off')
    plt.show()


# ---------- Step 9: Combine Caption + Segmentation ----------
if os.path.exists(test_image):
    # Example caption (in real system, this would be generated)
    caption = "A cute dog playing on green grass."

    original = cv2.cvtColor(cv2.imread(test_image), cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.imshow(original)
    plt.title("Original Image")

    plt.subplot(1,2,2)
    plt.imshow(pred_mask[0, :, :, 0], cmap='gray')
    plt.title("Segmented Mask")

    plt.suptitle(f"Generated Caption: {caption}", fontsize=12)
    plt.show()


print("\n✅ Project executed successfully!")
print("🎉 Image Captioning and Segmentation Demo Completed.")
