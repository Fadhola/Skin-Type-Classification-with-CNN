# Skin Type Classification with CNN

This project uses deep learning to classify human facial skin types into **oily**, **dry**, or **normal**, helping skincare beginners select suitable products and avoid harmful misusage.

## Objective

To build a Computer Vision model that detects skin type from images, providing users with personalized skincare advice based on their actual skin condition.

## Why ?

Many people misjudge their skin type, often choosing inappropriate skincare products, which can worsen issues like irritation or acne.  
Source references:

- [Healthshots: Dehydrated vs Oily Skin](https://www.healthshots.com/beauty/skin-care/dehydrated-oily-skin-symptoms-and-treatment/)
- [Canoe: Common Skincare Issues](https://canoe.com/life/fashion/common-skincare-issues-how-to-treat)

## Target Users

People who are new to skincare and unsure about their skin type.

## Approach

1. **Data Collection**: Kaggle dataset (Oily-Dry-Normal Skin Types) split into train/validation/test.
2. **EDA**: Identified data quality issues like wrong labels, noise (ads, partial faces), lighting inconsistencies.
3. **Preprocessing**:

   - Resizing, rescaling.
   - Data augmentation: rotation, zoom, brightness control.
   - Class rebalancing and visual inspection.

4. **Model 1**:

   - CNN with Conv2D → MaxPooling → Flatten → Dense.
   - Issues: Overfitting, low validation accuracy (~35–40%).

5. **Model 2 Improvements**:

   - HeNormal kernel initializer.
   - BatchNormalization and Dropout (0.5, 0.3).
   - Learning rate: 1e-4.
   - Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint.
   - Results: Smoother loss curves, validation accuracy ↑, more balanced class performance.

6. **Deployment**:
   - Built Streamlit web app.
   - Users can upload an image to receive real-time skin type predictions.

**Try it here:** [Skin Type Classification App](https://huggingface.co/spaces/fadhol/Skin-Type-Classification)

## Results

- **Validation Accuracy**: ~41%
- Improved classification for “dry” and “oily” classes.
- Significant reduction in model overfitting.
- Final model saved as `.keras`.

> > ⚠️ Note: The current validation accuracy (~41%) indicates this is a proof of concept. Further improvements, especially in data quality and model architecture, are ongoing.

## Tech Stack

- **Python**
- **TensorFlow / Keras**
- **Streamlit**
- **Pandas / NumPy / Matplotlib / Seaborn**
- **PIL / OpenCV**
- **scikit-learn**
- **Kaggle Dataset**

## Model Inference

Live prediction available in `inference.ipynb`.  
Model correctly predicts skin types but still needs data cleaning and possibly transfer learning for stronger generalization.

## To-Do (Next Steps)

- Clean the dataset (remove mislabeled/irrelevant images).
- Try transfer learning (e.g., MobileNetV2, EfficientNet).
- Improve UI/UX in Streamlit.
- Possibly include lighting correction or face detection.

## Author

Fadhola Asandi  
Skin Type Classification Project — June 2025
