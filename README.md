# AI E-Commerce Product Generator

An AI-powered e-commerce application that analyzes fashion product images using deep learning and generates product descriptions with an LLM.

## 🚀 Live Demo

[Try the live app](https://semanrbingl-ai-ecommerce-product-generator-appapp-i82ha7.streamlit.app/)

## Features

- Fashion product category classification with ResNet50
- Confidence score for model predictions
- Grad-CAM visualization for model explainability
- Image-based product description generation with an LLM
- Streamlit web application

## Model Performance

- Best validation accuracy: **87.63%**
- Test accuracy: **86.54%**
- Training epochs: **5**

## Dataset

This project uses the [Fashion Product Images Small](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small) dataset from Kaggle.

The dataset contains fashion product images and metadata used for training and evaluating the image classification model.

The full dataset is not included in this repository due to its size.

For best results, the application is expected to perform more consistently on product images that are visually similar to the training dataset.

## Technologies

- Python
- PyTorch
- ResNet50
- OpenCV
- Grad-CAM
- Streamlit
- OpenAI API

## Project Pipeline

**Product Image → ResNet50 → Category Prediction → Grad-CAM → LLM Product Description**
