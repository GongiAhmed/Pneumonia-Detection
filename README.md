# Pneumonia Detection using TensorFlow/Keras
This notebook demonstrates a pneumonia detection system using a TensorFlow/Keras model trained on the RSNA Pneumonia Detection Challenge dataset.  It utilizes a multi-output model to simultaneously classify images as having pneumonia or not, and to localize the pneumonia region with a bounding box.
<p align="center">
  <img src="https://github.com/GongiAhmed/Pneumonia-Detection/blob/main/Pneumonia%20Detection/output.png" />
</p>

## Overview

The notebook performs the following steps:

1. **Data Loading and Preprocessing:** Loads the training labels from "https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/data" and the DICOM images from `stage_2_train_images`.  Images are preprocessed by resizing and padding to a fixed size (244x244) and normalized. Bounding box coordinates are also normalized.
2. **Data Formatting:**  A custom function `format_image` handles the resizing and padding, while `format_instance` converts labels to one-hot encoding for classification and scales bounding box coordinates.
3. **Dataset Creation and Tuning:** TensorFlow `tf.data` is used to create training, validation, and test datasets. Performance optimizations like `map` with `num_parallel_calls`, `shuffle`, `repeat`, `batch`, and `prefetch` are applied.
4. **Model Building:** A multi-output convolutional neural network is constructed. It consists of:
    - A feature extractor based on convolutional layers and average pooling.
    - A model adaptor using a dense layer.
    - Two output heads: a classification head (softmax activation) and a regression head (for bounding box coordinates).
5. **Model Training:**  The model is trained using the Adam optimizer with categorical crossentropy loss for classification and mean squared error (MSE) loss for regression.
6. **Evaluation:** The trained model is evaluated on a test set using accuracy for classification and Intersection over Union (IoU) for bounding box localization. Predictions are visualized with bounding boxes overlaid on the images, and an overall summary of accuracy and mean IoU is provided. The predicted images are saved in the `output_predictions` directory.
7. **Output:**  The `output_predictions` directory is zipped into `predictions.zip` which contains individual prediction images and an image showing all predictions ("all_predictions.png").
<p align="center">
  <img src="https://github.com/GongiAhmed/Pneumonia-Detection/blob/main/Pneumonia%20Detection/result.png" />
</p>

## Requirements

This notebook requires the following libraries:

- `opencv-python` (cv2)
- `numpy`
- `matplotlib`
- `pandas`
- `pydicom`
- `scikit-image`
- `tensorflow`
- `tqdm`

You can install them using pip:  `pip install opencv-python numpy matplotlib pandas pydicom scikit-image tensorflow tqdm`

## Usage

1. Make sure you have the RSNA Pneumonia Detection Challenge dataset available in the `/kaggle/input/rsna-pneumonia-detection-challenge/` directory.  This path is hardcoded in the notebook.
2. Run all cells of the notebook.
3. The trained model and evaluation results will be saved in the `output_predictions` directory and archived as `predictions.zip`.


## Key Parameters

- `input_size = 244`:  Size of the input images for the model.
- `BATCH_SIZE = 32`: Batch size for training.
- `DROPOUT_FACTOR = 0.5`: Dropout rate used in the feature extractor.
- `EPOCHS = 100`: Number of training epochs.

These parameters can be adjusted to experiment with different model configurations and training regimes.


## Further Improvements

- **Data Augmentation:** Adding data augmentation techniques (rotations, flips, etc.) could improve model robustness.
- **Model Architecture:**  Experimenting with different model architectures, such as pre-trained models like ResNet or Inception, might yield better performance.
- **Loss Function:**  Exploring alternative loss functions for bounding box regression, such as IoU loss, could improve localization accuracy.
- **Hyperparameter Tuning:** Fine-tuning hyperparameters, such as the learning rate and dropout rate, could optimize model performance.
