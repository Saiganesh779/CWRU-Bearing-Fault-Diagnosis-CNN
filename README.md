# CWRU Bearing Fault Diagnosis Using CNN

A deep learning project for classifying bearing vibration data into different fault conditions using a Convolutional Neural Network (CNN).

The project uses preprocessed vibration data from the Case Western Reserve University (CWRU) bearing dataset and performs 10-class bearing fault classification using TensorFlow/Keras.

## Project Overview

Bearing faults can produce characteristic vibration patterns that can be used for condition monitoring and predictive maintenance.

In this project, vibration samples are represented as 32 × 32 input arrays and classified into 10 bearing conditions using a CNN.

### Fault Classes

The model classifies the following conditions:

- Normal
- IR_007
- IR_014
- IR_021
- OR_007
- OR_014
- OR_021
- Ball_007
- Ball_014
- Ball_021

Where:

- IR = Inner Race Fault
- OR = Outer Race Fault
- Ball = Ball Element Fault

## Dataset

The project uses a preprocessed CWRU bearing vibration dataset.

Dataset characteristics:

- Total samples: 4,600
- Number of classes: 10
- Samples per class: 460
- Input size: 32 × 32
- Balanced classes

The dataset file used by the program is:

```text
CWRU_48k_load_1_CNN_data.npz
```

The dataset is not included in this repository. Please obtain the dataset from its original source and place the required `.npz` file in the project directory.

## Model Architecture

The CNN consists of:

1. Conv2D – 32 filters
2. Batch Normalization
3. Max Pooling
4. Conv2D – 64 filters
5. Batch Normalization
6. Max Pooling
7. Conv2D – 128 filters
8. Batch Normalization
9. Global Average Pooling
10. Dense layer – 64 neurons
11. Dropout – 0.3
12. Softmax output layer

The model is trained using:

- Optimizer: Adam
- Learning rate: 0.001
- Loss: Sparse Categorical Crossentropy
- Batch size: 32
- Maximum epochs: 50
- Early stopping
- Learning-rate reduction on plateau

## Data Split

The original implementation uses a stratified:

- 70% training
- 15% validation
- 15% test

split.

The final test set contains 690 samples.

## Results

The CNN achieved:

**Test Accuracy: 99.28%**

The test classification report produced an overall F1-score of approximately 0.99.

The confusion matrix showed only five misclassified samples in the 690-sample test set.

### Important Evaluation Note

The reported 99.28% accuracy comes from a randomly stratified sample-level split.

Because the processed `.npz` file does not contain original recording/source IDs, recording-level independence between training and test samples could not be verified.

Therefore, the result should be interpreted as:

> **99.28% accuracy on a randomly held-out test set**

and not as a guaranteed real-world or unseen-recording accuracy.

An additional ordered holdout experiment produced a substantially lower accuracy of 11.96%. However, because the dataset does not provide recording IDs, this experiment cannot conclusively determine whether the samples represent different physical recordings.

Further recording-level validation is required to establish true cross-recording generalization.

## Project Structure

```text
CWRU-Bearing-Fault-Diagnosis-CNN/
│
├── model.py
├── generalization.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── bearing_fault_cnn.keras
└── bearing_fault_label_encoder.joblib
```

## Requirements

- Python 3.11
- NumPy
- TensorFlow
- Scikit-learn
- Joblib

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Model

Place:

```text
CWRU_48k_load_1_CNN_data.npz
```

in the project directory.

Then run:

```bash
python model.py
```

The program will:

1. Load the dataset
2. Encode the class labels
3. Prepare the CNN input
4. Split the dataset
5. Train the CNN
6. Evaluate the model
7. Generate a classification report
8. Generate a confusion matrix
9. Save the trained model
10. Save the label encoder

## Saved Files

After training, the following files are generated:

```text
bearing_fault_cnn.keras
bearing_fault_label_encoder.joblib
```

### Trained Model

```text
bearing_fault_cnn.keras
```

Contains the trained TensorFlow/Keras CNN.

### Label Encoder

```text
bearing_fault_label_encoder.joblib
```

Stores the mapping between class names and numerical labels.

## Generalization Experiment

The repository also contains:

```text
generalization.py
```

This script performs an additional ordered holdout experiment using the available dataset.

Run:

```bash
python generalization.py
```

This experiment should be treated as a supplementary evaluation because the processed dataset does not contain original recording IDs.

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Scikit-learn
- Joblib
- Convolutional Neural Networks
- Vibration-based fault diagnosis
- Predictive maintenance

## Applications

This type of bearing fault classification can be applied to:

- Predictive maintenance
- Condition monitoring
- Rotating machinery diagnostics
- Industrial equipment monitoring
- Automotive component health monitoring

## Limitations

- The dataset is preprocessed into 32 × 32 representations.
- Original recording/source IDs are not available in the processed dataset.
- Recording-level data leakage cannot be completely ruled out.
- The reported 99.28% accuracy is based on a random sample-level split.
- Cross-dataset and real-world generalization has not been established.
- The model has not been validated on physical sensor data collected independently from the CWRU dataset.

## Future Improvements

- Perform recording-level train/test splitting using source IDs.
- Evaluate the model on an independent bearing dataset such as SUBF.
- Compare 2D CNN with 1D CNN approaches.
- Investigate raw vibration signals and time-frequency representations.
- Evaluate robustness under different loads and operating conditions.
- Perform cross-dataset validation.
- Deploy the model for real-time bearing condition monitoring.

## Conclusion

This project demonstrates the use of a CNN for multi-class bearing fault classification using vibration data.

The model achieved **99.28% accuracy on the randomly held-out CWRU test set**. Additional evaluation shows that the method requires more rigorous recording-level validation before its performance can be considered representative of generalization to completely unseen bearing recordings.

## Author

Saiganesh Nangunoori
