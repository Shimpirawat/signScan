## signScan (Traffic Sign Classifier)

### Overview
The Traffic Sign Classifier is a machine learning model based on Convolutional Neural Networks (CNN) and implemented using TensorFlow. The purpose of this project is to develop a robust system capable of recognizing and classifying traffic signs commonly found on roads. The model can be used to enhance road safety by providing real-time predictions of traffic signs from images or video streams.

### Dataset
The data used is taken from [kaggle](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign) website and the dataset used for training and evaluating the Traffic Sign Classifier is sourced from:

GTSRB Dataset: It contains a Train folder that has traffic sign images in 43 different classes, a Test folder that has over 12,000 images for testing purposes. A test.csv file that contains the path of the test images along with their respective classes.

### Libraries and Tools
The Traffic Sign Classifier is written Python for developing the model and the associated web interface., using the following libraries and tools:

- Pandas: Used for data manipulation and analysis during data preprocessing.

- NumPy: Essential for numerical operations and handling multidimensional arrays.

- TensorFlow: The deep learning framework employed to build and train the CNN model.

- Scikit-learn: Utilized for data preprocessing, model evaluation, and performance metrics.

- Matplotlib: Employed for data visualization and plotting graphs.

- Tkinter: Used to create a simple web-based interface for interacting with the model.

- PIL: Necessary for image processing tasks.

### Algorithms
- The Traffic Sign Classifier uses the Convolutional Neural Network (CNN) algorithm. CNNs are a type of deep learning model that are particularly effective for image recognition tasks. They can automatically learn relevant features from images through convolutional layers and pooling operations, leading to higher accuracy in classification tasks.

The architecture of the CNN model used for the Traffic Sign Classifier consists of multiple convolutional layers, followed by fully connected layers, and ends with a softmax layer for the final classification output.

### GUI
The GUI provides a simple and intuitive way for users to interact with the Traffic Sign Classifier model. Users can perform the following actions using the GUI:

Upload an image: Users can upload an image of a traffic sign using the "Browse" button.

Classify the traffic sign: Once an image is uploaded, users can click the "Classify" button to let the model predict the type of traffic sign.

View prediction: The GUI displays the predicted class label along with the confidence score.

### Conclusion
The Traffic Sign Classifier using CNN achieves high accuracy of about 98.2% in recognizing and classifying traffic signs. After extensive training on the combined GTSRB, the model demonstrates robust performance even in challenging real-world scenarios. The CNN-based approach outperforms traditional image processing methods, making it a reliable solution for enhancing road safety and automating traffic sign recognition.
