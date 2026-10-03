# Face-Recognition-using-deep-learning
The objective of this project is to automatically recognize human emotions from facial images using deep learning models. The system classifies images into seven emotion categories: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise.

 # Introduction
Facial expressions are one of the most important forms of non-verbal communication. Emotion recognition systems are widely used in healthcare, education, human-computer interaction, surveillance, and customer experience analysis.

Traditional machine learning methods require manual feature extraction, whereas deep learning models automatically learn facial features directly from images, resulting in better performance.

# Objectives

The main objectives of this project are:

1)Build a Custom CNN model from scratch.

2)Implement transfer learning using MobileNetV2

3)Implement EfficientNetB0 for improved performance.

4)Compare the performance of all models.

5)Evaluate models using accuracy, precision, recall, F1-score, and confusion matrix
                                                                                                                                                                                                                                                                                                                                                                                    
# Analysis
 # Custom CNN
*Lowest accuracy.

*Useful as a baseline model.

*Simpler architecture but weaker feature extraction.

# MobileNetV2

*Highest precision.

*Lightweight and fast.

*Excellent for deployment on low-resource devices.

*Slightly lower F1-score than Custom CNN due to lower recall.

# EfficientNetB0

*Highest accuracy (60.30%).

*Highest recall (45.81%).

*Highest F1-score (0.5670).

*Best balance between precision and recall.

*Best overall emotion recognition performance.

# Best Model
Based on standard evaluation metrics, EfficientNetB0 is the best model because it achieves:

Highest Accuracy: 60.30%  
Highest Recall: 45.81%

Highest F1-Score: 0.5670

The F1-score is especially important in your emotion dataset because the classes are imbalanced (e.g., disgust has far fewer samples than happy). A higher F1-score indicates better overall classification performance across classes.
# Report Conclusion
Three deep learning models were evaluated for facial emotion recognition: a Custom CNN, MobileNetV2, and EfficientNetB0. Experimental results showed that EfficientNetB0 achieved the best overall performance with an accuracy of 60.30%, precision of 74.41%, recall of 45.81%, and F1-score of 0.5670. Therefore, EfficientNetB0 was selected as the final model for deployment in the real-time emotion recognition system. The trained model was saved and integrated with OpenCV-based face detection to classify emotions from live video streams in real time.
