#  ``` Q&A ```

```
1.which algorithm use and why not another one ?
=> We chose Random Forest because it is a practical starting point for classifying our 63 numerical hand-landmark features—not because other algorithms are bad or Random Forest is guaranteed to be the best.
   Different algorithms learn patterns in different ways. Let's compare them in the context of your project.
      1. Decision Tree
      ->Random Forest is an ensemble of decision trees, helping reduce the risk of relying on one tree's decisions.
      2.Support Vector Machine (SVM)
      ->It can be an excellent option for small datasets; it simply wasn't the algorithm implemented in our project.
 ```

```
2.how many approaches to use this application?

=> <1> Uses a local webcam with OpenCV (predict_webcam.py) to continuously recognize hand postures and predict Marathi characters.
  <2>The project includes a data collection feature to gather hand-posture samples for retraining and improving the model.
```

```
3.How you train this model?

=>We train our model using supervised learning. First, we collect labeled hand-posture samples, 
  extract and normalize 63 features from MediaPipe's 21 hand landmarks, 
  and train a Random Forest classifier using Scikit-learn. Finally,
  we save the trained model and label encoder so they can be reused for gesture recognition.
```
```
4.what is Joblib?
=>Joblib is a Python library that helps us save and load trained machine-learning models. In SignBridge AI, 
we use it to store the trained Random Forest classifier and Label Encoder so that the application can load them and 
predict Marathi characters without retraining the model every time.
```
```
5.how you use Training and testing in your project?
=> We use supervised learning to train our Random Forest model on labeled hand-landmark features.
 We normally reserve 20% of the data for testing and train on the remaining 80%.
The testing data helps us evaluate predictions on unseen examples. However, our current dataset
does not meet the minimum sample requirement for evaluation, so we need to collect more data before reporting reliable accuracy.
```

