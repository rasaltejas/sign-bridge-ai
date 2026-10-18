# Improving Prediction Reliability in SignBridge AI

**Prepared:** October 18, 2026  
**Project:** SignBridge AI — Marathi hand-posture recognition  
**Purpose:** Summarize the recent model update, the observed result, and the
work still needed to establish measured accuracy.

## Executive summary

The classifier was retrained after the training CSV was restored and updated.
The current dataset contains 87 landmark samples across all 26 posture classes.
The application loaded the newly trained model, and an informal check was
reported to produce correct predictions. This is encouraging, but it is not a
measured accuracy result: the number and independence of test attempts were not
recorded, and the training script could not run a holdout evaluation with the
current sample counts.

**No test-accuracy percentage is available.** The next priority is collecting
more varied examples for every posture and evaluating on examples not used to
train the model.

## What was changed

1. **Restored the missing training examples.** The original 32 samples for
   P11–P26 were restored from a saved CSV snapshot after they had been removed.
   The current file contains all classes from P01 through P26.
2. **Kept the added P01 and P02 examples.** The final dataset has 11 P01 samples
   and 15 P02 samples, alongside the restored samples and other existing data.
3. **Retrained the classifier from the current CSV.** The updated model was
   written to `models/signbridge_model.joblib`, with the matching class encoder
   and training metrics.
4. **Refreshed the running app.** The Streamlit app was reloaded so that it
   would use the newly trained model rather than the older model artifact.
5. **Checked the live result informally.** The user reported that the tested
   predictions were correct. The trial count and test conditions were not
   recorded, so this observation is useful as a demo check, not as a benchmark.

## How the current prediction pipeline works

- MediaPipe detects 21 hand landmarks from the camera image.
- Each landmark contributes x, y, and z values, producing 63 numeric features.
- The features are normalized relative to the wrist and scaled before
  classification.
- A scikit-learn Random Forest with 400 trees predicts one of the trained
  posture classes, P01–P26.
- The app maps the predicted class to its configured Devanagari character and
  warns when the top model score is below the configured threshold.

The displayed score is a model score, not a calibrated probability that the
prediction is correct.

## Dataset and evaluation status

| Measure | Current result |
|---|---:|
| Training samples | 87 |
| Posture classes | 26 |
| Samples in the smallest class | 2 |
| Classes with only 2 samples | 21 |
| Holdout test accuracy | Not evaluated |

Only P01–P05 have more than two examples: P01 has 11, P02 has 15, P03 has 7,
P04 has 5, and P05 has 7. The remaining 21 classes have two examples each.
Because the training script requires at least five examples in every class
before creating a stratified holdout split, it trained on all 87 samples and
recorded the test accuracy as “Not evaluated.” This value must not be presented
as either zero accuracy or perfect accuracy.

The quick live check and the training evaluation answer different questions:
the quick check indicates that the updated app behaved as expected for the
tested postures, while an evaluation score requires a documented set of
previously unseen test examples.

## How to improve reliability further

1. **Collect more examples for every class.** Aim for at least 30 varied,
   correctly labelled samples per class as an initial target. The project
   recommends this as a starting point, not as a guarantee of accuracy.
2. **Capture natural variation.** Vary lighting, background, hand distance,
   rotation, and small position changes while keeping the intended posture
   correct. Use multiple recording sessions where possible.
3. **Balance the dataset.** Add examples preferentially to the classes with
   fewer samples. Balanced class weights in the current Random Forest help
   reduce bias toward larger classes, but cannot replace missing examples.
4. **Create a trustworthy test set.** Reserve new examples that are never used
   for training. Prefer separating by recording session or capture batch so
   nearly identical frames do not appear in both training and test data.
5. **Report more than one metric.** Once each class has enough examples,
   evaluate overall accuracy, per-class precision and recall, macro F1, and a
   confusion matrix. These reveal errors hidden by a single overall score.
6. **Review misclassified examples.** Check whether each error came from a
   mislabeled sample, an unclear posture, poor hand detection, or two visually
   similar classes. Correct the data and retrain before changing model settings.
7. **Record repeatable demo results.** For a presentation check, write down the
   true posture, predicted class/character, score, lighting, and whether the
   sample was part of training. Do not describe training samples as independent
   test successes.

## Conclusion

The immediate update ensured the classifier was trained on the restored,
26-class dataset and that the live app loaded that updated model. The reported
correct demo predictions are a positive functional check. However, the dataset
is still sparse and uneven, so the model's real-world generalization has not
yet been established. More diverse per-class samples and an independent test
set are required before making a numerical accuracy claim.

## Suggested presentation summary

> “We updated the SignBridge classifier using 87 hand-landmark samples across
> all 26 posture classes, restored the missing class examples, retrained the
> 400-tree Random Forest, and refreshed the live app. The tested demo
> predictions were reported correct. We are not claiming a percentage
> accuracy, because 21 classes currently have only two samples each and the
> training script therefore did not run a holdout test. Our next step is to
> collect varied examples for every class and evaluate on a separate,
> session-independent test set.”
