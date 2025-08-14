from data_preparation import test_train_prep, get_agg_data, get_angle_data
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from xgboost import plot_importance

def train_test(data, feature_names):
    # Get testing and training data
    non_test_examples, test_examples, non_test_labels, test_labels = test_train_prep(data)

    # Define all hyperparameters for testing
    hyperparameters = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'gamma': [0, 0.1, 0.2],
        'reg_alpha': [0, 0.1, 1],
        'reg_lambda': [1, 10, 100],
        'scale_pos_weight': [1, 2, 5]
    }

    # Initialise XGB model and GridSearch
    xgb_model = xgb.XGBClassifier(random_state=42)
    grid_search = GridSearchCV(xgb_model, hyperparameters, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
    grid_search.fit(non_test_examples, non_test_labels)

    print("Best Parameters:\n", grid_search.best_params_)

    # Best model
    XGB_best = grid_search.best_estimator_

    # Testing using test data
    y_pred = XGB_best.predict(test_examples)

    # Confusion Matrix
    cm = confusion_matrix(test_labels, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # Plotting Confusion Matrix
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Novice', 'Skilled'], yticklabels=['Novice', 'Skilled'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

    # ROC Curve and AUC
    fpr, tpr, _ = roc_curve(test_labels, XGB_best.predict_proba(test_examples)[:, 1])
    roc_auc = auc(fpr, tpr)

    # Plotting ROC curve
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

    # Evaluation Metrics
    accuracy = accuracy_score(test_labels, y_pred)
    precision = precision_score(test_labels, y_pred)
    recall = recall_score(test_labels, y_pred)
    f1 = f1_score(test_labels, y_pred)

    # Print Evaluation Metrics
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")

    # Get feature importance
    feature_important = XGB_best.get_booster().get_score(importance_type='weight')
    # Map 'f0', 'f1', etc., to your feature names
    mapping = {f"f{i}": feature_names[i] for i in range(len(feature_names))}

    # Rename the keys with actual feature names
    feature_important_named = {mapping.get(k, k): v for k, v in feature_important.items()}

    # Convert to DataFrame and sort by importance
    data = pd.DataFrame(data=list(feature_important_named.values()), 
                        index=list(feature_important_named.keys()), 
                        columns=["score"]).sort_values(by="score", ascending=False)

    data = data[~data.index.isin(['Video', 'Category', 'Video File'])]

    # Display the importance
    data.nlargest(40, columns="score").plot(kind='barh', figsize=(20,10))
    plt.title('Top Feature Importances')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.gca().invert_yaxis() 
    plt.show()

# Labels for importance
features_agg = [
    'Video File',
    'Shoulder Flexion Angle_mean', 'Shoulder Flexion Angle_std', 'Shoulder Flexion Angle_min', 
    'Shoulder Flexion Angle_max', 'Wrist Flexion Angle_mean', 'Wrist Flexion Angle_std', 
    'Wrist Flexion Angle_min', 'Wrist Flexion Angle_max', 'Wrist Pronation Angle_mean', 
    'Wrist Pronation Angle_std', 'Wrist Pronation Angle_min', 'Wrist Pronation Angle_max', 
    'Elbow Flexion Angle_mean', 'Elbow Flexion Angle_std', 'Elbow Flexion Angle_min', 
    'Elbow Flexion Angle_max', 'Shoulder Abduction Angle_mean', 'Shoulder Abduction Angle_std', 
    'Shoulder Abduction Angle_min', 'Shoulder Abduction Angle_max', 'Wrist Palmar Angle_mean', 
    'Wrist Palmar Angle_std', 'Wrist Palmar Angle_min', 'Wrist Palmar Angle_max', 
    'Torso Rotation Angle_mean', 'Torso Rotation Angle_std', 'Torso Rotation Angle_min', 
    'Torso Rotation Angle_max', 'Trunk Tilt Back Angle_mean', 'Trunk Tilt Back Angle_std', 
    'Trunk Tilt Back Angle_min', 'Trunk Tilt Back Angle_max', 'Category'
]

features_angles = [
    'Video', 'Category', 'Shoulder Flexion Change', 'Wrist Flexion Change', 
    'Wrist Pronation Change', 'Elbow Flexion Change', 'Shoulder Abduction Change', 
    'Wrist Palmar Flexion Change', 'Torso Rotation Change', 'Trunk Tilt Backward Change'
]

train_test(get_agg_data(), features_agg)
train_test(get_angle_data(), features_angles)