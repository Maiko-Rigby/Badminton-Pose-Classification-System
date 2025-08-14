from data_preparation import test_train_prep, get_agg_data, get_angle_data
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score, accuracy_score
import seaborn as sns

def train_test(data):
    # Get testing and training data
    non_test_examples, test_examples, non_test_labels, test_labels = test_train_prep(data)

    # Define all hyperparameters for testing
    hyperparameters = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False],
        'max_features': ['auto', 'sqrt', 'log2']
    }

    # Initialise RF model and GridSearch
    RF = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(RF, hyperparameters, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
    grid_search.fit(non_test_examples, non_test_labels)

    print("Best Parameters:\n", grid_search.best_params_)

    # Best model
    RF_best = grid_search.best_estimator_

    # Testing using test data
    y_pred = RF_best.predict(test_examples)

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
    fpr, tpr, _ = roc_curve(test_labels, RF_best.predict_proba(test_examples)[:, 1])
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

train_test(get_agg_data())
train_test(get_angle_data())