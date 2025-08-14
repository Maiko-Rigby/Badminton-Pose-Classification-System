from data_preparation import get_lstm_data
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

def create_lstm_model(input_shape, units=50, dropout_rate=0.2, learning_rate=0.001):
    model = Sequential()
    model.add(LSTM(units=units, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units=units))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_test():
    # Get testing and training data
    non_test_examples, test_examples, non_test_labels, test_labels = get_lstm_data()

    # Create the hyperparameter arrays
    units_options = [50, 100]
    dropout_options = [0.2, 0.3]
    learning_rate_options = [0.001, 0.01]
    batch_size_options = [32, 64]
    epochs_options = [10, 20]

    # Prepare the data
    input_shape = non_test_examples.shape[1:]  # (time_steps, num_features)

    # Initialize variables to store best model and results
    best_accuracy = 0
    LSTM_best = None
    best_params = {}

    # Finding best model
    for units in units_options:
        for dropout_rate in dropout_options:
            for learning_rate in learning_rate_options:
                for batch_size in batch_size_options:
                    for epochs in epochs_options:
                        print(f"Training with units={units}, dropout_rate={dropout_rate}, "
                            f"learning_rate={learning_rate}, batch_size={batch_size}, epochs={epochs}")
                        
                        # Initialise model with the current hyperparameters
                        model = create_lstm_model(input_shape, units=units, dropout_rate=dropout_rate, learning_rate=learning_rate)
                        
                        # Fit the model
                        model.fit(non_test_examples, non_test_labels, epochs=epochs, batch_size=batch_size, verbose=0)
                        
                        # Evaluate the model
                        test_predictions = model.predict(test_examples, batch_size=batch_size)
                        test_predictions = (test_predictions > 0.5).astype(int)  # Threshold at 0.5 for binary classification
                        
                        accuracy = accuracy_score(test_labels, test_predictions)
                        precision = precision_score(test_labels, test_predictions)
                        recall = recall_score(test_labels, test_predictions)
                        f1 = f1_score(test_labels, test_predictions)

                        print(f"Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1 Score: {f1:.3f}")

                        # Update the best model if needed
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            LSTM_best = model
                            best_params = {
                                'units': units,
                                'dropout_rate': dropout_rate,
                                'learning_rate': learning_rate,
                                'batch_size': batch_size,
                                'epochs': epochs
                            }

    print("Best Hyperparameters: ", best_params)

    # Best model
    LSTM_best_predictions = LSTM_best.predict(test_examples)

    # Testing using test data
    y_pred = LSTM_best_predictions
    y_pred_classes = (y_pred > 0.5).astype(int)

    # Confusion Matrix
    cm = confusion_matrix(test_labels, y_pred_classes)
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
    fpr, tpr, _ = roc_curve(test_labels, LSTM_best.predict(test_examples).ravel())
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

    # Predict probabilities
    y_pred_probs = LSTM_best.predict(test_examples).ravel()

    # Convert probs to 0 or 1
    y_pred = (y_pred_probs > 0.5).astype(int)


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

train_test()