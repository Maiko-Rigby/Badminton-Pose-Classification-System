import pandas as pd
import numpy as np
from sklearn .model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.sequence import pad_sequences
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Function to plot heatmap
def plot_heatmap(data):
    plt.figure(figsize=(12, 10))
    corr_matrix = data.select_dtypes(include=[np.number]).corr()
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()

# Function to plot boxplots
def plot_boxplots(data, title):
    data.boxplot(figsize=(24, 12))
    plt.title(title)
    plt.xticks(rotation=90)
    plt.show()

# Load datasets
angle_observations = pd.read_csv("angle_results.csv")
smash_observations = pd.read_csv("time_series.csv")

# Change skilled and novice labels to 1:0 respectively
angle_observations['Category'] = smash_observations['Category'].map({'Skilled': 1, 'Novice': 0})
smash_observations['Category'] = smash_observations['Category'].map({'Skilled': 1, 'Novice': 0})

# Removing .mp4 from video name
angle_observations['Video'] = angle_observations['Video'].str.replace('.mp4', '', regex=False)
smash_observations['Video File'] = smash_observations['Video File'].str.replace('.mp4', '', regex=False)

# Checking for missing values
print(angle_observations.isnull().sum())
print(smash_observations.isnull().sum())

# Correct time as float
smash_observations['Time (seconds)'] = pd.to_numeric(smash_observations['Time (seconds)'], errors='coerce')

plot_boxplots(angle_observations, "Before Removing Outliers (Angle Dataset)")
plot_boxplots(smash_observations, "Before Removing Outliers (Smash Observations)")

# Outlier conditions after viewing the boxplots
outlier_conditions_smash = {
    'Wrist Flexion Acceleration': (smash_observations['Wrist Flexion Acceleration'] >= -2550) & (smash_observations['Wrist Flexion Acceleration'] <= 2500),
    'Shoulder Flexion Acceleration' :  (smash_observations['Shoulder Flexion Acceleration'] <= 2500),
    'Wrist Pronation Acceleration': (smash_observations['Wrist Pronation Acceleration'] >= -2000) & (smash_observations['Wrist Pronation Acceleration'] <= 2000),    
    'Elbow Flexion Acceleration': (smash_observations['Elbow Flexion Acceleration'] >= -3000) &(smash_observations['Elbow Flexion Acceleration'] <= 2500),    
    'Shoulder Abduction Acceleration': (smash_observations['Shoulder Abduction Acceleration'] >= -2000) & (smash_observations['Shoulder Abduction Acceleration'] <= 2000),    
    'Wrist Palmer Acceleration': (smash_observations['Wrist Palmar Acceleration'] >= -3500) & (smash_observations['Wrist Palmar Acceleration'] <= 2000),
}

outlier_conditions_angles = {
    'Shoulder Flexion Change': angle_observations['Shoulder Flexion Change'] <= 60,
    'Wrist Pronation Change': angle_observations['Wrist Pronation Change'] <= 85,
    'Elbow Flexion Change': angle_observations['Elbow Flexion Change'] <= 90,
    'Wrist Palmar Flexion Change': angle_observations['Wrist Palmar Flexion Change'] <= 80,
}

# Removing anomalies after reviewing boxplot
for col,condition in outlier_conditions_angles.items():
    angle_observations = angle_observations[condition]

for col, condition in outlier_conditions_smash.items():
    smash_observations = smash_observations[condition]

smash_observations = smash_observations.reset_index(drop=True)

plot_boxplots(angle_observations, "After Removing Outliers (Angle Dataset)")
plot_boxplots(smash_observations, "After Removing Outliers (Smash Observations)")


# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Creating new dataset containing the mean, std, mininum and maximum
agg_smash_observations = smash_observations.groupby('Video File').agg(
    {
        'Shoulder Flexion Angle': ['mean', 'std', 'min', 'max'],
        'Wrist Flexion Angle': ['mean', 'std', 'min', 'max'],
        'Wrist Pronation Angle': ['mean', 'std', 'min', 'max'],
        'Elbow Flexion Angle': ['mean', 'std', 'min', 'max'],
        'Shoulder Abduction Angle': ['mean', 'std', 'min', 'max'],
        'Wrist Palmar Angle': ['mean', 'std', 'min', 'max'],
        'Torso Rotation Angle': ['mean', 'std', 'min', 'max'],
        'Trunk Tilt Back Angle': ['mean', 'std', 'min', 'max'],
    }
)

# Flatten the multi-level columns
agg_smash_observations.columns = ['_'.join(col).strip() for col in agg_smash_observations.columns.values]

# Make Video File a column again
agg_smash_observations = agg_smash_observations.reset_index()

video_labels = smash_observations[['Video File', 'Category']].drop_duplicates(subset='Video File')
agg_smash_observations = agg_smash_observations.merge(video_labels, on='Video File')

# Removing duplicates
agg_smash_observations = agg_smash_observations.drop_duplicates()

plot_boxplots(agg_smash_observations, "Before Removing Outliers (Aggregated Dataset)")

# Define outlier conditions 
outlier_conditions_agg_smash = {
    'Shoulder Flexion Angle_max': (agg_smash_observations['Shoulder Flexion Angle_max'] <= 75),
    'Wrist Flexion Angle_min': (agg_smash_observations['Wrist Flexion Angle_min'] >= 45),
    'Elbow Flexion Angle_max': (agg_smash_observations['Elbow Flexion Angle_max'] >= 135),
    'Shoulder Abduction Angle_min': (agg_smash_observations['Shoulder Abduction Angle_min'] <= 55),
    'Wrist Palmar Angle_min' : (agg_smash_observations['Wrist Palmar Angle_min'] > 50),
    "Wrist Pronation Angle_max" : (agg_smash_observations["Wrist Pronation Angle_max"] <= 85),

}

# Removing anomalies after reviewing boxplot
for col, condition in outlier_conditions_agg_smash.items():
    agg_smash_observations = agg_smash_observations[condition]

plot_boxplots(agg_smash_observations, "Before Removing Outliers (Aggregated Dataset)")

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def test_train_prep(data, target_feature='Category'):
    # Splitting examples and labels
    examples = data.drop(columns=target_feature).to_numpy()
    labels = data[target_feature].to_numpy()

    # Test Train Split
    non_test_examples, test_examples, non_test_labels, test_labels = train_test_split(
        examples,
        labels,
        test_size=0.4,
        random_state=99,
        shuffle=True,
        stratify=labels
    )

    # Scaling data
    scaler = StandardScaler()
    scaler.fit(non_test_examples)
    non_test_examples = scaler.transform(non_test_examples)
    test_examples = scaler.transform(test_examples)

    return non_test_examples, test_examples, non_test_labels, test_labels


def get_lstm_data():
     # Make arrays for the data and label
    lstm_data = []
    lstm_labels = []

    # Group the data
    grouped = smash_observations.groupby('Video File')


    max_time_steps = 0
    for video_name, group in grouped:
        group = group.sort_values('Time (seconds)')
        features = group[[
            'Shoulder Flexion Angle', 'Wrist Flexion Angle', 'Wrist Pronation Angle',
            'Elbow Flexion Angle', 'Shoulder Abduction Angle', 'Wrist Palmar Angle',
            'Torso Rotation Angle', 'Trunk Tilt Back Angle'
        ]].to_numpy()
        
        # Extract the category
        label = group['Category'].iloc[0]
        
        # Add to the features and labels
        lstm_data.append(features)
        lstm_labels.append(label)

        max_time_steps = max(max_time_steps, features.shape[0])

    # Make all data have the same length
    padded_data = pad_sequences(lstm_data, padding='post', dtype='float32', maxlen=max_time_steps)
    
    lstm_data = np.array(padded_data)
    lstm_labels = np.array(lstm_labels)

    # Splitting examples and labels
    examples = padded_data
    labels = lstm_labels

    # Test Train Split
    non_test_examples, test_examples, non_test_labels, test_labels = train_test_split(
        examples,
        labels,
        test_size=0.4,
        random_state=99,
        shuffle=True,
        stratify=labels
    )

    # Scaling data
    scaler = StandardScaler()

    # Flatten the data to fit the scaler
    non_test_examples_flattened = non_test_examples.reshape(-1, non_test_examples.shape[2])
    test_examples_flattened = test_examples.reshape(-1, test_examples.shape[2])

    scaler.fit(non_test_examples_flattened)

    # Scaled data and reshaped back
    non_test_examples = scaler.transform(non_test_examples_flattened).reshape(non_test_examples.shape)
    test_examples = scaler.transform(test_examples_flattened).reshape(test_examples.shape)

    return non_test_examples, test_examples, non_test_labels, test_labels

def get_agg_data():
    return agg_smash_observations

def get_angle_data():
    return angle_observations
