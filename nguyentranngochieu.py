import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Load and filter dataset
def load_and_filter_data(train_path, test_path, activities):
    try:
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
    except FileNotFoundError:
        raise FileNotFoundError("Dataset files not found. Please check the file paths.")
    
    # Filter for specified activities
    train_data = train_data[train_data['Activity'].isin(activities)]
    test_data = test_data[test_data['Activity'].isin(activities)]
    
    if train_data.empty or test_data.empty:
        raise ValueError("Filtered dataset is empty. Check the activity names.")
    
    return train_data, test_data

# Preprocess data
def preprocess_data(train_data, test_data):
    X_train = train_data.drop(['Activity', 'subject'], axis=1).values
    y_train = train_data['Activity'].values
    X_test = test_data.drop(['Activity', 'subject'], axis=1).values
    y_test = test_data['Activity'].values
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Encode labels
    label_map = {'WALKING': 0, 'WALKING_UPSTAIRS': 1, 'WALKING_DOWNSTAIRS': 2}
    y_train_encoded = np.array([label_map[label] for label in y_train])
    y_test_encoded = np.array([label_map[label] for label in y_test])
    
    return X_train_scaled, y_train_encoded, X_test_scaled, y_test_encoded

# Train HMM models
def train_hmm_models(X_train, y_train, n_components=3):
    models = {}
    for activity in range(3):  # 3 activities
        # Select data for this activity
        X_activity = X_train[y_train == activity]
        if len(X_activity) == 0:
            raise ValueError(f"No data for activity {activity}. Check dataset filtering.")
        # Train HMM
        model = hmm.GaussianHMM(n_components=n_components, covariance_type="full", n_iter=100)
        model.fit(X_activity)
        models[activity] = model
    return models

# Particle Filter implementation
class ParticleFilter:
    def __init__(self, n_particles, n_states, transition_prob, observation_model):
        self.n_particles = n_particles
        self.n_states = n_states
        self.particles = np.random.choice(n_states, size=n_particles, p=[1/n_states]*n_states)
        self.weights = np.ones(n_particles) / n_particles
        self.transition_prob = transition_prob
        self.observation_model = observation_model
    
    def predict(self):
        # Predict next state based on transition probabilities
        for i in range(self.n_particles):
            self.particles[i] = np.random.choice(self.n_states, p=self.transition_prob[self.particles[i]])
    
    def update(self, observation):
        # Update weights based on observation likelihood
        for i in range(self.n_particles):
            state = self.particles[i]
            likelihood = self.observation_model[state].score_samples(observation.reshape(1, -1))[0]
            self.weights[i] *= np.exp(likelihood)
        self.weights += 1.e-300  # Avoid zero weights
        self.weights /= np.sum(self.weights)
    
    def resample(self):
        # Resample particles based on weights
        indices = np.random.choice(self.n_particles, size=self.n_particles, p=self.weights)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.n_particles) / n_particles
    
    def estimate(self):
        # Estimate the most likely state
        return np.argmax(np.bincount(self.particles, weights=self.weights))

# Main execution
if __name__ == "__main__":
    # Paths to dataset
    train_path = '../input/human-activity-recognition-with-smartphones/train.csv'
    test_path = '../input/human-activity-recognition-with-smartphones/test.csv'
    
    # Activities to keep
    activities = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS']
    
    try:
        # Load and preprocess data
        train_data, test_data = load_and_filter_data(train_path, test_path, activities)
        X_train, y_train, X_test, y_test = preprocess_data(train_data, test_data)
        
        # Train HMM models
        hmm_models = train_hmm_models(X_train, y_train, n_components=3)
        
        # Define transition probabilities (example)
        transition_prob = np.array([
            [0.8, 0.1, 0.1],  # From WALKING
            [0.1, 0.8, 0.1],  # From WALKING_UPSTAIRS
            [0.1, 0.1, 0.8]   # From WALKING_DOWNSTAIRS
        ])
        
        # Initialize Particle Filter
        pf = ParticleFilter(n_particles=1000, n_states=3, transition_prob=transition_prob, observation_model=hmm_models)
        
        # Test Particle Filter on test data
        predictions = []
        for t in range(min(100, len(X_test))):  # Limit to 100 samples for demo
            pf.predict()
            pf.update(X_test[t])
            pf.resample()
            predicted_state = pf.estimate()
            predictions.append(predicted_state)
        
        # Evaluate accuracy
        true_labels = y_test[:len(predictions)]
        accuracy = np.mean(predictions == true_labels)
        print(f"Accuracy with Particle Filter: {accuracy:.4f}")
        
        # Plot results
        plt.figure(figsize=(12, 6))
        plt.plot(true_labels, label='True Labels', marker='o')
        plt.plot(predictions, label='Predicted Labels', marker='x')
        plt.title('Activity Recognition with HMM and Particle Filter')
        plt.xlabel('Sample Index')
        plt.ylabel('Activity (0: WALKING, 1: UPSTAIRS, 2: DOWNSTAIRS)')
        plt.legend()
        plt.show()
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")