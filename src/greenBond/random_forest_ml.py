import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.ensemble  import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, classification_report, confusion_matrix,
                             ConfusionMatrixDisplay)

BASE_DIR = Path(__file__).resolve().parent.parent.parent

class TrainRandomForest:
    def __init__(self, X_train_scaled, X_test_scaled, y_train, y_test):
        self.X_train_scaled = X_train_scaled
        self.X_test_scaled = X_test_scaled
        self.y_train = y_train
        self.y_test = y_test

        
    def train_random_forest(self, eigen_vals, eigen_vecs):
        #List of (eigenvalue, eigenvector) tuples
        eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]    
        
        #Sort the (eigenvalue, eigenvector) tuples from high to low
        #Note: lambda is an anonymous callable function; k is parameter of function
        eigen_pairs.sort(reverse=True)
        w = np.hstack((eigen_pairs[0][1][:, np.newaxis],
                       eigen_pairs[1][1][:, np.newaxis]))
        
        # eigen_pairs.sort(reverse=True)
        X_train_pca = self.X_train_scaled.dot(w)
        X_test_pca = self.X_test_scaled.dot(w)
        
        # Training The Model
        # Random Forest
        self.rf_model = RandomForestClassifier()
        self.rf_model.fit(X_train_pca, self.y_train)
        
        self.y_pred = self.rf_model.predict(X_test_pca)
    
    def calculate_confusion_matrix(self):
        return confusion_matrix(self.y_test, self.y_pred)
        
    def display_confusion_matrix(self):
        cm = self.calculate_confusion_matrix()
        cm_display = ConfusionMatrixDisplay(confusion_matrix = cm)
        
        
        cm_display.plot()
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.savefig(BASE_DIR / 'confusion_matrix.png', dpi=300)
        plt.show()
        
    def calculate_metrics(self):
        # Metrics
        accuracy = accuracy_score(self.y_test, self.y_pred)
        
        precision = precision_score(self.y_test, self.y_pred, average='weighted')
        recall = recall_score(self.y_test, self.y_pred, average='weighted')
        f1 = f1_score(self.y_test, self.y_pred, average='weighted')
        
        return accuracy, precision, recall, f1