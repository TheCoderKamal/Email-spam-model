import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def find_data_file(filename):
    """Search for the data file in current and parent directories"""
    # First, try the current directory
    if os.path.exists(filename):
        return filename
    
    # Try the parent directory
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parent_file = os.path.join(parent_dir, filename)
    if os.path.exists(parent_file):
        return parent_file
    
    # Try looking in a 'data' subdirectory
    data_dir_path = os.path.join(parent_dir, 'data')
    data_file = os.path.join(data_dir_path, filename)
    if os.path.exists(data_file):
        return data_file
    
    raise FileNotFoundError(f"Could not find {filename} in current directory, parent directory, or data subdirectory")

def load_and_preprocess_data(test_size=0.2, random_state=3):
    try:
        # Find and load data file
        data_path = find_data_file('mail_data.csv')
        print(f"Loading data from: {data_path}")
        
        raw_mail_data = pd.read_csv(data_path)
        mail_data = raw_mail_data.where(pd.notnull(raw_mail_data), '')
        
        # Convert labels
        mail_data.loc[mail_data['Category'] == 'spam', 'Category'] = 0
        mail_data.loc[mail_data['Category'] == 'ham', 'Category'] = 1
        mail_data['Category'] = pd.to_numeric(mail_data['Category'])
        
        X = mail_data['Message']
        Y = mail_data['Category']
        
        # Split data
        return train_test_split(X, Y, test_size=test_size, random_state=random_state)
    
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        print("\nPlease ensure that 'mail_data.csv' is in one of these locations:")
        print(f"1. Same directory as the script: {os.path.abspath('.')}")
        print(f"2. Parent directory: {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}")
        print(f"3. In a 'data' subdirectory of the parent directory")
        sys.exit(1)

def compare_models(X_train, X_test, Y_train, Y_test):
    models = {
        'Logistic Regression': LogisticRegression(random_state=3),
        'Naive Bayes': MultinomialNB(),
        'SVM': SVC(kernel='linear', random_state=3),
        'Random Forest': RandomForestClassifier(random_state=3)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, Y_train)
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        results[name] = {
            'train_accuracy': accuracy_score(Y_train, train_pred),
            'test_accuracy': accuracy_score(Y_test, test_pred),
            'model': model
        }
        print(f"{name} - Train Accuracy: {results[name]['train_accuracy']:.4f}, Test Accuracy: {results[name]['test_accuracy']:.4f}")
    
    return results

def plot_model_comparison(results):
    names = list(results.keys())
    train_scores = [results[name]['train_accuracy'] for name in names]
    test_scores = [results[name]['test_accuracy'] for name in names]
    
    plt.figure(figsize=(10, 6))
    x = np.arange(len(names))
    width = 0.35
    
    plt.bar(x - width/2, train_scores, width, label='Training Accuracy')
    plt.bar(x + width/2, test_scores, width, label='Testing Accuracy')
    
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title('Model Comparison')
    plt.xticks(x, names, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

def tune_best_model(X_train, Y_train, model_name):
    print(f"\nTuning {model_name}...")
    
    if model_name == 'Logistic Regression':
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'max_iter': [100, 200, 300]
        }
        model = LogisticRegression(random_state=3)
        
    elif model_name == 'Random Forest':
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10]
        }
        model = RandomForestClassifier(random_state=3)
        
    elif model_name == 'SVM':
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto', 0.1, 1],
        }
        model = SVC(random_state=3)
        
    else:
        print(f"No hyperparameter tuning defined for {model_name}")
        return None, None

    print("Starting Grid Search... This might take a few minutes...")
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, Y_train)
    
    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_params_, grid_search.best_estimator_

def plot_learning_curves(model, X_train, Y_train):
    """Plot learning curves to show over/underfitting"""
    print("\nGenerating learning curves...")
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train, Y_train, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10))
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training score', color='blue')
    plt.plot(train_sizes, test_mean, label='Cross-validation score', color='red')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='red')
    
    plt.xlabel('Training Examples')
    plt.ylabel('Score')
    plt.title('Learning Curves')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

def main():
    print("Starting spam detection model analysis...")
    
    # Load and preprocess data
    X_train, X_test, Y_train, Y_test = load_and_preprocess_data()
    
    # Feature extraction
    print("\nPerforming feature extraction...")
    vectorizer = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
    X_train_features = vectorizer.fit_transform(X_train)
    X_test_features = vectorizer.transform(X_test)
    
    # Compare models
    print("\nComparing different models...")
    results = compare_models(X_train_features, X_test_features, Y_train, Y_test)
    plot_model_comparison(results)
    
    # Find best performing model
    best_model = max(results.items(), key=lambda x: x[1]['test_accuracy'])
    print(f"\nBest performing model: {best_model[0]}")
    print(f"Test accuracy: {best_model[1]['test_accuracy']:.4f}")
    
    # Tune best model
    best_params, tuned_model = tune_best_model(X_train_features, Y_train, best_model[0])
    
    # Only proceed with learning curves if we have a tuned model
    if tuned_model is not None:
        # Plot learning curves to demonstrate overfitting/underfitting
        plot_learning_curves(tuned_model, X_train_features, Y_train)
        
        # Test the tuned model
        if best_params is not None:
            print("\nEvaluating tuned model performance...")
            Y_pred = tuned_model.predict(X_test_features)
            final_accuracy = accuracy_score(Y_test, Y_pred)
            print(f"Final tuned model accuracy: {final_accuracy:.4f}")
            
            # Save the best model
            print("\nSaving the best model...")
            try:
                model_dir = 'saved_models'
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
                
                model_path = os.path.join(model_dir, 'spam_ham_model.pkl')
                vectorizer_path = os.path.join(model_dir, 'vectorizer.pkl')
                
                joblib.dump(tuned_model, model_path)
                joblib.dump(vectorizer, vectorizer_path)
                print(f"Model saved to {model_path}")
                print(f"Vectorizer saved to {vectorizer_path}")
            except Exception as e:
                print(f"Error saving model: {str(e)}")

if __name__ == "__main__":
    main()