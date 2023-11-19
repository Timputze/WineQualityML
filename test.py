#!/usr/bin/env python
# coding: utf-8

# ## Wine Quality Prediction Model

# ### Importing Libraries

#GPU Processing
#from numba import jit, cuda
import time

# Import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings("ignore")


# ### Loading Dataset

# Load the data from a file (e.g., CSV)
data = pd.read_csv(r"C:path")

# Display the first few rows of the data
print("First we have to load the data and examine it:")
print("\nThe first 10 rows of the data set: ") 
print(data.head(10))

# ### Statistical Data Analysis

#Information about the dataset
print("\nInformation about the type of data used in the data set: ")
print(data.info())


#Descriptive statistics of the dataset
print("\nStatistical description of the data used:")
print(data.describe())
print("\n")


#Number of missing values in each column
print('\nNumber of missing values in each column:')
print(data.isna().sum())


# ### Visual Analysis

# Check the correlation between variables
print("\nThe following is a representation of the correlation of the variables: \n")
plt.figure(figsize=(12, 10))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True)
plt.title("Correlation of variables")
plt.show()


# Creating a Count Plot for the 'quality' Column
print("Count Plot for the 'Quality' Column\n")
sns.countplot(x='quality', data=data)
plt.title("Count plot for Quality")
plt.show()


# Creating Bar Plots for Numerical Columns based on 'Quality
print("Bar Plots for all Numerical Columns based on 'Quality'\n")
numerical_columns = data.select_dtypes(include=['float64', 'int64'])
for i, col in enumerate(numerical_columns):
    plt.figure(i)
    sns.barplot(x='quality', y=col, data=data)
    plt.title('Quality for ' + str(col))
plt.show()

#Visualise the data from earlier using histograms
print("\nHere is a visualisation of the variables: \n")
data.hist(bins=25,figsize=(10,10))
plt.title("Histograms")
plt.show()


#Remove highly correlated variables
print("Variables too closely correlated:")
correlation_threshold = 0.7
corr_features = set()
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > correlation_threshold:
            colname = correlation_matrix.columns[i]
            corr_features.add(colname)
print(corr_features)


#Combine density and alcohol into a new variable, alcohol concentration
#Correct conversion factor for calculating alcohol concentration from density and alcohol content is 0.789
data['alcohol_concentration'] = data['alcohol'] * (data['density'] / 0.789)

#Drop the original variables
data = data.drop(['density', 'alcohol'], axis=1)

#Check the correlation again to ensure suitable variables for testing
print("\nThe following is a representation of the correlation of the variables after combining density and alcohol: \n")
plt.figure(figsize=[19, 10], facecolor='white')
sns.heatmap(data.corr(), annot=True)
plt.title("Correlation of variables after combining density and alcohol")
plt.show()


#Create Categorical Labels for the 'quality' Column
data['quality'] = pd.cut(data['quality'], bins=(2, 4, 7, 9), labels=['bad','good','excellent'])
y = data['quality']

#Separate the features (X) and target variable (y)
X = data.drop('quality', axis=1)

#Encode Target Variable
encoder = LabelEncoder()
y = encoder.fit_transform(y)

#Check Dataset for imbalance
print("##### WITHOUT SAMPLING #####")
unique, counts = np.unique(y, return_counts=True)
print(np.asarray((unique,counts)).T)

#Impement balanced samppling method to downsample for performance reasons
def balanced_subsample(y, size=None, random_state=None): 
    # returns a List with randomly chosen row numbers 
    subsample = [] 
    if size is None: 
        n_smp = y.value_counts().min() 
    else: 
        n_smp = int(size / len(y.value_counts().index)) 
    if not random_state is None: 
        np.random.seed(random_state) 
    for label in y.value_counts().index: 
        samples = y[y == label].index.values 
        index_range = range(samples.shape[0]) 
        indexes = np.random.choice(index_range, size=n_smp,replace=False) 
        subsample += samples[indexes].tolist() 
    return subsample

#Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#Use SMOTE to Upsample minority classes
smote = SMOTE(sampling_strategy='auto', random_state=42, n_jobs=-1)
X_train, y_train = smote.fit_resample(X_train, y_train)
print("##### RESAMPLING #####")
unique, counts = np.unique(y_train, return_counts=True)
print(np.asarray((unique,counts)).T)

#combine dataset (data, category labels)
combined = np.vstack((X_train.T, y_train))
combined = pd.DataFrame(combined)

#Downsample to 2000 due to performance
rows = balanced_subsample(combined.T[10],size=2000)
downsampled = combined.T.iloc[rows,:]
y_down = downsampled[10]
X_down = downsampled.drop(10, axis=1)
X_down.columns=list(X.columns)
print("##### DOWNSAMPLING #####")
unique, counts = np.unique(y_down, return_counts=True)
print(np.asarray((unique,counts)).T)

print('\nDimension of Train and Test Feature and target Variables')
print("Training Data:")
print(X_train.shape)
print(y_train.shape)
print("Testing Data")
print(X_test.shape)
print(y_test.shape)
print("Downsampled Data")
print(X_down.shape)
print(y_down.shape)

#print('Head of Datasets\n')
#print(X_train.head())
#print(X_down.head())
#print(X_test.head())


# Scale the features using StandardScaler
y_train = y_down #Use the downsampled dataset
scaler = StandardScaler()
X_train = scaler.fit_transform(X_down) #Use the downsampled dataset
X_test = scaler.transform(X_test)

# ### Cross Validation 
#@jit(target_backend='cuda')  
#Implement GridSearch
def plot_grid_search_2d(clf_name,cv_results, grid_param_1, grid_param_2, name_param_1, name_param_2):
    scores_mean = cv_results['mean_test_score']
    scores_mean = np.array(scores_mean).reshape(len(grid_param_2), len(grid_param_1))

    _, ax = plt.subplots(1,1)
    for idx, val in enumerate(grid_param_2):
        ax.plot(grid_param_1, scores_mean[idx,:], '-o', label=name_param_2 + ': ' + str(val))
    ax.set_title("Grid Search Scores" + str(clf_name), fontsize=12, fontweight='bold')
    ax.set_xlabel(name_param_1, fontsize=10)
    ax.set_ylabel('CV Average Validation Accuracy', fontsize=10)
    ax.legend(loc="best", fontsize=8)
    ax.grid('on')


# ### ML Models

# Create a dictionary to store classifiers
classifiers = {
    'Random Forest': RandomForestClassifier(criterion='entropy'),
    'XGBoost': XGBClassifier(objective='multi:softprob', eval_metric='merror'),
    'SVC': SVC(),
    #'Logistic Regression': LogisticRegression(solver='liblinear', penalty='l2'), #only for two class classification
    'Neural Network': MLPClassifier(activation='relu', batch_size='auto') 
}

#Different predefined hyperparameter values per allogrithm

param_grid_RF = {    
    'max_depth': [None, 1, 5, 10],
    'n_estimators': [1, 5, 10, 25, 50, 100, 150, 200],
    }

param_grid_XGB = {
    'max_depth': [None, 1, 5, 10],
    'n_estimators': [1, 5, 10, 25, 50, 100, 150, 200],
}

param_grid_SVC = {
    'C':[0.1, 0.2, 0.5, 0.75, 1],
    'kernel':['linear','rbf', 'poly']
}

param_grid_LR = {
    'C': [0.1, 0.2, 0.5, 0.75, 1],
    'max_iter': [1, 5, 10, 25, 50, 100, 150, 200]
}

param_grid_NN = {
    'hidden_layer_sizes': [1, 2, 5, 10],    
    'learning_rate': ['constant', 'adaptive'],
}

results = {}
modellist={}

#For each classifier take the predefined hyperparameters and perform a Gridsearch and Cross Validation
#Store and Print the results for further analysis
for clf_name, classifier in classifiers.items():

    starttime = time.time()

    if(clf_name == "Random Forest"):
        param_grid = param_grid_RF
    elif(clf_name == "XGBoost"):
        param_grid = param_grid_XGB
    elif(clf_name == "SVC"):
        param_grid = param_grid_SVC
    elif(clf_name == "Logistic Regression"):
        param_grid = param_grid_LR
    elif(clf_name == "Neural Network"):
        param_grid = param_grid_NN

    
    #Create report template
    report= pd.DataFrame(columns=['Model','Mean Acc. Training', 'Standard Deviation', 'Acc. Test'])
    
    print("\nStarting Gridsearch for: " + str(clf_name) + "\nUsing the following parameters: " + str(param_grid))       
    
    #Assign GridSearch on each classifier with respective hyperparameter list   
    CV_model = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=10)
    
    #Fit Data to each classifier and execute GridSearchCV()
    CV_model.fit(X_train, y_train)

    endtime = time.time()
    print(f"Completed Gridsearch witin {(endtime - starttime):.5f} seconds")
    print("Best Hyperparameters for " + str(clf_name) + ": " + str(CV_model.best_params_))

    #Assign best parameters to model (extra safety)
    classifier = classifier.set_params(**CV_model.best_params_)
    classifier.fit(X_train, y_train)

    #GridSearch automatically assigns the best hyperparameter combination to a classifier which is then aufgerufen by predict
    #accuracies = cross_val_score(classifier, X_train, y_train, scoring='accuracy', cv=10)
    #print(accuracies)
    #acc_mean = accuracies.mean()
    #print(acc_mean)
    #acc_std = accuracies.std()    
    #print(acc_std)
    #Cross_val_score is not needed as those metrics are listed in cv_results_

    #predict using provided test data set
    y_test_pred = classifier.predict(X_test)
    
    #The accuracy score provides an overall indication of how well the model is able to classify the samples correctly
    accuracy = accuracy_score(y_test, y_test_pred)
    print(f"{clf_name} Accuracy: {round(accuracy,2)*100}%")
    
    #Store Results and model
    results[clf_name] = accuracy  
    modellist[str(clf_name)]=classifier

    #Print confusion matricies
    print("\n")
    print(f"Confusion Matrix for {clf_name}:")
    print(confusion_matrix(y_test, y_test_pred))
    print("\n")
    print(f"Classification Report for {clf_name}:")
    print(f'{classification_report(y_test, y_test_pred)}\n')

    #Store data in prestructured report template
    report.loc[len(report)] = [str(clf_name),
        CV_model.cv_results_['mean_test_score'][CV_model.best_index_],
        CV_model.cv_results_['std_test_score'][CV_model.best_index_],
        accuracy]
    print(report.loc[len(report)-1])

    #Plot average validation accuracies for every hyperparameter (per allgo) 
    plot_model = CV_model.cv_results_
    param_name1 = list(param_grid.keys())[0]
    param_name2 = list(param_grid.keys())[1]
    param1_values = param_grid[param_name1]
    param2_values = param_grid[param_name2]
    plot_grid_search_2d(clf_name,plot_model, param1_values, param2_values, param_name1, param_name2)
    plt.title("Average validation accuracies for " + str(clf_name))
    plt.show()
    

################ FEATURE IMPOPRTANCE BY ALLGO ################
######### RANDOM FOREST FEATURE IMPORTANCE ################
model=modellist['Random Forest']
model.fit(X_train,y_train)
feature_importance = model.feature_importances_
feature_name = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'alcohol_concentration']
(pd.Series(feature_importance,index=feature_name).plot(kind='barh',xlabel='Features Importance',ylabel='Features',title='Feature Importance - Random Forest'))
plt.show()

######### XGBoost FEATURE IMPORTANCE ################
model=modellist['XGBoost']
model.fit(X_train,y_train)
feature_importance = model.feature_importances_
feature_name = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'alcohol_concentration']
(pd.Series(feature_importance,index=feature_name).plot(kind='barh',xlabel='Features Importance',ylabel='Features',title='Feature Importance - XGBoost'))
plt.show()


#Choose the best classifier
best_classifier_name = max(results, key=results.get)
best_classifier = classifiers[best_classifier_name]
print("\nBest classifier:", best_classifier_name)

#Train the best classifier using all the training data
best_classifier.fit(X_train, y_train)

#Create a function to predict wine quality based on user input using the best classifier
def predict_wine_quality_(input_data):
    # Preprocess input data and select features
    input_scaled = scaler.transform(input_data)      
    # Predict the wine quality
    quality_prediction = best_classifier.predict(input_scaled)
    return quality_prediction


#### Wine Quality Prediction  #### 

print("Using" + best_classifier_name + "for prediction")

#Assuming you have a list of column names
column_names = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'alcohol_concentration']

#Prompt for input values
input_values = []
for column_name in column_names:
    value = float(input(f"Enter value for '{column_name}': "))
    input_values.append(value)

#Print the input feature values 
print("Input Feature Values: ", input_values)

#Convert the input values to a numpy array
input_data = np.array([input_values])

#Make predictions
predicted_quality = predict_wine_quality_(input_data)
predicted_quality_label = encoder.inverse_transform(predicted_quality)

#Print the predicted wine quality
print("Predicted Wine Quality is:", str(predicted_quality_label))


