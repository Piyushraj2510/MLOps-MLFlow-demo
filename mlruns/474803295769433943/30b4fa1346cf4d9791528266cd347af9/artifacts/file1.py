import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt 
import seaborn as sns

#Load the dataset
wine = load_wine()
x = wine.data
y = wine.target

#Train test split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=42)

#Training Parameters

n_estimators=100
max_depth= 5

#files/http format error solved using mlflow url (mlflow.set_tracking_uri)
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Wine-RF-Experiment")


with mlflow.start_run():
    rf = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,random_state=42)
    rf.fit(x_train,y_train)

    y_pred = rf.predict(x_test)
    accuracy = accuracy_score(y_test,y_pred)

    mlflow.log_metric('accuracy',accuracy)
    mlflow.log_param('N_estimators',n_estimators)
    mlflow.log_param('Max_depth',max_depth)

    #Confusion matrix
    cm = confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')

    # save plot
    plt.savefig("Confusion-matrix.png")
    
    # log artifacts using mlflow
    mlflow.log_artifact("Confusion-matrix.png")
    mlflow.log_artifact(__file__)

    # tags
    mlflow.set_tags({"Author": 'Piyush', "Project": 'Wine Classifiaction'})

    # Log the model
    mlflow.sklearn.log_model(rf, "Random-Forest-Model")

    print('accuracy=',accuracy)
    
    





