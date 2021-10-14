import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import GridSearchCV

data = np.genfromtxt("colorrectal_2_classes_formatted.txt", delimiter=",")

classes = data[:, 142]
attributes = np.delete(data,(142), axis=1)

min_max_scaler = MinMaxScaler()
attributes_norm = min_max_scaler.fit_transform(attributes)

x_train, x_test, y_train, y_test = train_test_split(attributes_norm, classes, test_size=0.2)

parameters = {'activation':('identity', 'logistic', 'tanh', 'relu'), 'hidden_layer_sizes':[3,5, 4], 'learning_rate_init': [0.001, 0.002, 0.003], 'max_iter':[210, 250, 200]}
mlp = MLPClassifier()

clf = GridSearchCV(mlp, parameters, cv=5, scoring="accuracy")

clf.fit(x_test, y_test)

y_pred = clf.predict(x_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("acc: ", round(acc, 2))
print("prec: ", round(prec, 2))
print("f1: ", round(f1, 2))
print("recall: ", round(recall, 2))