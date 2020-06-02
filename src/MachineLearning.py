from PreProecesssing import data, cleaning_data
from sklearn.model_selection import train_test_split
from Algorithms import *


data = cleaning_data(data)

sentences = data["text"].values
y = data['category'].values

categories = ['deprem', 'değil', 'yangın', 'sel', 'heyelan']
algorithm_names = ['logistic regresion', 'random forest', 'lsvm', 'naive bayes']
accuracy_list = []

x_train, x_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25, random_state=100)  # SPLIT DATA

y_pred, accuracy_lg = logistic_regression(x_train, y_train, x_test, y_test, categories)
draw_confusion_matrix(y_test, y_pred, categories)
accuracy_list.append(accuracy_lg)

y_pred, accuracy_rf = random_forest(x_train, y_train, x_test, y_test, categories)
draw_confusion_matrix(y_test, y_pred, categories)
accuracy_list.append(accuracy_rf)

y_pred, accuracy_lsvm = linear_support_vector_machine(x_train, y_train, x_test, y_test, categories)
draw_confusion_matrix(y_test, y_pred, categories)
accuracy_list.append(accuracy_lsvm)

y_pred, accuracy_nb = naive_bayes(x_train, y_train, x_test, y_test, categories)
draw_confusion_matrix(y_test, y_pred, categories)
accuracy_list.append(accuracy_nb)





