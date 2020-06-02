import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from matplotlib import pyplot as plt



def linear_support_vector_machine(x_train, y_train, x_test, y_test, categories):

    sgd = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),  # Strip_accents = unicode olması nedir?
                    ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3,
                                          random_state=42, max_iter=5, tol=None))
                    ])
    sgd.fit(x_train, y_train)
    y_pred = sgd.predict(x_test)
    accuracy = accuracy_score(y_pred, y_test)
    print(classification_report(y_test, y_pred, categories))

    return y_pred, accuracy


def random_forest(x_train, y_train, x_test, y_test, categories):

    rf = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('clf', RandomForestClassifier()),
                   ])
    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_test)
    accuracy = accuracy_score(y_pred, y_test)
    print(classification_report(y_test, y_pred, categories))

    return y_pred, accuracy


def logistic_regression(x_train, y_train, x_test, y_test, categories):

    logreg = Pipeline([('vect', CountVectorizer()),
                       ('tfidf', TfidfTransformer()),
                       ('clf', LogisticRegression(n_jobs=1, C=1e5)),
                       ])

    logreg.fit(x_train, y_train)
    y_pred = logreg.predict(x_test)
    accuracy = accuracy_score(y_pred, y_test)
    print(classification_report(y_test, y_pred, target_names=categories))

    return y_pred, accuracy


def naive_bayes(x_train, y_train, x_test, y_test, categories):
    nb = Pipeline([('vect', CountVectorizer()),
                   ('tfidf', TfidfTransformer()),
                   ('clf', MultinomialNB()),
                   ])
    nb.fit(x_train, y_train)
    y_pred = nb.predict(x_test)
    accuracy = accuracy_score(y_pred, y_test)
    print(classification_report(y_test, y_pred, target_names=categories))

    return y_pred, accuracy


def draw_confusion_matrix(y_test, y_pred, categories):

    plt.title('Random Forest Confusion Matrix')
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', xticklabels=categories, yticklabels=categories)
    plt.ylabel('Actual')
    plt.xlabel('Pred')
    plt.show()
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


