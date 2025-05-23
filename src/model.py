from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

def train_model(model_type, X_train, y_train):
    if model_type == 'logistic':
        model = LogisticRegression(max_iter=1000)
    elif model_type == 'naive_bayes':
        model = MultinomialNB()
    elif model_type == 'svm':
        model = SVC()
    else:
        raise ValueError("Unsupported model type: {}".format(model_type))
    model.fit(X_train, y_train)
    return model
