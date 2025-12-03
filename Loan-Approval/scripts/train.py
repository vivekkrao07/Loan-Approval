from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

def train_decision_tree(X, y):
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )
    clf = DecisionTreeClassifier(random_state=1)
    clf.fit(x_train, y_train)
    return clf, x_train, x_test, y_train, y_test
