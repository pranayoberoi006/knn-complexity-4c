from sklearn.datasets import load_iris 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_predict
import matplotlib.pyplot as plt

data =load_iris()
print(data)
X=data.data
y=data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size= 0.3, random_state=42
)

X_train.shape, X_test.shape, y_train.shape, y_test.shape

from sklearn.model_selection import cross_val_score
k_range = range(1, 31)
cv_scores = []
for k in k_range:
    knn=KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5)
    print(f"k={k}, cross-validation Scores: {scores}, Mean Score: {scores.mean()}")
    cv_scores.append(scores.mean())


plt.figure(figsize=(8,5))
plt.plot(k_range, cv_scores, marker='*', color='green')
plt.xlabel('value of k for knn')
plt.ylabel('cross-validated accuracy')
plt.title('finding optimal k (bias-variance tradeoff)')
plt.grid(True)
plt.show()