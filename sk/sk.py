 from sklearn import datasets
 import numpy as np
 iris = datasets.load_iris()
# print(iris)
 X = iris.data[:, [2, 3]]
 y = iris.target
print(type(X))
print(X)


 from sklearn.cross_validation import train_test_split
 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


 from sklearn.preprocessing import StandardScaler
 sc = StandardScaler()
 sc.fit(X_train)
 X_train_std = sc.transform(X_train)
 X_test_std = sc.transform(X_test)


#Perceptron

 from sklearn.linear_model import Perceptron
 ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)
 ppn.fit(X_train_std, y_train)


y_pred = ppn.predict(X_test_std)
print('Misclassified samples: %d' % (y_test != y_pred).sum())



    from sklearn.metrics import accuracy_score
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))


      from matplotlib.colors import ListedColormap
   import matplotlib.pyplot as plt
   def plot_decision_regions(X, y, classifier,
                       test_idx=None, resolution=0.02):
       # setup marker generator and color map

           
       markers = ('s', 'x', 'o', '^', 'v')
       colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
       cmap = ListedColormap(colors[:len(np.unique(y))])
       # plot the decision surface
       x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
       x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
       xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                            np.arange(x2_min, x2_max, resolution))
       Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
       Z = Z.reshape(xx1.shape)
       plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
       plt.xlim(xx1.min(), xx1.max())
       plt.ylim(xx2.min(), xx2.max())
       # plot all samples
       for idx, cl in enumerate(np.unique(y)):
           plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                       alpha=0.8, c=cmap(idx),
                       marker=markers[idx], label=cl)
       # highlight test samples
       if test_idx:
           X_test, y_test = X[test_idx, :], y[test_idx]
           plt.scatter(X_test[:, 0], X_test[:, 1], c='',
                   alpha=1.0, linewidths=1, marker='o',
                   s=55, label='test set')



    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))
    plot_decision_regions(X=X_combined_std,
   ...                       y=y_combined,
   ...                       classifier=ppn,
   ...                       test_idx=range(105,150))
    plt.xlabel('petal length [standardized]')
    plt.ylabel('petal width [standardized]')
    plt.legend(loc='upper left')
    plt.show()


       import matplotlib.pyplot as plt
    import numpy as np
    def sigmoid(z):
   ...     return 1.0 / (1.0 + np.exp(-z))
    z = np.arange(-7, 7, 0.1)
    phi_z = sigmoid(z)
    plt.plot(z, phi_z)
    plt.axvline(0.0, color='k')
    plt.axhspan(0.0, 1.0, facecolor='1.0', alpha=1.0, ls='dotted')
    plt.axhline(y=0.5, ls='dotted', color='k')
    plt.yticks([0.0, 0.5, 1.0])
    plt.ylim(-0.1, 1.1)
    plt.xlabel('z')
    plt.ylabel('$\phi (z)$')
    plt.show()


#LogisticRegression

       from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1000.0, random_state=0)
    lr.fit(X_train_std, y_train)
    plot_decision_regions(X_combined_std,
   ...                       y_combined, classifier=lr,
   ...                       test_idx=range(105,150))
    plt.xlabel('petal length [standardized]')
    plt.ylabel('petal width [standardized]')
    plt.legend(loc='upper left')
    plt.show()


   lr.predict_proba(X_test_std[0,:])


       weights, params = [], []
    for c in np.arange(-5, 5):
   ...     lr = LogisticRegression(C=10**c, random_state=0)
   ...     lr.fit(X_train_std, y_train)
   ...     weights.append(lr.coef_[1])
   ...     params.append(10**c)
    weights = np.array(weights)
    plt.plot(params, weights[:, 0],
   ...          label='petal length')
    plt.plot(params, weights[:, 1], linestyle='--',
   ...          label='petal width')
    plt.ylabel('weight coefficient')
    plt.xlabel('C')
    plt.legend(loc='upper left')
    plt.xscale('log')
    plt.show()



# support vector machine SVM and SVC

from sklearn.svm import SVC
svm = SVC(kernel='linear', C=1.0, random_state=0)
svm.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std,y_combined, classifier=svm,test_idx=range(105,150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()



# SGDClassifier
# when data is too large to load to memory use SGDClassifier
from sklearn.linear_model import SGDClassifier
ppn = SGDClassifier(loss='perceptron')
lr = SGDClassifier(loss='log')
svm = SGDClassifier(loss='hinge')



# random generate dataset
np.random.seed(0)
X_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
y_xor = np.where(y_xor, 1, -1)
plt.scatter(X_xor[y_xor==1, 0], X_xor[y_xor==1, 1],c='b', marker='x', label='1')
plt.scatter(X_xor[y_xor==-1, 0], X_xor[y_xor==-1, 1],c='r', marker='s', label='-1')
plt.ylim(-3.0)
plt.legend()
plt.show()


svm = SVC(kernel='rbf', random_state=0, gamma=0.10, C=10.0)
svm.fit(X_xor, y_xor)
plot_decision_regions(X_xor, y_xor, classifier=svm)
plt.legend(loc='upper left')
plt.show()



svm = SVC(kernel='rbf', random_state=0, gamma=0.2, C=1.0)
svm.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, y_combined, classifier=svm,test_idx=range(105,150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()
