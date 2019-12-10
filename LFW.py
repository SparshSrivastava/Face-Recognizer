# Importing the dataset as it is inbuilt in sklearn so downloading it from there
from sklearn import datasets
lfw = datasets.fetch_lfw_people(min_faces_per_person = 100, resize = 0.4)

# Looking at how our dataset looks like
lfw.keys()
print(lfw.DESCR)
print(lfw.images.shape)
print(lfw.data.shape)
print(lfw.target_names)

# Now plotting the first 64 images to have a look at how our images look like in this case
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(8,8)) # Creating a grid of 8*8 for plotting the images

for i in range(64):
    ax = fig.add_subplot(8, 8, i+1) # Addind a subplot to each block in the entire space 
    ax.imshow(lfw.images[i], cmap = plt.cm.bone) # Plotting the subplot
    
plt.show()

# Now importing the data and target into variables
X = lfw.data
Y = lfw.target

from sklearn import model_selection
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, random_state = 0)

# Now appying PCA on the dataset
from sklearn.decomposition import PCA
pca = PCA()

# Fitting this pca object onto the data
pca.fit(X_train)

# Now finding the optimal value of the n_components for this dataset
total_variance = pca.explained_variance_.sum() # to hold the sum of all variances

current_variance = 0 # To hold the current variance value
k = 0 # Taken as a ccounter

while current_variance/total_variance < 0.99:
    current_variance += pca.explained_variance_[k]
    k += 1

print(k)

# Now applying the pca to the optimal value of n_components found
pca = PCA(n_components=k, whiten = True)

X_train_pca = pca.fit_transform(X_train)

# Now we need to check whether the images now after reduction also looks intact
X_approx = pca.inverse_transform(X_train_pca)
X_approx.shape

# We need to reshape it into 3D to plto it and compare with the previous plotting
X_approx = X_approx.reshape((855, 50, 37))

# Now plotting this X_approx for checking purposes
fig = plt.figure(figsize=(8,8))

for i in range(64):
    ax = fig.add_subplot(8,8,i+1)
    ax.imshow(X_approx[i], cmap = plt.cm.bone)

plt.show()

# Now we are going to look at eigenfaces to look at interesting features coming out of them
eigen_vectors = pca.components_
eigen_vectors.shape

eigen_faces = eigen_vectors.reshape((316, 50, 37))

# Now plotting the eigen faces

fig = plt.figure(figsize=(8,8))

for i in range(64):
    ax = fig.add_subplot(8, 8, i+1)
    ax.imshow(eigen_faces[i], cmap = plt.cm.bone)

plt.show()

# Now getting our test data ready for PCA applied
X_test_pca = pca.transform(X_test)

# As the problem is of classification therefore we need a classifier in this case lets begin with random forest
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier()

# We need to check how fast is PCA performing
import time

# First performing on data without PCA
start = time.time()
forest.fit(X_train,Y_train)
end = time.time()

print(end-start) # Total time taken for fitting

# Now predicting the responses using our classifier
Y_pred = forest.predict(X_test)

# Now printing the classification Report
from sklearn.metrics import classification_report
print(classification_report(Y_test,Y_pred))

#Now printing the confusion matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(Y_test, Y_pred))

# Now applying the classifier with PCA
forest = RandomForestClassifier()

start = time.time()
forest.fit(X_train_pca,Y_train)
end = time.time()

print(end-start)

# Making the predictions using the Random Forest Classifer on PCA dataset
Y_pred_pca = forest.predict(X_test_pca)

# Checking the model accuracy via classification report and confusion matrix
print(classification_report(Y_test,Y_pred_pca))
print(confusion_matrix(Y_test, Y_pred_pca))


# Now using the SVM classifier
from sklearn.svm import SVC
svc = SVC()

# Using Grid Search to find the optimal values of C and gamma
from sklearn.model_selection import GridSearchCV

grid = {"C": [1e2, 1e3, 5e3, 1e4, 5e4, 1e5],
        "gamma": [1e-3, 5e-4, 1e-4, 5e-3]}

svm_grid_search = GridSearchCV(svc, grid)

# Fitting the classifer with training data after grid search
svm_grid_search.fit(X_train, Y_train)

# SVM Predictions for wihtout PCA dataset
Y_pred_svm = svm_grid_search.predict(X_test)

# Checking the accuracy of the model using classification report and confusion matrix
print(classification_report(Y_test, Y_pred_svm))
print(confusion_matrix(Y_test, Y_pred_svm))

#Now applying SVM classifier on PCA Dataset

# Fitting the SVM classifier on the PCA applied data
svm_pca = svm_grid_search.fit(X_train_pca, Y_train)

# Making the predictions
Y_pred_svm_pca = svm_pca.predict(X_test_pca)

# Checking the model's accuracy via classification report and confusion matrix
print(classification_report(Y_test, Y_pred_svm_pca))
print(confusion_matrix(Y_test, Y_pred_svm_pca))

# Printing the optimal values of C and gamma as calculated by grid search
print(svm_grid_search.best_estimator_)

# Now using the KNN classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()

# Appyling Grid Search for finding the optimal value of k

grid_knn = {'n_neighbors':[i for i in range(1,25,2)]}

knn_grid_search = GridSearchCV(knn, grid_knn)

# fitting the KNN classifier after grid search
knn_grid_search.fit(X_train, Y_train)

# Making the prediction on the without PCA dataset using the KNN classifier
Y_pred_knn = knn_grid_search.predict(X_test)

# Checking the accuracy of the model via classification report and confusion matrix
print(classification_report(Y_test, Y_pred_knn))
print(confusion_matrix(Y_test, Y_pred_knn))

# Now appyling the KNN classifier on PCA data
knn_grid_search_pca = knn_grid_search.fit(X_train_pca, Y_train)

# Making the predictions
Y_pred_knn_pca = knn_grid_search_pca.predict(X_test_pca)

# Checking the accuracy of the model
print(classification_report(Y_test, Y_pred_knn_pca))
print(confusion_matrix(Y_test, Y_pred_knn_pca))


# Therefore after applying Random Forest Classifier, SvM classfier and KNN classifier on both PCA dataset and without PCA datasets we conclude that SVM on PCA dataset works the best giving us a better accuracy then the other models.