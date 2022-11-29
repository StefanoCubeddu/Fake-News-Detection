import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfTransformer
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

#read the file True.csv and Fake.csv
df_true=pd.read_csv('True.csv')
df_fake=pd.read_csv('Fake.csv')

#merge the two files into one adding a column with the label True or Fake
df_true['label']='True'
df_fake['label']='Fake'
df=pd.concat([df_true,df_fake])

#split the data into train and test randomly keeping title and text and the label
X_train,X_test,y_train,y_test=train_test_split(df['text'],df['label'],test_size=0.2,random_state=6)

#implement the tfidf vectorizer 
count_vectorizer=CountVectorizer(stop_words='english')
count_train=count_vectorizer.fit_transform(X_train)
count_test=count_vectorizer.transform(X_test)
tf_idf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tf_idf_transformer.fit(count_train)
tf_idf_train=tf_idf_transformer.transform(count_train)
tf_idf_test=tf_idf_transformer.transform(count_test)


#initialize a PassiveAggressiveClassifier
pac=PassiveAggressiveClassifier(max_iter=75)
pac.fit(tf_idf_train,y_train)

#predict on the test set and calculate the accuracy
y_pred=pac.predict(tf_idf_test)
score=accuracy_score(y_test,y_pred)

#print the accuracy
print(f'Accuracy: {round(score*100,2)}%')

#plot the confusion matrix
plot_confusion_matrix(confusion_matrix(y_test,y_pred,labels=['Fake','True']),classes=['Fake','True'])



#print some of the results displaying the text, the label and the prediction
for i in range(0,10):
    print('Text:',X_test.iloc[i])
    print('Label:',y_test.iloc[i])
    print('Prediction:',y_pred[i])
    print('')












