import pickle
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

class MultiModelClassifier:

    def __init__(self,confidence_standard):
        """
        Initialize model objects
        """
        self.confidence_standard = confidence_standard

        file = open("models/CountVectorizer.pickle","rb")
        self.cv = pickle.load(file)
        file.close()

        file = open("models/multinomialNB.pickle","rb")
        self.multinomialNB = pickle.load(file)
        file.close()

        file = open("models/logisticRegression.pickle","rb")
        self.logisticRegression = pickle.load(file)
        file.close()

        file = open("models/sgdClassifier.pickle","rb")
        self.sgdClassifier = pickle.load(file)
        file.close()

        file = open("models/linearSVC.pickle","rb")
        self.linearSVC = pickle.load(file)
        file.close()

        # file = open("models/randomForestClassifier.pickle","rb")
        # self.randomForestClassifier = pickle.load(file)
        # file.close()

    def predict(self, X):
        X = self.cv.transform(X)
        # predictions = self.multinomialNB.predict(X) + self.logisticRegression.predict(X) + self.sgdClassifier.predict(X) + self.linearSVC.predict(X) + self.randomForestClassifier.predict(X)
        predictions = self.multinomialNB.predict(X) + self.logisticRegression.predict(X) + self.sgdClassifier.predict(X) + self.linearSVC.predict(X)
        confidence = []

        for i, prediction in enumerate(predictions):
            if prediction >= self.confidence_standard :
                predictions[i] = 1
                confidence.append(prediction/4)
            else:
                predictions[i] = 0
                confidence.append((4-prediction)/4)

        return predictions, confidence

    # def informative_features(self):
    #     return sorted(tuple(zip(self.linearSVC.coef_[0],self.cv.get_feature_names())))
    #
    # def relative_words(self):
    #     return sorted(tuple(zip(self.multinomialNB.coef_[0],self.cv.get_feature_names())))

    def linearSVCCoefficients(self):
        return sorted(tuple(zip(self.linearSVC.coef_[0],self.cv.get_feature_names())))

    def multinomialNBCoefficients(self):
        return sorted(tuple(zip(self.multinomialNB.coef_[0],self.cv.get_feature_names())))

    def logisticRegressionCoefficients(self):
        return sorted(tuple(zip(self.logisticRegression.coef_[0],self.cv.get_feature_names())))

    def stochasticGradientCoefficients(self):
        return sorted(tuple(zip(self.sgdClassifier.coef_[0],self.cv.get_feature_names())))

    # def most_informative_features_positive(self):
    #     z = sorted(tuple(zip(self.linearSVC.coef_[0],self.cv.get_feature_names())))
    #     return z[-100:]
    #
    # def most_informative_features_negative(self):
    #     z = sorted(tuple(zip(self.linearSVC.coef_[0],self.cv.get_feature_names())))
    #     return z[:100]
