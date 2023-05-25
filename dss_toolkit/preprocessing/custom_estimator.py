 from sklearn.base import BaseEstimator

 class CustomEstimator(BaseEstimator):
    def __init__(
        self, 
        estimator = LogisticRegression(),
    ):
        """
        A Custom BaseEstimator that can switch between classifiers.
        :param estimator: sklearn object - The classifier
        """ 

        self.estimator = estimator
        self.cv_results = None


    def fit(self, X, y=None, **kwargs):
        self.estimator.fit(X, y)
        return self


    def predict(self, X, y=None):
        return self.estimator.predict(X)


    def predict_proba(self, X):
        return self.estimator.predict_proba(X)


    def score(self, X, y):
        return self.estimator.score(X, y)
    
    def set_cv_results(self, cv_results_):
        self.cv_results = cv_results_
        
    def get_cv_results(self):
        return self.cv_results