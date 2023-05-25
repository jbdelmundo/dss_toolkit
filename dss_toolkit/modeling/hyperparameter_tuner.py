from dss_toolkit.modeling.model_builder import ModelBuilder


class HyperparmeterTuner:
    def __init__(self, task, algorithm, hyperparameter_search_space, metric, maximize, how="gridsearch"):
        self.task = task
        self.algorithm = algorithm
        self.hyperparameter_search_space = hyperparameter_search_space
        self.metric = metric
        self.maximize = maximize
        self.how = how

        self.params = []
        self.metrics = []
        self.best_model_ix = -1

    def search(self, X, y):

        if self.how == "gridsearch":

            # Select Hyperparameter (loop)
            for hyperparameter in self.hyperparameter_search_space:

                model = ModelBuilder(self.task, self.algorithm, hyperparameter)
                eval_metrics = model.evaluate(X, y)  # Add Cross Validation to evaluate on separate dataset
                self.params.append(hyperparameter)
                self.metrics.append(eval_metrics[self.metric])
