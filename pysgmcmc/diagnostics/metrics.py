from functools import wraps
from keras import metrics as keras_metrics

custom_metrics = (

)


def ignore_noise_predictions(metric_function):
    # ignore noise predictions from bayesian neural networks and compute
    # the metric only on the mean prediction
    @wraps(metric_function)
    def decorated_metric(y_true, y_pred):
        prediction = y_pred[:, 0]
        return metric_function(y_true, prediction)
    return decorated_metric


def metric_function(metric):
    if isinstance(metric, str):
        function = keras_metrics.deserialize(
            metric, custom_objects=custom_metrics
        )
    else:
        function = metric
    return ignore_noise_predictions(function)

# Idea: Add Sampler metrics: ESS, gelman_rubin, one_lag_autocorrelation, ...
