# from https://stackoverflow.com/questions/76441777/huggingface-evaluate-function-use-multiple-labels/78617839#78617839
import datasets
import evaluate
from evaluate import Metric
from sklearn.metrics import accuracy_score


class MulticlassAccuracy(Metric):
    """Workaround for the default Accuracy class which doesn't support passing 'average' to the compute method."""

    def _info(self):
        return evaluate.MetricInfo(
            description="Accuracy",
            citation="",
            inputs_description="",
            features=datasets.Features(
                {
                    "predictions": datasets.Sequence(datasets.Value("int32")),
                    "references": datasets.Sequence(datasets.Value("int32")),
                }
                if self.config_name == "multilabel"
                else {
                    "predictions": datasets.Value("int32"),
                    "references": datasets.Value("int32"),
                }
            ),
            reference_urls=["https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html"],
        )

    def _compute(self, predictions, references, normalize=True, sample_weight=None, **kwargs):
        # take **kwargs to avoid breaking when the metric is used with a compute method that takes additional arguments
        return {
            "accuracy": float(
                accuracy_score(references, predictions, normalize=normalize, sample_weight=sample_weight)
            )
        }

        
def get_full_metrics(metric: str | None, num_labels: int | None):
    if num_labels == 2:
        kwargs = None
    else:
        kwargs = {"average": "macro"}
        # kwargs = {"average": "weighted"}

    metrics_dict = {metric: metric}
    metrics_dict["accuracy"] = MulticlassAccuracy()
    metrics_dict["f1"] = "f1"
    metrics_dict["precision"] = "precision"
    metrics_dict["recall"] = "recall"

    return evaluate.combine(metrics_dict).compute, kwargs