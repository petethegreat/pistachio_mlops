
from kfp import dsl
from kfp import compiler




@dsl.component(base_image='python:3.11')
def slicedmetrics_component(
    evaluation_metrics: dsl.Output[dsl.SlicedClassificationMetrics]):

    evaluation_metrics._sliced_metrics = {}

    # dummy roc data
    fpr = [ 0.0, 0.0, 0.0, 1.0]
    tpr = [0.0, 0.5, 1.0, 1.0]
    thresholds = [float('inf'), 0.99, 0.8, 0.01]

    evaluation_metrics.load_roc_readings(
        'test_slice', [thresholds, tpr, fpr]
    )




@dsl.pipeline(
    name='slicedclassification_sample_pipeline')
def example_pipeline():
    metrics_op = slicedmetrics_component()

compiler.Compiler().compile(example_pipeline, package_path='slicedclassification_pipeline.json')


