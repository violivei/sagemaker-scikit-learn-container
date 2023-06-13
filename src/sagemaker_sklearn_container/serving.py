from __future__ import absolute_import
import os
import logging
import shlex
import subprocess
import sys
import numpy as np
from subprocess import CalledProcessError

from retrying import retry
from sagemaker_inference import model_server
from sagemaker_inference import content_types, decoder, default_inference_handler, encoder
from sagemaker_sklearn_container import handler_service
from sagemaker_sklearn_container.serving_mms import start_model_server


def is_multi_model():
    return os.environ.get('SAGEMAKER_MULTI_MODEL')

def main():
    start_model_server(is_multi_model())
    subprocess.call(["tail", "-f", "/dev/null"])

############# Unit Tests Only #############
def default_model_fn(model_dir):
    """Loads a model. For Scikit-learn, a default function to load a model is not provided.
    Users should provide customized model_fn() in script.
    Args:
        model_dir: a directory where model is saved.
    Returns: A Scikit-learn model.
    """
    return handler_service.HandlerService().DefaultSKLearnUserModuleInferenceHandler().default_model_fn(model_dir)

def default_input_fn(input_data, content_type):
    """Takes request data and de-serializes the data into an object for prediction.
        When an InvokeEndpoint operation is made against an Endpoint running SageMaker model server,
        the model server receives two pieces of information:
            - The request Content-Type, for example "application/json"
            - The request data, which is at most 5 MB (5 * 1024 * 1024 bytes) in size.
        The input_fn is responsible to take the request data and pre-process it before prediction.
    Args:
        input_data (obj): the request data.
        content_type (str): the request Content-Type.
    Returns:
        (obj): data ready for prediction.
    """
    np_array = decoder.decode(input_data, content_type)
    return np_array.astype(np.float32) if content_type in content_types.UTF8_TYPES else np_array

def default_predict_fn(input_data, model):
    """A default predict_fn for Scikit-learn. Calls a model on data deserialized in input_fn.
    Args:
        input_data: input data (Numpy array) for prediction deserialized by input_fn
        model: Scikit-learn model loaded in memory by model_fn
    Returns: a prediction
    """
    output = model.predict(input_data)
    return output
    
def default_output_fn(prediction, accept):
    """Function responsible to serialize the prediction for the response.
    Args:
        prediction (obj): prediction returned by predict_fn .
        accept (str): accept content-type expected by the client.
    Returns:
        (worker.Response): a Flask response object with the following args:
            * Args:
                response: the serialized data to return
                accept: the content-type that the data was transformed to.
    """
    return encoder.encode(prediction, accept), accept
