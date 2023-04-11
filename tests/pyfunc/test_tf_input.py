import json

import pytest

import mlflow.pyfunc.scoring_server as pyfunc_scoring_server
from mlflow.types import Schema, ColSpec
from mlflow.models.utils import _enforce_schema
from mlflow.exceptions import MlflowException


def test_parse_tf_input():
    schema = Schema([ColSpec("string", "nested")])
    json_input = json.dumps({
        "instances": [{
            "nested": [{
                "nest1": 1
            }],
        }]
    })
    data = pyfunc_scoring_server.infer_and_parse_json_input(json_input, schema=schema)
    with pytest.raises(MlflowException):
        _enforce_schema(data, schema)