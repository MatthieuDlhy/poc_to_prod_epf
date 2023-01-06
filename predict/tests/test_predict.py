import unittest
from unittest.mock import MagicMock
import tempfile

from train.train import run
from predict.predict.run import TextPredictionModel
from preprocessing.preprocessing import utils
from train.tests import test_model_train


class TestPredict(unittest.TestCase):

    def test_predict(self):
        # create a dictionary params for train conf
        params = {
            'batch_size': 1,
            'epochs': 3,
            'dense_dim': 32,
            'min_samples_per_label': 1,
            'verbose': 1
        }
        utils.LocalTextCategorizationDataset.load_dataset = MagicMock(return_value=test_model_train.load_dataset_mock())

        # we create a temporary file to store artefacts
        with tempfile.TemporaryDirectory() as model_dir:
            # run a training
            accuracy, artefacts_path = run.train('fake_dataset_path', params, model_dir, True)

        model = TextPredictionModel.from_artefacts(artefacts_path)
        prediction = model.predict(["ruby on rails: how to change BG color of options in select list, ruby-on-rails"], 1)

        self.assertEqual([['ruby-on-rails']], prediction)
