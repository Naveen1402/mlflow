import os
import main
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import LabelEncoder
import logging
from mlflow.utils import PYTHON_VERSION
import mlflow.utils.requirements_utils
import importlib_metadata
from mlflow.utils.requirements_utils import _parse_requirements, _infer_requirements
from packaging.requirements import Requirement, InvalidRequirement

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

data = pd.read_csv("/Users/mac/PycharmProjects/ai4i2020_mlflow/ai4i2020.csv")
data.drop(columns = ['UDI', "Product ID"], inplace = True)



le = LabelEncoder()
data['Type'] = le.fit_transform(data['Type'])

data.columns = ['Type', 'AirTemperature', 'ProcessTemperature',
       'RotationalSpeed', 'Torque', 'ToolWear',
       'MachineFailure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']

from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
X = ["AirTemperature", "ProcessTemperature", "RotationalSpeed", "Torque", "ToolWear"]
data[X] = scale.fit_transform(data[X])
Y = data["AirTemperature"]
X = data.drop(['AirTemperature'], 1)
from sklearn.model_selection import train_test_split
X_train, X_test,Y_train,Y_test = train_test_split(X,Y,test_size =0.3, random_state= 42)

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, Y_train)

rfe = RFE(lm, 2)             # running RFE
rfe = rfe.fit(X_train, Y_train)
list(zip(X_train.columns,rfe.support_,rfe.ranking_))

col = X_train.columns[rfe.support_]


alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.9
l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8


with mlflow.start_run():
    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    lr.fit(X_train[col], Y_train)

    predicted_AirTemp = lr.predict(X_test[col])

    (rmse, mae, r2) = eval_metrics(Y_test, predicted_AirTemp)

    print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

    mlflow.log_param("alpha", alpha)
    mlflow.log_param("l1_ratio", l1_ratio)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)

    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

    # Model registry does not work with file store
    if tracking_url_type_store != "file":

        # Register the model
        # There are other ways to use the Model Registry, which depends on the use case,
        # please refer to the doc for more information:
        # https://mlflow.org/docs/latest/model-registry.html#api-workflow
        mlflow.sklearn.log_model(lr, "model", registered_model_name="Elasticnetai4i2020Model")
    else:
        mlflow.sklearn.log_model(lr, "model")