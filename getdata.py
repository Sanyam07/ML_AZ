import os
import tarfile
from six.moves import urllib
import pandas as pd
from sklearn.datasets import fetch_openml

class GetHousingData:
    DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
    HOUSING_PATH = os.path.join("datasets", "housing")
    HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"

    def __init__(self, download_root=DOWNLOAD_ROOT):
        self.DOWNLOAD_ROOT = download_root


    def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
        os.makedirs(housing_path, exist_ok=True)
        tgz_path = os.path.join(housing_path, "housing.tgz")
        urllib.request.urlretrieve(housing_url, tgz_path)
        housing_tgz = tarfile.open(tgz_path)
        housing_tgz.extractall(path=housing_path)
        housing_tgz.close()
        print('finished')
        return tgz_path

    def load_housing_data(self, housing_path = HOUSING_PATH):
        csv_path = os.path.join(housing_path, "housing.csv")
        return pd.read_csv(csv_path)

# fetch_housing_data()
# housing = load_housing_data()

class MNISTData:

    @staticmethod
    def get_mnist_data():
        mnist = fetch_openml('mnist_784', version=1)
        return mnist


