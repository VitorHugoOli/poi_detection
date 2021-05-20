from enum import Enum
import pytz
from tensorflow.keras.optimizers import Adam, Adadelta, SGD, RMSprop, Nadam


class NextPoiCategoryPredictionConfiguration:

    # Radius for the nearestneighbors algorithm - 100m
    def __init__(self):
        self.SEQUENCES_SIZE = ("sequences_size", 4)

        self.N_SPLITS = ("n_splits", 5)

        self.EPOCHS = ("epochs", 10)

        self.N_REPLICATIONS = ("n_replications", 1)

        self.BATCH = ("batch", {'mfa': 200, 'serm': 200, 'map': 200, 'stf': 200, 'next': 200})

        self.OPTIMIZER = ("learning_rate", {'mfa': Adam(), 'serm': Adam(), 'map': Adam(), 'stf': Adam(),
                                            'next': Adam()})

        self.FORMAT_MODEL_NAME = ("format_model_name", {'mfa': 'MFA-RNN', 'serm': 'SERM', 'map': 'MAP', 'stf': 'STF-RNN',
                                            'next': 'MHSA+PE'})

        self.OUTPUT_BASE_DIR = (
        "output_dir", "output/next_poi_category_prediction/", False, "output directory for the poi_categorization")

        self.MODEL_NAME = ("model_name", {'mfa': "mfa/", 'serm': "serm/", 'map': "map/", 'stf': "stf/",
                                          'next': "next/"})

        self.DATASET_TYPE = ("dataset_type", {'users_steps': "users_steps/"})

        self.CATEGORY_TYPE = ("category_type",
                         {'10_categories': "10_categories/",
                          '8_categories': "8_categories/",
                          '3_categories': "3_categories/"})

        self.CLASS_WEIGHT = ("class_weight",
                        {'10_categories': {'serm': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1},
                                           'map': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1},
                                           'stf': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1},
                                           'mfa': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1},
                                           'next': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1}},
                         '8_categories': {'serm': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1},
                                           'map': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1},
                                           'stf': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1},
                                           'mfa': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1},
                                          'next': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1}},
                         '3_categories': {'serm': {0: 1, 1: 1, 2: 1},
                                           'map': {0: 1, 1: 1, 2: 1},
                                           'stf': {0: 1, 1: 1, 2: 1},
                                           'mfa': {0: 1, 1: 1, 2: 1},
                                          'next': {0: 1, 1: 1, 2: 1}}})

        self.DATASET_COLUMNS = ("dataset_columns", {"users_steps": {"datetime": "datetime",
                                                                  "userid": "id",
                                                                  "locationid": "placeid",
                                                                  "category": "poi_resulting",
                                                                  "latitude": "latitude",
                                                                  "longitude": "longitude",
                                                                    "country": "country_name"}})

        self.CATEGORIES_10 = ['Home', 'Work', 'Other', 'Commuting', 'Amenity', 'Leisure', 'Office', 'Shop', 'Sport', 'Tourism']

        self.CATEGORIES_8 = ['Home', 'Work', 'Other', 'Commuting', 'Amenity', 'Leisure', 'Shop', 'Tourism']

        self.CATEGORIES_3 = ['displacement', 'home', 'other']

        self.CATEGORIES_TO_INT = ("categories_to_int", {"users_steps":
                                                            {"10_categories": {self.CATEGORIES_10[i]: i for i in range(len(self.CATEGORIES_10))},
                                                             "8_categories": {self.CATEGORIES_8[i]: i for i in range(len(self.CATEGORIES_8))},
                                                             "3_categories": {self.CATEGORIES_3[i]: i for i in range(len(self.CATEGORIES_3))}}})

        self.INT_TO_CATEGORIES = ("int_to_categories", {"users_steps": {"10_categories": {str(i): self.CATEGORIES_10[i] for i in range(len(self.CATEGORIES_10))},
                                                                        "8_categories": {str(i): self.CATEGORIES_8[i] for i in range(len(self.CATEGORIES_8))},
                                                                        "3_categories": {str(i): self.CATEGORIES_3[i] for i in range(len(self.CATEGORIES_3))}}})

        self.MAX_POIS = ("max_pois", 10)

        self.REPORT_10_INT_CATEGORIES = ("report_10_int_categories",
                                        {'0': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
                                         '1': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
                                         '2': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
                                         '3': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
                                         '4': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
                                         '5': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
                                         '6': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
                                         '7': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
                                         '8': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
                                         '9': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
                                         'accuracy': [],
                                         'macro avg': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
                                         'weighted avg': {'precision': [], 'recall': [], 'f1-score': [],
                                                          'support': []}},
                                        "report")

        self.REPORT_8_INT_CATEGORIES = ("report_8_int_categories",
                                         {'0': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
                                          '1': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
                                          '2': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
                                          '3': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
                                          '4': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
                                          '5': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
                                          '6': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
                                          '7': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
                                          'accuracy': [],
                                          'macro avg': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
                                          'weighted avg': {'precision': [], 'recall': [], 'f1-score': [],
                                                           'support': []}},
                                         "report")

        self.REPORT_3_INT_CATEGORIES = ("report_10_int_categories",
                                         {'0': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
                                          '1': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
                                          '2': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
                                          'accuracy': [],
                                          'macro avg': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
                                          'weighted avg': {'precision': [], 'recall': [], 'f1-score': [],
                                                           'support': []}},
                                         "report")

        self.REPORT_MODEL = ("report_model",
                             {'10_categories': self.REPORT_10_INT_CATEGORIES[1],
                              '8_categories': self.REPORT_8_INT_CATEGORIES[1],
                              '3_categories': self.REPORT_3_INT_CATEGORIES[1]})

        self.NUMBER_OF_CATEGORIES = ("number_of_categories", {'10_categories': 10,
                                                              '8_categories': 8,
                                                              '3_categories': 3})

        self.STEP_SIZE = ("step_size", 4)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def get_key(self):
        return self.value[0]

    def get_value(self):
        return self.value[1]