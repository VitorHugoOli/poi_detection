import pandas as pd

from domain.user_step_domain import UserStepDomain
from foundation.abs_classes.job import Job
from domain.points_of_interest_domain import PointsOfInterestDomain
from loader.file_loader import FileLoader
from foundation.configuration.input import Input

class PointOfInterest(Job):

    def __init__(self):
        self.user_steps_domain = UserStepDomain()
        self.points_of_interest_domain = PointsOfInterestDomain()
        self.file_loader = FileLoader()

    def start(self):
        users_steps = self.user_steps_domain.users_steps_from_csv()
        filename = Input.get_arg("poi_detection_output")

        """
        Identifying and classifying PoIs of each user
        """
        users_pois_classified = users_steps.groupby(by='id').\
            apply(lambda e: self.points_of_interest_domain.
                  individual_point_interest(e['id'].tolist(), e['latitude'].tolist(), e['longitude'].tolist(), e['datetime'].tolist()))

        """
        Organazing the results into a single table
        """
        users_pois_classified_concatenated = pd.DataFrame({"id": [], "poi_type": [], "latitude": [], "longitude": [],
                           "work_time_events": [], "home_time_events": []})

        for i in range(users_pois_classified.shape[0]):
            users_pois_classified_concatenated = users_pois_classified_concatenated.\
                append(users_pois_classified.iloc[i], ignore_index=True)

        users_pois_classified_concatenated['id'] = users_pois_classified_concatenated['id'].astype('int64')
        self.file_loader.save_df_to_csv(users_pois_classified_concatenated, filename)
