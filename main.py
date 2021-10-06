import sys
import ast

from job.points_of_interest_job import PointOfInterest
from job.points_of_interest_validation_job import PointsOfInterestValidation
from job.next_poi_category_prediction_sequences_generation_job import NextPoiCategoryPredictionSequencesGenerationJob
from job.next_poi_category_prediction_job import NextPoiCategoryPredictionJob
from job.poi_transactions_analysis_job import PoiTransactionsAnalysisJob
from job.performance_plots_job import PerformancePlots
from foundation.configuration.input import Input

def start_input(args):
    Input().set_inputs(args)


def start_job(args):

    start_input(args)
    job_name = Input.get_instance().inputs['job']
    print(job_name)
    if job_name == "points_of_interest_job":
        job = PointOfInterest()
    elif job_name == "points_of_interest_validation_job":
        job = PointsOfInterestValidation()
    elif job_name == "next_poi_category_prediction_sequences_generation_job":
        job = NextPoiCategoryPredictionSequencesGenerationJob()
    elif job_name == "next_poi_category_prediction_job":
        job = NextPoiCategoryPredictionJob()
    elif job_name == "poi_transactions_job":
        job = PoiTransactionsAnalysisJob()
    elif job_name == "performance_plots_job":
        job = PerformancePlots()

    job.start()

if __name__ == "__main__":
    try:

        args = ast.literal_eval(sys.argv[1])
        start_job(args)

    except Exception as e:
        raise e