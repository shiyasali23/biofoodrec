"""Food recommendations from blood test results. Every bias, weight and setting can be tuned per person.

    from biofoodrec import FoodRecommender, sample_patient
    result = FoodRecommender().recommend(sample_patient())
    print(result.summary())
"""
from .labs import Labs, read_labs, sample_patient
from .recommender import FoodRecommender, Recommendation
from .reference import ReferenceData
from .tuning import Tuning

__all__ = ["FoodRecommender", "Labs", "Recommendation", "ReferenceData", "Tuning", "read_labs", "sample_patient"]
__version__ = "0.2.0"
