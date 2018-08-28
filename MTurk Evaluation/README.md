## Instructions for Evaluation
All the test responses by different models can be found in `./MTurk2 model responses/`. All the Mturk annotations for Content Richness and Plausibity can be found in `./Mturk2 Movie C Results/` and `./Mturk2 Movie P Results/` respectively.

- Run `python results_evaluator_C_movie_mturk2.py` to get the evaluation of Content Richness of all the models
- Run `python results_evaluator_P_movie_mturk2.py` to get the evaluation of Plausibility of all the models
- Run `python calculate_C_bootstrap_significance.py` to do the bootstrap significance testing of Content Richness among the models
- Run `python calculate_P_bootstrap_significance.py` to do the bootstrap significance testing of Plausibility among the models

