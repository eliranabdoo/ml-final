To run the program, install the requirements specified under reqs.txt and run "python main.py"

To tune the hyperparameters of the baseline model (LightGBM) see the constant.py file (these are taken from exercise 2) - "comp_model_params" in constant.py .

To tune the hyperparameter of the compared model (RBoost or EPLBoost) see the main.py file (variable named "model_params")




            'estimator__model__kappa': The capping constant for the samples weights
            
            'estimator__model__T': The number of weak learners
            
            'estimator__model__reg': The regularization parameter that scales the riemennian distance between the samples distribution and
                the initial samples distribution (uniform)
                
            'estimator__model__silent': Silents any output
            
            'estimator__model__verbose': Verbose mode flag

Switch between RBoost and EPLBoost using the USE_RBOOST flag in constant.py

