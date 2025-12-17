from types import SimpleNamespace


TRAIN_LR = 0.002

BB_GM =  dict(
        lambda_val =  80.0,
        solver_name = "lpmp",
        solver_params = {
            "maxIter": 100,
            "primalComputationInterval": 10,
            "timeout": 1000
        }
)
BB_GM = SimpleNamespace(**BB_GM)