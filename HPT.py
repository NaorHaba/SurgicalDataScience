import optuna
from train_experiment_optuna import main


if __name__ == '__main__':
    # HPT_Tune_1
    study = optuna.create_study(study_name = 'HPT_tune_5', direction='maximize')
    # study = optuna.create_study(study_name = 'debug', direction='maximize')
    study.optimize(main)
    # main()  # uncomment to run normaly
