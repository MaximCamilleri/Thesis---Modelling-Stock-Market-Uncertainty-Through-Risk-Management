import joblib
import optuna
from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_param_importances

class LoggingCallback:
    def __init__(self,threshold,trial_number,patience):
        '''
        threshold:int tolerance for increase in objective
        trial_number: int Prune after minimum number of trials
        patience: int patience for the threshold
        '''
        self.threshold = threshold
        self.trial_number  = trial_number
        self.patience = patience
        print(f'Callback threshold {self.threshold}, \
            trial_number {self.trial_number}, \
            patience {self.patience}')
        self.cb_list = [] #Trials list for which threshold is reached
    
    def __call__(self,study:optuna.study, frozen_trial:optuna.Trial):
        #Setting the best value in the current trial
        study.set_user_attr("previous_best_value", study.best_value)
      
        #Checking if the minimum number of trials have pass
        if frozen_trial.number >self.trial_number:
            previous_best_value = study.user_attrs.get("previous_best_value",None)
            #Checking if the previous and current objective values have the same sign
            if previous_best_value * study.best_value >=0:
                #Checking for the threshold condition
                if abs(previous_best_value-study.best_value) < self.threshold:  
                    self.cb_list.append(frozen_trial.number)
                    #If threshold is achieved for the patience amount of time
                    if len(self.cb_list)>self.patience:
                        print('The study stops now...')
                        print('With number',frozen_trial.number ,'and value ',frozen_trial.value)
                        print('The previous and current best values are {} and {} respectively'
                                .format(previous_best_value, study.best_value))
                        study.stop()

class hyperparameter_optimization:
    def __init__(self, alg_name):
        self.alg_name = alg_name
    
    def create_study(self, initial_params):
        self.study = optuna.create_study(
            study_name = self.alg_name + "_study",
            direction = "maximize",
            sampler = optuna.samplers.TPESampler(),
            pruner = optuna.pruners.HyperbandPruner()
        )

        if len(initial_params) > 1:
            self.study.enqueue_trial(initial_params)

    def run_study(self, n_trails, objective):
        logging_callback = LoggingCallback(
            threshold = 1e-5,
            patience = 15,
            trial_number = 100
        )
        
        self.study.optimize(
            objective, 
            n_trials = n_trails, 
            catch = (ValueError,), 
            callbacks = [logging_callback]
        )

        self.save_study(self.alg_name + "_study")

    def save_study(self, study_name):
        joblib.dump(self.study, ("Studies/" + study_name +".pkl"))
    
    def load_study(self, study_name):
        self.study = joblib.load("Studies/" + study_name +".pkl") 
    
    def print_parma_importance(self):
        plot_param_importances(self.study)
    
    def print_optimization_history(self): 
        plot_optimization_history(self.study)
    
    def print_edf(self):
        plot_edf(self.study)

