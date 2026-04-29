### --- START imports --- ###
import random
### --- END imports --- ### 

class AdaptiveHyperTuner:
    """
    Motivation: In order to get a tight enough searching space the hyperparameter tunning process revolved around 
    a well-defined mission that regarded finding the best combination of hyperparameters (e.g., 'momentum' and 'weight decay') 
    for the preferred optimizer, which in this case  was RMSprop.

    Def: It enables for a dynamic/adaptive hyperparameter optimization process to be carried out with ease. The process is 
    implemented in a fashion that makes the underlying setup (e.g., baseline parameters) step away from the best known 
    configuration when a more optimal suit is found.

    --- Constructor ---
    :param dict result_dict: container holding the history of trials upon which the decision to update/or not update the best suit of known parameters is being made;
    :param int num_trials: number of trials (e.g., number of distinct pairs to be tried out)
    :param int num_epochs: number of epochs (e.g., for how long one single trial should last?)

    --- Class Function ---
      --record_feedback--
    :param float mom: momentum proposal;
    :param float wd: weight decay proposal;
    :param float score: the averaged (over the epochs) dice score;
    """
    def __init__(self, results_dict: dict, num_trials: int, num_epochs: int):

        self.results_dict = results_dict
        self.num_trials = num_trials
        self.num_epochs = num_epochs
        
        # 1. The Starting Baseline
        self.best_mom:float = 0.90
        self.best_wd:float = 1e-5
        self.best_score:float = 0.0
        
        self.trial_count = 0 

    def get_experiment_setup(self):
        """Proposes a new setup by 'stepping' away from the best known configuration."""

        if self.trial_count == 0:
            # First trial always runs the baseline
            return self.best_mom, self.best_wd, self.num_epochs
            
        # 1. The Adaptive Logic (Hill Climbing) [1](https://www.geeksforgeeks.org/artificial-intelligence/introduction-hill-climbing-artificial-intelligence/)
        # We randomly tweak the BEST known parameters slightly to see if we can improve
        mom_step = random.choice([-0.05, 0.0, 0.05])  # Nudge momentum up, down, or stay
        wd_multiplier = random.choice([0.5, 1.0, 2.0]) # Halve, keep, or double weight decay
        # Apply the steps and keep them within safe mathematical bounds
        next_mom = max(0.0, min(0.99, self.best_mom + mom_step))
        next_wd = self.best_wd * wd_multiplier
        
        return next_mom, next_wd, self.num_epochs
        
    def record_feedback(self, mom, wd, avg_dice):
        """The training loop calls this to report how the trial went."""

        self.trial_count += 1
        self.results_dict[f"Mom={mom:.2f}, WD={wd:.1e}"] = avg_dice
        
        # 3. Updating the "memory"
        if avg_dice > self.best_score:
            print("NEW BEST FOUND (Dice: {:.4f},)" \
            "Updating search center to Mom={:2f}, WD={:.1e}".format(avg_dice, mom, wd))
            # ... updating the underlying setting choices as a new global optima was found
            self.best_score = avg_dice
            self.best_mom = mom
            self.best_wd = wd
        else:
            print("No improvment. Reverting search center back to Mom = {:.2f} and WD={:.1e}".format(self.best_mom, self.best_wd))

    def display_leaderboard(self):
        print("\n--- ADAPTIVE HPO LEADERBOARD ---")
        sorted_results = sorted(self.results_dict.items(), key=lambda item: item[1], reverse=True)
        for rank, (params, score) in enumerate(sorted_results):
            print(f"Rank {rank+1}: {params} -> Avg Dice: {score:.4f}")
