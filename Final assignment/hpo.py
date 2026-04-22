import random

class AdaptiveHyperTuner:
    def __init__(self, results_dict: dict, num_trials: int, num_epochs: int):
        self.results_dict = results_dict
        self.num_trials = num_trials
        self.num_epochs = num_epochs
        
        # 1. The Starting Baseline
        self.best_mom = 0.90
        self.best_wd = 1e-5
        self.best_score = 0.0
        
        self.trial_count = 0

    def get_experiment_setup(self):
        """Proposes a new setup by 'stepping' away from the best known configuration."""
        if self.trial_count == 0:
            # First trial always runs the baseline
            return self.best_mom, self.best_wd, self.num_epochs
            
        # 2. The Adaptive Logic (Hill Climbing)
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
        
        # 3. The "Memory" Update
        if avg_dice > self.best_score:
            print(f"\n🏆 NEW BEST FOUND! (Dice: {avg_dice:.4f})")
            print(f"Updating search center to Mom={mom:.2f}, WD={wd:.1e}")
            self.best_score = avg_dice
            self.best_mom = mom
            self.best_wd = wd
        else:
            print(f"\n❌ No improvement. Reverting search center back to Mom={self.best_mom:.2f}, WD={self.best_wd:.1e}")

    def display_leaderboard(self):
        print("\n--- ADAPTIVE HPO LEADERBOARD ---")
        sorted_results = sorted(self.results_dict.items(), key=lambda item: item[1], reverse=True)
        for rank, (params, score) in enumerate(sorted_results):
            print(f"Rank {rank+1}: {params} -> Avg Dice: {score:.4f}")