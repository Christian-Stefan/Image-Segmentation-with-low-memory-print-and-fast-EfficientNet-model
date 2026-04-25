### --- START Imports --- ###
import wandb
import pandas as pd
### --- END Imports --- ###


### --- Api Init and hardcoded variables (e.g., project's path; no. of jobs; accumulated kWh)
api = wandb.Api()
PROJECT_PATH = "5lsm0-cityscapes-segmentation" 
runs = api.runs(PROJECT_PATH)

total_wh = 0
processed_runs = 0

print(f"Targeting Metric: system.gpu.0.powerWatts\n" + "-"*30)

for run in runs:
    # 1. Set-up:  We pull the 'system' stream specifically for hardware metrics
    #... Samples=10000 ensures we get enough data points for a good average
    system_metrics = run.history(stream="system", samples=10000) 
    
    # 2. Define graph corresponding env. variable
    target_key = "system.gpu.0.powerWatts"
    
    # 3. Determining the Watts
    if target_key in system_metrics.columns:
        # 3.1. Calculate the mean power usage (Watts)
        avg_power = system_metrics[target_key].mean()
        
        # 3.2 Get duration in hours
        duration_hrs = run.summary.get("_runtime", 0) / 3600
        
        # 3.3 Handler to exclude jobs for which GPU PoerWatts was not calculated
        if avg_power > 0 and duration_hrs > 0:
            run_wh = avg_power * duration_hrs
            total_wh += run_wh
            processed_runs += 1
            # Debug Statement 1
            # print(f"✅ {run.name}: {avg_power:.1f}W for {duration_hrs:.2f}h -> {run_wh:.2f} Wh")
            continue # don't exist the loop if not True

    print(f"❌ {run.name}: Metric not found in system stream.")

if processed_runs > 0:
    print(f"Final Count: {processed_runs} runs processed.")
    print(f"Total Project Energy Burn: {total_wh / 1000:.4f} kWh")
else:
    print("Still no data. Check if the metric name in the 'Expressions' tab uses different slashes or underscores.")