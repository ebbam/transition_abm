


# FULL OMN - RUN ALL SCRIPTS FOR FULL OMN
N = 25
python3 auto_run_model_modified.py
python3 diagnose_wages.py
python3 diagnose_wasserstein.py
python3 analyze_ltuer_ts_fit.py
python3 entry_policy_exercise.py


# ALL NETWORKS - PARAMETER RESULTS AND STEADY STATE FIGURES
N = 1
python3 auto_calibration_summary.py
python3 auto_run_model_modified.py run = "steady_state"


# OTHER NETWORKS - RUN WITH N10 
N = 10
python3 auto_run_model_modified.py run = "covid_oos"
python3 diagnose_wages.py
python3 diagnose_wasserstein.py
python3 analyze_ltuer_ts_fit.py

