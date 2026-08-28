"""
main.py
=======
Entry point for reproducing the Enhanced Holistic Regression (EHR)
multicollinearity-detection experiments. Set BASE_DIR (results are written
there and the folder is created automatically), then choose which simulations
and which data scales to run.

Data scales follow Table 2 of the manuscript. The MR list is ordered
[MR2, MR3, MR4, MR8, MR10, MR15, MR20]; MR2 = 0 as the table has no MR(2).
"""

import os

import Simulation as Sim

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# Root for all results. Change this to wherever you want the output to go.
BASE_DIR = os.path.join( os.path.dirname( os.path.abspath( __file__ ) ), 'Results' )

# Folder holding the pre-processed real-world datasets (the *_Processed.csv files).
DATA_DIR = os.path.join( os.path.dirname( os.path.abspath( __file__ ) ), 'Data' )

# --------------------------------------------------------------------------- #
# Data settings (Table 2): [MR2, MR3, MR4, MR8, MR10, MR15, MR20]
# --------------------------------------------------------------------------- #
SCALES = {
    'small': {  'n': 2000,  'p': 1000,  'MR': [0, 5, 3, 1, 1, 0, 0], 'outlier_threshold': 4, \
                'run_A': True, 'run_B': True, 'run_C': False, 'run_D': False, 'run_E': False, \
                'run_SMR_Corr': True, 'run_SMR_Eigen': True, 'run_Bertsimas': True },
    'large': {  'n': 20000, 'p': 10000, 'MR': [0, 3, 3, 1, 1, 1, 1], 'outlier_threshold': 5, \
                'run_A': False, 'run_B': False, 'run_C': False, 'run_D': False, 'run_E': False, \
                'run_SMR_Corr': True, 'run_SMR_Eigen': True, 'run_Bertsimas': True },
}

# --------------------------------------------------------------------------- #
# What to run
# --------------------------------------------------------------------------- #
RUN_SCALES     = ['small', 'large']   # which data scales to run
RUN_ABLATION   = True
RUN_REDUCTION  = True
RUN_SMR_COMPARISON   = True
RUN_REALWORLD  = True

# Coefficient-sweep experiment: SMR Corr vs SMR Eigvec across coef_min = 0.0..1.0 on the large-scale dataset.
COEF_SWEEP_SCALE     = 'large'
COEF_SWEEP_MINS      = [round( 0.1 * i, 1 ) for i in range( 11 )]   # 0.0, 0.1, ..., 1.0

# Shared simulation settings.
NOISE_SCALE = 0.01
COEF_MINS   = [0.0]
SIMULATIONS = 10
SEED        = 911122
SAVE_TRACE  = False   # detection: also write a per-method/simulation trace CSV


if __name__ == '__main__':
    os.makedirs( BASE_DIR, exist_ok = True )
    
    for scale in RUN_SCALES:
        cfg = SCALES[scale]
        n, p, MR, outlier_threshold = cfg['n'], cfg['p'], cfg['MR'], cfg['outlier_threshold']
        print( f'\n########## {scale}-scale | n={n} p={p} MR={MR} outlier_threshold={outlier_threshold} ##########' )
        
        # ---- 1. Synthetic detection ablation study + significance test ---- #
        if RUN_ABLATION:
            Sim.run_ablation_simulation(
                BASE_DIR = BASE_DIR,
                n = n, p = p, MR = MR,
                noise_scale = NOISE_SCALE,
                coef_mins = COEF_MINS,
                simulations = SIMULATIONS,
                seed = SEED,
                outlier_threshold = outlier_threshold,
                run_A = cfg['run_A'], run_B = cfg['run_B'], run_C = cfg['run_C'], run_D = cfg['run_D'], run_E = cfg['run_E'],
                run_SMR_Corr = cfg['run_SMR_Corr'], run_SMR_Eigen = cfg['run_SMR_Eigen'], run_Bertsimas = cfg['run_Bertsimas'],
                save_trace = SAVE_TRACE,
                run_significance = False,
                significance_baseline = 'Original',
            )
        
        # ---- 2. Dimensionality-reduction screen comparison ---- #
        if RUN_REDUCTION:
            Sim.run_reduction_simulation(
                BASE_DIR = BASE_DIR,
                n = n, p = p, MR = MR,
                noise_scale = NOISE_SCALE,
                coef_mins = COEF_MINS,
                simulations = SIMULATIONS,
                seed = SEED,
            )
    
    # ---- Coefficient sweep: SMR-Corr vs SMR-Eigvec, coef_min 0.0..1.0 ---- #
    if RUN_SMR_COMPARISON:
        cfg = SCALES[COEF_SWEEP_SCALE]
        print( f'\n########## coef-sweep | {COEF_SWEEP_SCALE}-scale | coef_min={COEF_SWEEP_MINS} ##########' )
        Sim.run_SMR_comparison(
            BASE_DIR = BASE_DIR,
            n = cfg['n'], p = cfg['p'], MR = cfg['MR'],
            noise_scale = NOISE_SCALE,
            coef_mins = COEF_SWEEP_MINS,
            simulations = SIMULATIONS,
            seed = SEED,
            outlier_threshold = cfg['outlier_threshold'],
            save_trace = SAVE_TRACE,
            csv_id = 'CoefSweep',
        )
    
    # ---- 3. Real-world detection (pre-processed datasets) ---- #
    if RUN_REALWORLD:
        Sim.run_realworld_detection(
            BASE_DIR = BASE_DIR,
            data_dir = DATA_DIR,
            datasets = None,              # None => every *_Processed.csv in DATA_DIR
            run_SMR = True,
            run_bertsimas = True,
            reduction = False,
            total_time_limit = 86400,
            per_solve_time_limit = 100,
        )
