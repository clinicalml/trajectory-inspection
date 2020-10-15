# Trajectory inspection: A Method for Iterative Clinician-Driven Design of Reinforcement Learning Studies

This repository contains the code for reproducing our paper [here](https://arxiv.org/abs/2010.04279). We use data from MIMIC-III, and the model we inspect is from the AI Clinician paper at https://www.nature.com/articles/s41591-018-0213-5.

To run this repository,
1. **Set up the environment**: Run `bash setup_env.sh`. Each time you use this repository, start with `conda activate trajectory-inspection`.
2. **Install MATLAB**: We used MATLAB 2017b.
3. **Set configuration variables**: The following files contain configuration variables which are specific to setup (e.g., path names, the name of the database, etc). These will need to be modified to fit your set-up.  Running `grep REPLACE -r .` in the home directory should surface these instances.
    * 3 preprocessing files specified in README in `notebooks/preprocess_data` subdirectory
    * All the notebooks in `notebooks/`
    * `replication.py`

To reproduce the AI Clinician,
1. **Preprocess the raw MIMIC-III data into trajectories**: The folder `notebooks/preprocess_data/` contains these scripts. These scripts generate the trajectories in a `.csv` format.
2. **Learn the RL policy**: `replication.py` is our re-implementation of `AIClinician_core_160219.m` (from the original repository in Python. The results are saved to disk. Note that this script takes a fair amount of time to run, as (per the original paper) it must loop over 500 different discrete state and action MDP models.

To reproduce the results in our paper on trajectory investigation,
1. **Heuristic that investigates surprisingly positive outcomes** (Figure 3c): `notebooks/SurprisinglyPositiveOutcomes.ipynb`
2. **Heuristic that investigates surprisingly aggressive treatments** (Figure 3a-b): `notebooks/SurprisinglyAggressiveTreatments.ipynb`
3. **Reviewing medical notes** (Figure 4): `notebooks/ChartReview.ipynb`
4. **Trajectory plots** (Figure 5): `notebooks/TrajectoryPlots.ipynb`
5. **Censoring investigation** (Figure 6-7): `notebooks/Censoring.ipynb`

Code acknowledgments:
1. The code in `notebooks/preprocess_data/` includes code from [matthieukomorowski/AI_Clinician](https://github.com/matthieukomorowski/AI_Clinician), which is described in `Komorowski M, Celi LA, Badawi O, Gordon AC, Faisal AA. The artificial intelligence clinician learns optimal treatment strategies for sepsis in intensive care. Nature medicine. 2018 Nov;24(11):1716-20`. We reproduce it here with parts converted to Python scripts and other minor changes.

2. In `pymdptoolbox`, we have the source code for the `pymdptoolbox` package from [sawcordwell/pymdptoolbox](https://github.com/sawcordwell/pymdptoolbox), which is in turn based the toolset described in `Chades I, Chapron G, Cros M-J, Garcia F & Sabbadin R (2014) 'MDPtoolbox: a multi-platform toolbox to solve stochastic dynamic programming problems', Ecography, vol. 37, no. 9, pp. 916–920, doi 10.1111/ecog.00888`. We reproduce it here because we needed to make a slight modification to the `mdp` class to bypass certain checks; in particular, it checks for whether or not the rows of the transition matrix sum to one, but can fail due to floating-point inaccuraries - we replace this check in the main code with an assertion using `np.allclose` instead of checking for strict equality.

3. The code in `notebooks/ChartReview.ipynb` was largely written for a visualization project by Anu Vajapey, Willie Boag, Emily Alsentzer, and Matthew McDermott.
