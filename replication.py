DATE_STR=''

import pandas as pd
import numpy as np
from sklearn.model_selection import ShuffleSplit
from sklearn.cluster import KMeans, MiniBatchKMeans
from copy import deepcopy
from tqdm import tqdm
from time import time
import shelve

import trajectoryInspection.mdp_utils as cf

# Custom configurations, loads things like colbin in the background
from trajectoryInspection.mimic_config import colbin, colnorm, collog
import trajectoryInspection.mimic_utils as utils

import logging as log
log.basicConfig(
        filename='logs/{}-replication.log'.format(DATE_STR), filemode='w',
        format='%(asctime)s - %(levelname)s \t %(message)s',
        level=log.DEBUG)

import argparse
parser = argparse.ArgumentParser(description="Main Replication Script")
parser.add_argument(
    '--full', action='store_true', help='Run in full (not debugging) mode')
parser.add_argument(
    '--use_tqdm', action='store_true', help='Use progress bars')
args = parser.parse_args()

DISCOUNT_Pol=0.99
DISCOUNT = 1.
NSIMSAMPS_RL = 1000
NCFSAMPS=5
N_HORIZON=20

EPS_SOFTEN_BEHAVIOR=0.01 # How much by which to soften the *behaviour* policy
EPS_SOFTEN_TX=0.01 # Add this much over all never-seen transitions for CFs

nra = 5
nact = nra**2  # Note: DO NOT CHANGE nact or nra, as it won't be reflected in the bins

# Set to false to run the full script, otherwise use small numbers (e.g., number of clusters)
DEBUG=not(args.full)
USE_TQDM=args.use_tqdm

ncl = 750
nclustering = 32
nr_reps = 500  # 500 in the original paper
prop = 0.25
USE_BOOTSTRAP = True # Note, this ONLY applies to WIS to get 95% LB, and test WIS
N_BOOTSTRAP = 750 # Note, this ONLY applies to WIS to get 95% LB, and test WIS
minibatch = False # Use minibatch k-means
subsample = None  # New var, subsample the whole dataset

if DEBUG:
    ncl = 750
    nclustering = 32
    nr_reps = 1 # Much fewer reps
    prop = 0.25
    USE_BOOTSTRAP = True
    N_BOOTSTRAP = 750
    minibatch = True # Use minibatch k-means
    subsample = None

log.info(("USING ARGUMENTS:"
          "\n\t ncl: {}"
          "\n\t nclustering: {}"
          "\n\t nr_reps: {}"
          "\n\t prop: {}"
          "\n\t USE_BOOTSTRAP: {}"
          "\n\t N_BOOTSTRAP: {}"
          "\n\t minibatch: {}"
          "\n\t subsample: {}").format(
              ncl, nclustering, nr_reps,
              prop, USE_BOOTSTRAP, N_BOOTSTRAP,
              minibatch, subsample)
          )

death_state_idx = ncl
lives_state_idx = ncl+1

# Import the dataframe
fpath = '<REPLACE WITH DATA PATH>'
# Export the results
datapath = '<REPLACE WITH OUTPUT PATH>'

log.info("Loading data from file")
raw = pd.read_csv("{}/mimic-table.csv".format(fpath)) #MIMICtable

data_dict = shelve.open("{}/{}-data_dict.db".format(datapath, DATE_STR))

# NOTE: There is a weird (mistaken column?) normalization they do, see def
log.info("Filtering and normalizing data")
X, y, a, bin_info_dict, scaler = utils.get_Xya(
        raw, subsample=subsample, colbin=colbin, colnorm=colnorm, collog=collog)
icuuniqueids = X.index.get_level_values('icustayid').unique()

data_dict['X'] = X
data_dict['y'] = y
data_dict['a'] = a
data_dict['bin_info_dict'] = bin_info_dict
data_dict['scaler'] = scaler

np.random.seed(0)
test_ids = np.random.choice(icuuniqueids, np.floor(len(icuuniqueids) * prop).astype(int))

# Filter down to the train/validation set
train_val_ids = icuuniqueids[~icuuniqueids.isin(test_ids)]
test_ids = icuuniqueids[icuuniqueids.isin(test_ids)]

data_dict['train_val_ids'] = train_val_ids
data_dict['test_ids'] = test_ids

# Initialize K-Means algorithm
obs_rewards = []
wis_rewards = []
obs_ho_rewards = []
wis_ho_rewards = []
wis_ho_rewards_lb = []
mb_rewards = []

best_result_dict = {}

best_ho_wis_lb = -100.
final_kmeans = None

# Train / Validation split on ICU Stay IDs
np.random.seed(0)
rs = ShuffleSplit(n_splits=nr_reps, test_size=.20, random_state=0)
i = 0

log.info("Starting primary loop")
for train_index, val_index in tqdm(rs.split(train_val_ids), total=nr_reps,
        disable=not(USE_TQDM), desc="Main Loop over {} models".format(nr_reps)):
    i += 1
    if not(USE_TQDM):
        log.info("Iteration {}/{}".format(i, nr_reps))

    np.random.seed(i)
    if minibatch:
        kmeans = MiniBatchKMeans(n_clusters=ncl,
                                 random_state=i,
                                 n_init=nclustering,
                                 batch_size=ncl*3)
    else:
        kmeans = KMeans(n_clusters=ncl,
                        random_state=i,
                        n_init=nclustering,
                        n_jobs=nclustering)

    train_index = train_val_ids[train_index]
    val_index = train_val_ids[val_index]

    # Training subset for k-means
    n_sub_idx = np.floor(train_index.shape[0]*prop).astype(int)
    sub_idx = np.random.choice(train_index, n_sub_idx)

    kmeans = kmeans.fit(X[X.index.get_level_values(0).isin(sub_idx)])

    # Predict with kmeans and get trajectories formatted correctly
    traj_train = utils.get_traj(X, y, a, train_index,
        death_state_idx, lives_state_idx, kmeans)
    traj_val = utils.get_traj(X, y, a, val_index,
        death_state_idx, lives_state_idx, kmeans)

    obs_tx_cts_unadjusted, obs_tx_mat_adjusted, obs_r_mat, obs_init_state = \
        utils.get_traj_stats(traj_train, nact, ncl, death_state_idx, lives_state_idx)

    # Learn an initial policy!
    obsMDP = cf.MatrixMDP(obs_tx_mat_adjusted, obs_r_mat,
                          p_initial_state=obs_init_state)

    assert np.allclose(obs_tx_mat_adjusted.sum(axis=-1), 1)
    RlPol = obsMDP.policyIteration(discount=DISCOUNT_Pol, skip_check=True)

    check_prop, check = utils.check_rl_policy(RlPol, obs_tx_cts_unadjusted)
    if check is False:
        log.error(
            "Warning: {} of RL actions were never taken!".format(check_prop))

    # Get a soft version of the RL policy for WIS
    RlPolSoft = np.copy(RlPol).astype(float)
    RlPolSoft[RlPolSoft == 1] = 0.99
    RlPolSoft[RlPolSoft == 0] = 0.01 / (nact - 1)

    # If you would rather do what the AI clinician did (truncate transitions in 
    # the WIS, as well as in the learned MDP) then replace this with obs_tx_cts_adjusted
    obs_b, obs_b_soft = utils.get_obs_policy(
            obs_tx_cts_unadjusted, eps=EPS_SOFTEN_BEHAVIOR)

    # Step 1: Reformat the trajectories as a numpy array with 
    # (N_trajectories x N_HORIZON x 7 features)  so that they have N_HORIZON 
    # steps each
    train_samps, _ = utils.reformat_samples(traj_train, N_HORIZON=N_HORIZON)
    val_samps, _ = utils.reformat_samples(traj_val, N_HORIZON=N_HORIZON)

    obs_reward = cf.eval_on_policy(
        train_samps, discount=DISCOUNT)

    wis_reward, _, _ = cf.eval_wis(
        train_samps, discount=DISCOUNT,
        obs_policy=obs_b_soft, new_policy=RlPolSoft)

    obs_ho_reward = cf.eval_on_policy(
        val_samps, discount=DISCOUNT)

    wis_ho_reward_boot,  _, _ = cf.eval_wis(
        val_samps, discount=DISCOUNT,
        bootstrap=USE_BOOTSTRAP, n_bootstrap=N_BOOTSTRAP,
        obs_policy=obs_b_soft, new_policy=RlPolSoft)

    if USE_BOOTSTRAP:
        wis_ho_reward = wis_ho_reward_boot.mean()
        wis_ho_reward_lb = np.quantile(wis_ho_reward_boot, 0.05)
    else:
        wis_ho_reward = wis_ho_reward_boot
        wis_ho_reward_lb = None

    BSampler = cf.BatchSampler(mdp=obsMDP)
    this_mb_samples_opt = BSampler.on_policy_sample(
        policy=RlPol, n_steps=N_HORIZON, n_samps=NSIMSAMPS_RL,
        use_tqdm=False, tqdm_desc='Model-Based OPE')

    mb_reward = cf.eval_on_policy(
        this_mb_samples_opt, discount=DISCOUNT)

    obs_rewards.append(obs_reward)
    wis_rewards.append(wis_reward)
    obs_ho_rewards.append(obs_ho_reward)
    wis_ho_rewards.append(wis_ho_reward)
    wis_ho_rewards_lb.append(wis_ho_reward_lb)
    mb_rewards.append(mb_reward)

    if wis_ho_reward_lb > best_ho_wis_lb:
        log.info("Found best WIS LB of {:.4f} at iteration {}".format(
            wis_ho_reward_lb, i))
        best_ho_wis_lb = wis_ho_reward_lb
        best_result_dict['kmeans'] = deepcopy(kmeans)
        best_result_dict['obs_b'] = np.copy(obs_b)
        best_result_dict['obs_b_soft'] = np.copy(obs_b_soft)
        best_result_dict['rl_pol'] = np.copy(RlPol)
        best_result_dict['rl_pol_soft'] = np.copy(RlPolSoft)
        best_result_dict['traj_train'] = deepcopy(traj_train)
        best_result_dict['train_samps'] = np.copy(train_samps)
        best_result_dict['traj_val'] = deepcopy(traj_val)
        best_result_dict['val_samps'] = np.copy(val_samps)

data_dict['best_results'] = best_result_dict
data_dict['best_ho_wis_lb'] = best_ho_wis_lb

# Note that this is *not* re-done on the entire train/val set, per what is done
# in the original paper
final_kmeans = best_result_dict['kmeans']
final_obs_b_soft = best_result_dict['obs_b_soft']
final_RlPolSoft = best_result_dict['rl_pol_soft']

# Record the overall reward statistics
obs_rewards = utils.conv_to_np(obs_rewards)
wis_rewards = utils.conv_to_np(wis_rewards)
obs_ho_rewards = utils.conv_to_np(obs_ho_rewards)
wis_ho_rewards = utils.conv_to_np(wis_ho_rewards)
wis_ho_rewards_lb = utils.conv_to_np(wis_ho_rewards_lb)
mb_rewards = utils.conv_to_np(mb_rewards)

data_dict['obs_rewards'] = obs_rewards
data_dict['wis_rewards'] = wis_rewards
data_dict['obs_ho_rewards'] = obs_ho_rewards
data_dict['wis_ho_rewards'] = wis_ho_rewards
data_dict['wis_ho_rewards_lb'] = wis_ho_rewards_lb
data_dict['mb_rewards'] = mb_rewards

# Pull out the test samples and check the WIS reward
traj_test = utils.get_traj(X, y, a, test_ids,
    death_state_idx, lives_state_idx, final_kmeans)
test_samps, test_idx = utils.reformat_samples(traj_test, N_HORIZON=N_HORIZON)

wis_test_reward_boot, _, _ = cf.eval_wis(
    test_samps, discount=DISCOUNT,
    bootstrap=USE_BOOTSTRAP, n_bootstrap=N_BOOTSTRAP,
    obs_policy=final_obs_b_soft, new_policy=final_RlPolSoft)

if USE_BOOTSTRAP:
    wis_test_reward = wis_test_reward_boot.mean()
    wis_test_reward_lb = np.quantile(wis_test_reward_boot, 0.05)
else:
    wis_ho_reward = wis_test_reward_boot
    wis_ho_reward_lb = None

data_dict['traj_test'] = traj_test
data_dict['wis_test_reward'] = wis_test_reward
data_dict['wis_test_reward_lb'] = wis_test_reward_lb

# Load the training trajectories, and calculate the feature lookup
traj_train = best_result_dict['traj_train']
feature_lookup = utils.unnormalize_features(
    X.loc[traj_train.index].groupby(traj_train['from_state_idx']).median(), 
    colbin, colnorm, collog, scaler)

action_lookup = pd.DataFrame(np.array(
    utils.action_idx_to_bins(np.arange(25))).T, columns=['iol', 'vcl'])

for i in range(nra):
    action_lookup.loc[action_lookup.iol == i, 'iol_median'] = bin_info_dict['iol']['medians'][i]
    action_lookup.loc[action_lookup.vcl == i, 'vcl_median'] = bin_info_dict['vcl']['medians'][i]

data_dict['feature_lookup'] = feature_lookup
data_dict['action_lookup'] = action_lookup

# NUMPY formatted samples
data_dict['test_samps'] = test_samps

# PANDAS formatted samples
data_dict['traj_test'] = traj_test

test_idx_flat = test_idx[:, 0]
traj_test_full = utils.traj_to_features(traj_test, feature_lookup, action_lookup)

data_dict['test_idx_flat'] = test_idx_flat
data_dict['traj_test_full'] = traj_test_full

log.info("Done")
