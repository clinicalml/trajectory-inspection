import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from trajectoryInspection import mdp_utils as cf

def conv_to_np(this_list):
    this_arr = np.array(this_list)[:, np.newaxis]
    if this_arr.shape == (1, 1):
        return this_arr
    else:
        # Make this idempotent
        this_arr = this_arr.squeeze()[:, np.newaxis]
    return this_arr

def filter_index_data(raw, subsample=None, verbose=False):
    bad_ids = \
    raw.query(("bloc == 1 & "
               "died_within_48h_of_out_time == 1 & "
               "delay_end_of_record_and_discharge_or_death < 24")
             ).icustayid.values

    raw_filtered_no_index = raw[~raw.icustayid.isin(bad_ids)]
    icuuniqueids = raw_filtered_no_index.icustayid.unique()

    if subsample is not None:
        np.random.seed(0)
        icuuniqueids = np.random.choice(
            icuuniqueids, 
            size=np.floor(icuuniqueids.shape[0] * subsample).astype(int),
            replace=False)

    raw_filtered_no_index = raw_filtered_no_index[raw_filtered_no_index.icustayid.isin(icuuniqueids)]

    max_blocs = raw_filtered_no_index.groupby(['icustayid'])['bloc'].max()
    num_blocs = raw_filtered_no_index.groupby(['icustayid'])['bloc'].count()

    if verbose:
        print("Blocs are not contiguous in {.2%} of cases".format(max_blocs != num_blocs).mean())

    contiguous_index = []
    # This wonkiness is so that later, we can set the reward to only occur at the final step...
    not_final_timestep_index = []
    for i in num_blocs.index.values:
        for j in np.arange(num_blocs[i]):
            contiguous_index.append((i, j))
            if j < num_blocs[i] - 1:
                not_final_timestep_index.append((i, j))

    raw_filtered_no_index.index = pd.MultiIndex.from_tuples(
        contiguous_index, names = ['icustayid', 'bloc'])

    raw_filtered = raw_filtered_no_index.drop(['bloc', 'icustayid'], axis=1)
    
    return raw_filtered, not_final_timestep_index

def unnormalize_features(df, colbin, colnorm, collog, scaler):
    colbin_idx = np.where(df.columns.isin(colbin))[0]
    colnorm_idx = np.where(df.columns.isin(colnorm))[0]
    collog_idx = np.where(df.columns.isin(collog))[0]
    cols_idx = np.concatenate([colbin_idx, colnorm_idx, collog_idx])
    
    orig_features = df.copy()
    orig_features.loc[:, colbin] = orig_features.loc[:, colbin] + 0.5
    
    orig_features.loc[:, colnorm + collog] = scaler.inverse_transform(
        orig_features.loc[:, colnorm + collog])
    
    orig_features.loc[:, collog] = np.exp(orig_features.loc[:, collog]) - 0.1    
    
    return orig_features

def normalize_features(df, colbin, colnorm, collog):
    # Numeric indices of columns, used in their script
    colbin_idx = np.where(df.columns.isin(colbin))[0]
    colnorm_idx = np.where(df.columns.isin(colnorm))[0]
    collog_idx = np.where(df.columns.isin(collog))[0]
    cols_idx = np.concatenate([colbin_idx, colnorm_idx, collog_idx])

    raw_features = df.iloc[:, cols_idx] #MIMICraw
    norm_features = raw_features.copy() # MIMICzs

    # Binary features
    norm_features.loc[:, colbin] = norm_features.loc[:, colbin] - 0.5

    scaler = StandardScaler()
    # Convert log vars back to normal using this transformation...
    norm_features.loc[:, collog] = np.log(0.1 + norm_features.loc[:, collog])
    norm_features.loc[:, colnorm + collog] = \
        scaler.fit_transform(norm_features.loc[:, colnorm + collog].astype(float))

    # NOTE: In their script they have these two lines, which I am going to ignore here, because I don't think that the fourth column correponds to what they think it does (The max dose)
    # MIMICzs(:,[4])=log(MIMICzs(:,[ 4])+.6);   % MAX DOSE NORAD 
    # MIMICzs(:,45)=2.*MIMICzs(:,45);   % increase weight of this variable
    
    return norm_features, scaler

# Define functions for converting back and forth from 0-24 action index
def bins_to_action_idx(df):
    iol_bin = df['iol_bin']
    vcl_bin = df['vcl_bin']
    nbin=5

    # Convert to zero index
    iol_idx = iol_bin
    vcl_idx = vcl_bin

    # Add one back at the end
    return (iol_idx * nbin) + vcl_idx

def action_idx_to_bins(action_idx):
    nbin=5

    vcl_idx = action_idx % nbin
    iol_idx = (action_idx - vcl_idx) / nbin

    return iol_idx.astype(int), vcl_idx

def get_actions(raw_filtered):
    '''
    Get the actions from the raw_filtered dataframe, with the specific action columns
    and the bins hard-coded within the function
    '''

    # The actual values
    a = raw_filtered.loc[:, ['input_4hourly', 'max_dose_vaso']].copy()

    iol_bins = np.quantile(a[a.input_4hourly > 0].input_4hourly, [0, 0.25, 0.5, 0.75])
    a['iol_bin'] = np.digitize(a.input_4hourly, iol_bins)

    vcl_bins = np.quantile(a[a.max_dose_vaso > 0].max_dose_vaso, [0, 0.25, 0.5, 0.75])
    a['vcl_bin'] = np.digitize(a.max_dose_vaso, vcl_bins)

    # Get median values for lookup later
    iol_medians = a.groupby(a['iol_bin']).input_4hourly.median()
    vcl_medians = a.groupby(a['vcl_bin']).max_dose_vaso.median()

    a['act_bin'] = bins_to_action_idx(a)
    
    return a, {'iol': {'bins': iol_bins, 'medians': iol_medians},
               'vcl': {'bins': vcl_bins, 'medians': vcl_medians}}


def get_Xya(raw, subsample=None, colbin=None, colnorm=None, collog=None):
    '''
    This proceeds as follows:
    1) convert "raw" file into "raw_filtered", a filtered and indexed file (by icu ID and bloc)
    2) extract normalized features from "raw_filtered"
    3) extract binned actions and outcomes
    
    NOTES:
    * The "blocs" are not continguous in the original data - e.g., ICU number 3 goes from 6 to 9.  We ignore this here on the assumption it does not matter - The code does not seem to treat this as an invariant.
    '''
    assert colbin is not None
    assert colnorm is not None
    assert collog is not None

    #######
    # Filter and extract normalized features
    #
    # a) Remove "bad" ICU stays, and fix the indexing
    # b) Scale all the features based on their type
    #######
    
    raw_filtered, not_final_timestep_index = filter_index_data(raw, subsample=subsample)
    X, scaler = normalize_features(raw_filtered, colbin=colbin, colnorm=colnorm, collog=collog)

    #######
    # Calculate the reward
    #
    # +100 if mortality is 0, -100 if mortality is 1
    #######
    
    y = raw_filtered.loc[:, ['mortality_90d']].copy()
    y['reward'] = 100 * (2 * (1 - y['mortality_90d']) - 1)
    y.loc[not_final_timestep_index, 'reward'] = 0
    
    #######
    # Extract binned actions
    #
    # 25 discrete actions are created from 5 bins of two continous treatments 
    # (max vasopressor dose and fluid input), and the (marginal) median values 
    # within each bin are recorded
    #######
    
    a, bin_info_dict = get_actions(raw_filtered)
    
    return X, y, a, bin_info_dict, scaler


def fill_in_gaps(traj, lives_state_idx, death_state_idx, N_HORIZON=20, FILL=-1, trim_length=True, return_idx=False):
    new_index = pd.MultiIndex.from_product([traj.index.get_level_values('icustayid').unique(), (np.arange(N_HORIZON)).astype(int)], names=['icustayid', 'bloc'])
    old_index = traj.index

    traj = traj.reindex(new_index, fill_value=FILL)
    traj.loc[~traj.reward.isin([100, -100]), 'reward'] = 0
    traj = traj.reindex(
        columns=['action_idx', 'from_state_idx', 'to_state_idx', 'from_hid_idx', 'to_hid_idx', 'reward'],
        fill_value=0)

    # The "to-state" is the from state, moved back one step
    traj.loc[(slice(None), slice(0, N_HORIZON - 2)), 'to_state_idx'] = \
        traj.loc[(slice(None), slice(1, N_HORIZON - 1)), 'from_state_idx'].values.astype(int)

    traj.loc[traj['reward'] > 0, 'to_state_idx'] = lives_state_idx
    traj.loc[traj['reward'] < 0, 'to_state_idx'] = death_state_idx
    
    if trim_length:
        traj = traj.reindex(old_index)
    
    assert not(return_idx and trim_length)
    
    if return_idx:
        return traj.reset_index().drop('icustayid', axis=1).values.reshape(-1, N_HORIZON, 7),\
               traj.index.get_level_values('icustayid').values.reshape(-1, N_HORIZON)
    else:
        return traj

    
def reformat_samples(traj, N_HORIZON=20, FILL=-1, lives_state_idx = 751, death_state_idx=750):
    return fill_in_gaps(traj, lives_state_idx, death_state_idx, N_HORIZON, FILL, trim_length=False, return_idx=True)

def get_traj(X, y, a, subset_index, 
             death_state_idx, lives_state_idx, 
             kmeans, N_HORIZON=20):
    
    ######
    # Train K-Means Model
    #
    # This is done on a subset of the training data
    ######
    
    ######
    # Get the Train / Test splits of relevant variables, 
    # and assign to trajectory dataframe
    ######
    X = X[X.index.get_level_values(0).isin(subset_index)]
    a = a[a.index.get_level_values(0).isin(subset_index)]
    y = y[y.index.get_level_values(0).isin(subset_index)]
    
    # Initialize emptry dataframe, which will look like 
        # Indexed by ICU stay ID and time
        # State (from / to)
        # Action Index (dataframe `a['act_idx']` contains this)
        # Reward (dataframe `y['reward']` contains this)
    
    traj = pd.DataFrame(index = X.index)
    
    traj['from_state_idx'] = kmeans.predict(X)
    traj['to_state_idx'] = np.zeros_like(traj['from_state_idx'])
    traj['action_idx'] = a['act_bin']
    traj['reward'] = y['reward']
    
    traj = fill_in_gaps(traj, lives_state_idx, death_state_idx, N_HORIZON)
    
    return traj

def get_traj_stats(traj, nact, ncl, death_state_idx, lives_state_idx):
    
    ########
    # Raw counts of transitions (Action, FromState, ToState)
    ########
    obs_tx_cts_unadjusted = np.zeros((nact, ncl+2, ncl+2))

    for index, row in traj.iterrows():
        # NOTE: Everything is 1-indexed in matlab, but 0-indexed in numpy...
        assert row['action_idx'] >= 0
        obs_tx_cts_unadjusted[int(row['action_idx']),
                              int(row['from_state_idx']), 
                              int(row['to_state_idx'])] += 1
    ########
    # Adjusted transition matrix
    #
    # 1) Remove low frequency transitions
    # 2) Add absorbing states
    # 3) *Not in original paper* send any unobserved actions to death
    ########        
    # Filter out transitions with less than 5 instances
    low_freq_idx = obs_tx_cts_unadjusted.sum(axis=-1) < 5
    obs_tx_cts_adjusted = obs_tx_cts_unadjusted.copy()
    obs_tx_cts_adjusted[low_freq_idx] = 0

    # Add the death / life absorbing state
    assert obs_tx_cts_adjusted[:, death_state_idx, :].sum() == 0
    assert obs_tx_cts_adjusted[:, lives_state_idx, :].sum() == 0
    obs_tx_cts_adjusted[:, death_state_idx, death_state_idx] = 1
    obs_tx_cts_adjusted[:, lives_state_idx, lives_state_idx] = 1

    # Assume that any such state/action pair leads to instant death
    no_tx_idx = obs_tx_cts_adjusted.sum(axis=-1) == 0
    obs_tx_cts_adjusted[no_tx_idx, death_state_idx] = 1

    obs_tx_mat_adjusted = obs_tx_cts_adjusted / obs_tx_cts_adjusted.sum(axis=-1, keepdims=True)
    assert np.allclose(1, obs_tx_mat_adjusted.sum(axis=-1))
        
    ########
    # Reward matrix (known, given absorbing states)
    ########        
    
    # Create Reward Matrix
    obs_r_mat = np.zeros((nact, ncl+2, ncl+2))
    obs_r_mat[..., death_state_idx] = -100
    obs_r_mat[..., lives_state_idx] = 100

    # No reward once in absorbing state
    obs_r_mat[..., death_state_idx, death_state_idx] = 0
    obs_r_mat[..., lives_state_idx, lives_state_idx] = 0  
    
    ########
    # Initial state distribution
    ########            
    # Get initial state distribution (for generating from the model later)
    t0_rows = traj.index.get_level_values('bloc') == 0
    obs_init_state = traj[t0_rows].groupby('from_state_idx').count()
    obs_init_state = obs_init_state.reindex(np.arange(ncl+2), fill_value=0)
    obs_init_state = obs_init_state.iloc[:, 0].values
    obs_init_state = obs_init_state /  obs_init_state.sum()
    obs_init_state = obs_init_state.squeeze()
    assert np.isclose(1, obs_init_state.sum())

    return (obs_tx_cts_unadjusted, obs_tx_mat_adjusted, 
            obs_r_mat, 
            obs_init_state)


def get_obs_policy(obs_tx_cts, eps):
    nact = obs_tx_cts.shape[0]

    # Ignore the actions *from* absorbing states, we'll add those "Actions" at the end
    no_absorb_obs_tx_cts = obs_tx_cts[:, :-2, :]
    
    # Estimate the observed (behavior) policy
    obs_b = no_absorb_obs_tx_cts.sum(axis=-1)  # Sum over the "to" state, but DON'T ignore death
    obs_b = obs_b.T # Switch from (a, s) to (s, a)
    obs_b_states = obs_b.sum(axis=-1) > 0 # Observed "from" states
    obs_b[obs_b_states] /= obs_b[obs_b_states].sum(axis=-1, keepdims=True)

    # Add back a uniform policy *from* the absorbing states
    absorbed_policy = np.ones((2, nact)) / nact
    obs_b = np.vstack([obs_b, absorbed_policy])
    
    '''
    Softening the behavior policy, as done in AI Clinician
     
    This becomes oddly important in their setup if you use the adjusted counts, as they've already removed all rare actions from the transition matrix, making some of the observed trajectories "impossible" under the behavior policy
    '''
    # NOTE: This was the original approach taken in AI Clinician, but has an error that is obvious in hindsight - If you're small enough, this can push you down past zero
    #actions_not_taken_idx = np.isclose(obs_b, 0)
    #n_actions_not_taken = actions_not_taken_idx.sum(axis=-1)
    #n_actions_taken = obs_b.shape[1] - n_actions_not_taken
    #obs_b_soft = np.copy(obs_b)
  
    #for i in range(obs_b_soft.shape[0]):
    #    if n_actions_not_taken[i] != 0:
    #        old_sum = obs_b_soft[i, :].sum()
    #        obs_b_soft[i, actions_not_taken_idx[i]] += eps / n_actions_not_taken[i]
    #        obs_b_soft[i, ~actions_not_taken_idx[i]] -= eps / n_actions_taken[i]
    #        new_sum = obs_b_soft[i, :].sum()
  
    # INSTEAD we will do the following
    obs_b_soft = np.copy(obs_b)
    actions_not_taken_idx = np.isclose(obs_b, 0)
    obs_b_soft[actions_not_taken_idx] += eps
    obs_b_soft = obs_b_soft / obs_b_soft.sum(axis=-1, keepdims=True)
  
    assert np.all(np.isclose(obs_b_soft.sum(axis=-1), 1))
    if not np.all(obs_b_soft > 0):
        import pdb
        pdb.set_trace()
    assert np.all(obs_b_soft > 0), "Softening failed to provide positivity"

    return obs_b, obs_b_soft


def recover_index(samps, idx, is_cf=False):
    if is_cf:
        assert idx.ndim == 2
        assert samps.ndim == 4
    else:
        assert idx.ndim == 2
        assert samps.ndim == 3

    nobs = samps.shape[0]

    if is_cf:
        tlen = samps.shape[2]
        ncf = samps.shape[1]
        ncf_index = np.transpose(np.arange(ncf) * np.ones((nobs, tlen, 1, ncf)).astype(int), [0, 3, 1, 2])
        joined = np.concatenate((ncf_index, samps), axis=-1)

        # Expand the patient id indices to cover all cf samples
        pt_idx = idx.repeat(ncf).reshape(nobs, ncf, tlen)

        joined = np.concatenate((pt_idx[..., np.newaxis], joined), axis=-1)

        ncols = joined.shape[-1]
        assert ncols == 9
    else:
        joined = np.concatenate((idx[..., np.newaxis], samps), axis=-1)
        ncols = joined.shape[-1]
        assert ncols == 8


    joined = joined.reshape(-1, ncols)

    colnames = ['icustayid']
    if is_cf:
        colnames += ['cf_index', 'bloc']
    else:
        colnames += ['bloc']

    colnames += ['action_idx', 'from_state_idx', 'to_state_idx', 'from_hid_idx', 'to_hid_idx', 'reward'] 

    ret = pd.DataFrame(joined, columns = colnames)
    if is_cf:
        index_set = ['icustayid', 'bloc', 'cf_index']
    else:
        index_set = ['icustayid', 'bloc']

    ret[index_set] = ret[index_set].astype(int)
    ret = ret.set_index(index_set)

    return ret


def traj_to_features(traj, feature_lookup, action_lookup):
    joined = pd.merge(traj, feature_lookup, left_on='from_state_idx', right_index=True, how='left').reindex(traj.index)
    joined = pd.merge(joined, action_lookup[['iol_median', 'vcl_median']], left_on='action_idx', right_index=True, how='left')

    assert joined.shape[0] == traj.shape[0], "{} != {}".format(joined.shape[0], traj.shape[0])

    return joined


def check_rl_policy(rl, obs_tx_cts):
    no_absorb_obs_tx_cts = obs_tx_cts[:, :-2, :-2]

    # re-estimate the observed (behavior) policy
    obs_b = no_absorb_obs_tx_cts.sum(axis=-1)  # Sum over the "to" state, ignore death
    obs_b = obs_b.T # Switch from (a, s) to (s, a)

    # Check if we always observe the RL policy in the non-absorbing states
    prop_rl_obs = (obs_b[rl[:-2, :]==1] > 0).mean()
    return prop_rl_obs, np.isclose(prop_rl_obs, 1)


def create_mdp(traj,
        remove_low_freq=True, insta_death=True, soften=False, factor=1e-4,
        nact=25, ncl=750, death_state_idx=750, lives_state_idx=751,
        enforce_complete=True, return_obs_b=False):
    
    obs_tx_cts_unadjusted, obs_tx_mat_adjusted, obs_r_mat, obs_init_state = \
        get_traj_stats(traj, nact, ncl, death_state_idx, lives_state_idx)

    tx_mat = obs_tx_cts_unadjusted
    
    if remove_low_freq:
        low_freq_idx = tx_mat.sum(axis=-1) < 5  # Threshold used in the paper
        tx_mat[low_freq_idx] = 0
    
    if insta_death:
        # Insta death if we never observe a state-action pair
        no_tx_idx = tx_mat.sum(axis=-1) == 0
        tx_mat[no_tx_idx, death_state_idx] = 1
    
    if soften:
        # Add one psuedo-count of every possible state - action - state tuple
        pseudo_obs = factor * np.ones_like(tx_mat)
        tx_mat = tx_mat + pseudo_obs
        
    # Zero-out and re-add aborbing states
    tx_mat[:, death_state_idx, :] = 0
    tx_mat[:, lives_state_idx, :] = 0
    tx_mat[:, death_state_idx, death_state_idx] = 1
    tx_mat[:, lives_state_idx, lives_state_idx] = 1

    if return_obs_b:
        # Throw away the softened version, but include all changes made thus far
        obs_b, _ = get_obs_policy(tx_mat, eps=0.01)
    
    if enforce_complete:
        assert np.all(tx_mat.sum(axis=-1) > 0)
        tx_mat = tx_mat / tx_mat.sum(axis=-1, keepdims=True)
        assert np.allclose(1, tx_mat.sum(axis=-1))
    else:
        non_zero = tx_mat.sum(axis=-1) > 0
        tx_mat[non_zero] = tx_mat[non_zero] / tx_mat[non_zero].sum(axis=-1, keepdims=True)

    obsMDP = cf.MatrixMDP(tx_mat, obs_r_mat,
                          p_initial_state=obs_init_state)
    
    sampler = cf.BatchSampler(mdp=obsMDP)
    
    if return_obs_b:
        return obsMDP, sampler, obs_b
    else:
        return obsMDP, sampler
