import warnings
warnings.filterwarnings( 'ignore' )

import time
import pandas as pd, numpy as np, gurobipy as gp

import math, os, random, csv

from gurobipy import GRB
from rich.text import Text
from rich import print
import datetime

np.set_printoptions( linewidth = np.inf )

# Columns of the incremental trace checkpoint (mirrors the caller-side
# detected-relationship table). Written every TRACE_FLUSH_EVERY records when a
# ``trace_file`` is supplied to Multicollinear.
TRACE_FILE_HEADERS = [ 'Status', 'Detected_Relationship', 'Norm', 'Coefficient', 'Gap', 'Time' ]
TRACE_FLUSH_EVERY  = 2

def Drop_Perfect( data: pd.DataFrame ):
    # Compute the correlation coefficient matrix
    corr_coef = data.corr().round( 4 )
    
    # Receive the upper triangle elements of the correlation coefficient matrix (not including the diagonal)
    corr_coef = corr_coef.where( np.triu( np.ones_like( corr_coef ), 1 ) == 1 ).to_numpy()
    
    # Receive the index where the correlation coefficient is 1 or -1
    drop_index = np.where( np.abs( corr_coef ) == 1 )
    
    data = data.drop( columns = data.columns[drop_index[0]] )
    
    return data

def Data_Preprocessing( X: pd.DataFrame, drop_first: bool = True, drop_single: bool = True, drop_perfect: bool = True ):
    # 1. Split into numerical and categorical features
    numerical_features = X.select_dtypes( exclude = ['object'] )
    num_col = numerical_features.columns
    categorical_features = X.select_dtypes( include = ['object'] )
    
    # 2. One-hot encode categorical features
    if not categorical_features.empty:
        encoded_categorical_df = pd.get_dummies( categorical_features, dtype = int, drop_first = drop_first )
        
        # 3. Remove encoded columns with only one '1'
        if drop_single:
            unique_cols = encoded_categorical_df.columns[encoded_categorical_df.sum( axis = 0 ) > 1]
            encoded_categorical_df = encoded_categorical_df[unique_cols]
    else:
        encoded_categorical_df = pd.DataFrame( columns = [] )
    
    # 4. Concat numerical and categorical features
    if drop_single:
        unique_cols = encoded_categorical_df.columns[encoded_categorical_df.sum( axis = 0 ) > 1]
        encoded_categorical_df = encoded_categorical_df[unique_cols]
    X = pd.concat( [numerical_features, encoded_categorical_df], axis = 1 )
    
    # 5. Drop constant features
    X = X.loc[:, ( X != X.iloc[0] ).any()]
    
    # 6. Implement Drop_Perfect for perfect collinearity
    if drop_perfect:
        X = Drop_Perfect( X )
    
    
    # 7. Obtain column names for categorical features
    categorical_feature_names = [_ for _ in X.columns if _ not in num_col and _ != 'Intercept']
    
    # 8. Obtain the feature index for the group sparsity
    group_sparsity = []
    for cat_ in categorical_features.columns.tolist():
        temp_ = []
        for col_ in categorical_feature_names:
            if col_.startswith( cat_ ):
                temp_.append( X.columns.get_loc( col_ ) )
        
        if len( temp_ ) < 2: continue
        group_sparsity.append( temp_ )
    
    return X, categorical_feature_names, group_sparsity

def Normalize( X_train: np.array, X_valid: np.array = None, X_test: np.array = None ) -> tuple[np.array, np.array]:
    _, p = X_train.shape
    Normalized_train = np.ones( X_train.shape )
    train_data_mean = np.mean( X_train, axis = 0 )
    
    if X_valid is not None:
        Normalized_valid = np.ones( X_valid.shape )
    else:
        Normalized_valid = None
    
    if X_test is not None:
        Normalized_test = np.ones( X_test.shape )
    else:
        Normalized_test = None
    
    for i in range( p ):
        Normalized_train[:, i] = ( X_train[:, i] - train_data_mean[i] ) / np.linalg.norm( X_train[:, i] - train_data_mean[i] )
        if X_test is not None:
            Normalized_test[:, i] = ( X_test[:, i] - train_data_mean[i] ) / np.linalg.norm( X_train[:, i] - train_data_mean[i] )
        if X_valid is not None:
            Normalized_valid[:, i] = ( X_valid[:, i] - train_data_mean[i] ) / np.linalg.norm( X_train[:, i] - train_data_mean[i] )
    return Normalized_train, Normalized_valid, Normalized_test

def _support_to_feature_str( support, feature_col: np.ndarray ) -> str:
    """
    Convert a list of feature indices to a comma-separated feature-name
    string, e.g. ``[3, 17, 402] -> 'X4, X18, X403'``. Empty support returns
    the empty string.
    """
    if support is None or len( support ) == 0:
        return ''
    return ', '.join( str( feature_col[ int( i ) ] ) for i in support )

def _coef_to_str( A_opt ) -> str:
    """
    Convert an A_opt eigenvector (list of floats) to a comma-separated
    string with six decimals. ``None`` (no inspection ran, or inspection
    failed) passes through as ``None`` so the cell is written empty.
    """
    if A_opt is None:
        return None
    return ', '.join( f'{float( a ):.6f}' for a in A_opt )

def Simulation_Data( n: int, p: int, MR: list, noise_scale: float, rand_seed: int, coef_min: int = 0 ):
    random.seed( int( rand_seed ) )
    rng = np.random.default_rng( seed = rand_seed )
    
    # Generate initial X
    X = rng.normal( loc = 0.0, scale = 1.0, size = ( n, p ) )
    
    noise = rng.normal( loc = 0.0, scale = noise_scale, size = ( n, p ) )
    
    # Total number of features in the multicollinearity relationships
    MR2, MR3, MR4, MR8, MR10, MR15, MR20 = MR
    no_var = 2 * MR2 + 3 * MR3 + 4 * MR4 + 8 * MR8 + 10 * MR10 + 15 * MR15 + 20 * MR20
    
    # Multicollinear relationship coefficients
    gamma = rng.uniform( -10, 10, size = no_var ).tolist()
    # Replace the gamma values that are in [-coef_min, coef_min]
    if coef_min != 0:
        unsatisfied_index = np.where( np.abs( gamma ) < coef_min )[0]
        
        while len( unsatisfied_index ) != 0:
            for index in unsatisfied_index:
                gamma[index] = rng.uniform( -10, 10 )
            unsatisfied_index = np.where( np.abs( gamma ) < coef_min )[0]
    
    # Multicollinear relationship indices
    col = random.sample( [i for i in range( 0, p )], no_var )
    
    # Save the indices and coefficients of the multicollinearity relationships into dict for returning
    indices, coef = {}, {}
    index = 0
    for num, relationships, name in zip( [2, 3, 4, 8, 10, 15, 20], MR, ['MR2', 'MR3', 'MR4', 'MR8', 'MR10', 'MR15', 'MR20'] ):
        indices[name] = []
        coef[name] = []
        for _ in range( relationships ):
            indices[name].append( col[index:index + num] )
            coef[name].append( gamma[index:index + num] )
            
            index += num
    
    # Multicollinear relationships
    for name in ['MR2', 'MR3', 'MR4', 'MR8', 'MR10', 'MR15', 'MR20']:
        for i in range( len( indices[name] ) ):
            X[:, indices[name][i][0]] = np.dot( X[:, indices[name][i][1:]], coef[name][i][1:] )
    
    # Add noise into the whole data
    X = X + noise
    
    # Data normalization
    X, _, _ = Normalize( X_train = X )
    
    feature_col = np.array( [f'X{i + 1}' for i in range( p )] )
    return indices, coef, X, feature_col

class Multicollinear:
    def __init__( self, epsilon: float = None, alpha: float = 0.5, norm_threshold: float = 10 ** (-2)\
                    , Inequality_Inspection: bool = True, Irreducibility_Inspection: bool = True, reduction: bool = True, reduction_method: str = 'eigvec', outlier_threshold: int = None, fastpath: bool = True\
                    , log_file: str = None, log_tag: str = None, trace: list = [], total_time_limit: int = None, per_solve_time_limit: int = None
                    , show_rejected: bool = False, show_accepted: bool = True, trace_file: str = None ):
        self.epsilon = epsilon
        self.alpha = alpha
        self.norm_threshold = norm_threshold
        self.Inequality_Inspection = Inequality_Inspection
        self.Irreducibility_Inspection = Irreducibility_Inspection
        self.reduction = reduction
        if reduction_method not in ( 'corr', 'eigvec' ):
            raise ValueError( f"reduction_method must be 'corr' or 'eigvec', got {reduction_method!r}" )
        self.fastpath = fastpath
        if self.fastpath and not self.Inequality_Inspection:
            raise ValueError( "fastpath=True requires Inequality_Inspection=True: the fast-path can only recover supports that fail the inequality (norm) check." )
        self.reduction_method = reduction_method
        self.outlier_threshold = outlier_threshold
        if self.reduction_method == 'corr' and self.outlier_threshold is None:
            raise ValueError( "outlier_threshold must be specified when reduction_method is 'corr'" )
        self.log_file = log_file
        self.log_tag = log_tag
        self.trace = trace
        self.total_time_limit = total_time_limit
        self.per_solve_time_limit = per_solve_time_limit
        self.show_rejected = show_rejected
        self.show_accepted = show_accepted
        self.trace_file = trace_file
        self._trace_flushed = 0
    
    def _init_trace_file( self ):
        if self.trace_file is None:
            return
        directory = os.path.dirname( self.trace_file )
        if directory:
            os.makedirs( directory, exist_ok = True )
        with open( self.trace_file, 'w', newline = '' ) as f:
            csv.writer( f ).writerow( TRACE_FILE_HEADERS )
        self._trace_flushed = 0
    
    def _format_trace_row( self, entry: dict, col_names: np.ndarray ):
        return {
            'Status'               : entry[ 'status' ],
            'Detected_Relationship': _support_to_feature_str( entry[ 'support' ], col_names ),
            'Norm'                 : entry[ 'norm' ],
            'Coefficient'          : _coef_to_str( entry[ 'A_opt' ] ),
            'Gap'                  : entry[ 'gap' ],
            'Time'                 : entry[ 'time' ],
        }
    
    def _flush_trace( self, col_names: np.ndarray, force: bool = False ):
        if self.trace_file is None:
            return
        pending = self.trace[ self._trace_flushed : ]
        if not pending or ( not force and len( pending ) < TRACE_FLUSH_EVERY ):
            return
        rows = [ self._format_trace_row( e, col_names ) for e in pending ]
        with open( self.trace_file, 'a', newline = '' ) as f:
            csv.DictWriter( f, fieldnames = TRACE_FILE_HEADERS ).writerows( rows )
        self._trace_flushed = len( self.trace )
    
    def _trace_add( self, entry: dict, col_names: np.ndarray ):
        self.trace.append( entry )
        self._flush_trace( col_names )
    
    def _write_log_banner( self, message: str ):
        if self.log_file is None:
            return
        with open( self.log_file, 'a', buffering = 1 ) as f:
            f.write( '\n' + '=' * 80 + '\n' )
            f.write( f'[{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {message}\n' )
            f.write( '=' * 80 + '\n' )
    
    def Reduction_score( self, true_idx: list, reduction_idx: list ):
        ACC, FPR = 0, 0
        ACC = ( len(set(true_idx) & set(reduction_idx)) / len(set(true_idx)) ) * 100
        FPR = ( len(set(reduction_idx) - set(true_idx)) / len(set(reduction_idx)) ) * 100 if len(set(reduction_idx)) > 0 else 0
        return ACC, FPR
    
    def Multicollinear_score( self, z_pos: list, sets: dict ) -> tuple[float, float]:
        total_set = 0
        for _, value in sets.items():
            total_set += len( value )
        
        # Accuracy
        if len( z_pos ) > total_set:
            ACC = 100
        else:
            ACC = ( len( z_pos ) / total_set ) * 100
        
        # False positive rate
        correct = 0
        for z in z_pos:
            if f'MR{len( z )}' not in sets.keys():
                continue
            
            sorted_z = sorted( z )
            for value in sets[f'MR{len( z )}']:
                sorted_set = sorted( value )
                if sorted_z == sorted_set:
                    correct += 1
                    break
        
        if len( z_pos ) == 0:
            FPR = 0
        else:
            FPR = ( 1 - ( correct / len( z_pos ) ) ) * 100
        return ACC, FPR
    
    def Corr_Dimensionality_Reduction( self, X: np.array, outlier_threshold: int = 3 ):
        _, p = X.shape
        
        # Pearson correlation matrix
        C = np.corrcoef( X, rowvar = False )
        
        # Mask the diagonal so it does not participate in the per-column statistics
        C_off = C.copy()
        np.fill_diagonal( C_off, np.nan )
        
        # For each column: index of largest-|corr| partner and that partner's correlation value
        top_partner = np.nanargmax( np.abs( C_off ), axis = 0 )
        top_corr    = C_off[top_partner, np.arange( p )]
        
        # Per-column off-diagonal mean / std.
        mean_c = np.nanmean( C_off, axis = 0 )
        std_c  = np.nanstd(  C_off, axis = 0, ddof = 0 )
        top_norm = np.round( ( top_corr - mean_c ) / std_c, 4 )
        
        # Keep features whose |normalized top-corr| meets the threshold; group by top partner
        keep_mask     = np.abs( top_norm ) >= outlier_threshold
        kept_features = np.where( keep_mask )[0]
        kept_partners = top_partner[kept_features]
        
        reduction_idx = []
        if kept_features.size > 0:
            unique_partners, counts = np.unique( kept_partners, return_counts = True )
            for partner, count in zip( unique_partners, counts ):
                if count > 1:
                    members = kept_features[kept_partners == partner].tolist()
                    reduction_idx.append( int( partner ) )
                    reduction_idx.extend( members )
        
        if len( reduction_idx ) > 0:
            reduction_idx = [int( idx ) for idx in reduction_idx]
            reduction_idx = np.sort( reduction_idx )
            reduction_idx = np.unique( reduction_idx )
            reduction_X = X[:, reduction_idx]
        else:
            reduction_X, reduction_idx = None, None
        return reduction_X, reduction_idx
    
    def Eigvec_Dimensionality_Reduction( self, X: np.array, eig_val: np.array = None, eig_vec: np.array = None ):
        V = self._construct_small_eigvec( X = X, eig_val = eig_val, eig_vec = eig_vec )
        
        # No small eigenvalues -> no near-collinear directions -> nothing to reduce
        if V.size == 0 or V.shape[1] == 0:
            return None, None
        
        # Binary "significant contributor" mask: |V[i, j]| > 1/sqrt(p)
        p_full   = V.shape[0]
        threshold = 1.0 / np.sqrt( p_full )
        significant = np.abs( V ) > threshold
        
        # Keep features that are significant in at least one small eigenvector
        keep_mask     = np.any( significant, axis = 1 )
        kept_features = np.where( keep_mask )[0]
        
        if kept_features.size > 0:
            reduction_idx = [int( idx ) for idx in kept_features]
            reduction_idx = np.sort( reduction_idx )
            reduction_idx = np.unique( reduction_idx )
            reduction_X = X[:, reduction_idx]
        else:
            reduction_X, reduction_idx = None, None
        return reduction_X, reduction_idx
    
    def _construct_small_eigvec( self, X: np.array, spec_epsilon: float = None, eig_val: np.array = None, eig_vec: np.array = None ):
        if eig_val is None or eig_vec is None:
            eig_val, eig_vec = np.linalg.eigh( X.T.dot( X ) )
        
        self.epsilon = spec_epsilon if spec_epsilon is not None else self.alpha * ( ( 1 - np.sqrt( X.shape[1] / (X.shape[0] - 1) ) )**2 )
        
        mask = eig_val < self.epsilon        # Find the indices of the eigenvalues that are smaller than epsilon
        V = eig_vec[:, mask]            # Extract the corresponding eigenvectors 
        return V
    
    def Minimum_Support( self, V: np.array, delta: float = 10 ** ( -4 ), time: float = None, super_exclusions: list = [], exact_exclusions: list = [] ):
        try:
            with gp.Env( empty = True ) as env:
                if self.log_file is not None:
                    env.setParam( 'OutputFlag', 1 )
                    env.setParam( 'LogToConsole', 0 )
                else:
                    env.setParam( 'OutputFlag', 0 )
                
                env.start()
                with gp.Model( "Minimum Support", env = env ) as model:
                    # Setting model parameters
                    model.Params.MIPFocus = 2
                    if time is not None:
                        model.Params.TimeLimit = time
                    if self.log_file is not None:
                        model.Params.LogToConsole = 0
                        model.Params.LogFile      = self.log_file
                    
                    p, m = V.shape # m: the number of small eigenvalue
                    
                    M = 1 / math.sqrt( m )
                    theta = model.addVars( m, vtype = GRB.CONTINUOUS, lb = -M, ub = M, name = "theta" ) 
                    theta_abs = model.addVars( m, vtype = GRB.CONTINUOUS, lb = 0, name = "theta_abs" )
                    z = model.addVars( p, vtype = GRB.BINARY, name = "z" )
                    u = model.addVars( p, vtype = GRB.BINARY, name = "u" )
                    a = model.addVars( p, vtype = GRB.CONTINUOUS, lb = -M, ub = M, name = "a" )
                    
                    model.setObjective( z.sum(), GRB.MINIMIZE ) # Finding the minimum support.
                    
                    model.addConstr( z.sum() >= 2 ) # At least two variables are selected. (Single feature is not considered as multicollinearity)
                    
                    # Linear combination of eigenvectors.
                    for i in range( p ):
                        model.addConstr( a[i] == gp.quicksum( V[i, j] * theta[j] for j in range( m ) ) )
                    
                    # Superset exclusions (accepted supports).
                    if len( super_exclusions ) >= 1:
                        for c in super_exclusions:
                            model.addConstr(  gp.quicksum( z[ c[j] ]  for j in range( len( c ) ) ) <= len( c ) - 1 )
                    
                    # Exact-match-only exclusions (rejected supports).
                    if len( exact_exclusions ) >= 1:
                        for c in exact_exclusions:
                            model.addConstr(
                                len( c ) + gp.quicksum( z[i] for i in range( p ) ) - 2 * gp.quicksum( z[ c[j] ] for j in range( len( c ) ) ) >= 1
                            )
                    
                    # SOS-1 constraint to construct the relationship between z and a.
                    for i in range( p ):
                        model.addConstr( u[i] == 1 - z[i] )
                        model.addSOS( GRB.SOS_TYPE1, [u[i], a[i]], [1, 2] )
                    
                    # Ensuring vector a is non-zero.
                    for j in range( m ): 
                        model.addGenConstrAbs( theta_abs[j], theta[j] )
                    model.addConstr( theta_abs.sum() >= delta )
                    
                    model.optimize()
                    
                    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
                        try:
                            detected_z = np.where( np.array( [ z[i].x for i in range( p ) ] ) > 0.5 )[0].tolist()
                            return detected_z, model.MIPGap, model.Runtime
                        except:
                            print( "Error in extracting z values" )
                            return None, None, None
                    else:
                        return None, None, None
        
        except gp.GurobiError as e:
            print( "Error code" + str( e.errno ) + ": " + str( e ) )
            return False
        
        except AttributeError:
            print( "Encountered an attribute error" )
            model.computeIIS()
            model.write( "min_support.ilp" )
            return False
    
    def _inequality_inspection( self, X: np.array, feature_idx: list ):
        try:
            if len( feature_idx ) == 0:
                return np.inf, []
            
            partial_X = X[:, feature_idx]
            G = partial_X.T @ partial_X
            eig_vals, eig_vecs = np.linalg.eigh( G )
            lam_min = max( float( eig_vals[0] ), 0.0 ) # Avoid numerical error.
            norm    = float( np.sqrt( lam_min ) )
            A_opt   = eig_vecs[:, 0].tolist()
            return norm, A_opt
        except Exception:
            print( 'Encounter an error during inequality inspection process!' )
            return np.inf, [0] * len( feature_idx )
    
    def _irreducibility_inspection( self, X: np.array, feature_idx: list ):
        for i in range( len( feature_idx ) ):
            trial = feature_idx[:i] + feature_idx[i + 1:]
            trial_norm, _ = self._inequality_inspection( X = X, feature_idx = trial )
            if trial_norm <= self.norm_threshold:
                return False
        return True
    
    def Inspection( self, X: np.array, feature_idx: list ):
        pass_ = True
        cause_ = "No inspection performed!"
        norm = np.inf
        A_opt = [0] * len( feature_idx )
        
        # Implement inequality inspection
        if self.Inequality_Inspection:
            norm, A_opt = self._inequality_inspection( X = X, feature_idx = feature_idx )
            if norm > self.norm_threshold:
                pass_ = False
                cause_ = "Inequality inspection failed!"
                return pass_, cause_, norm, A_opt
            else:
                cause_ = ""
        
        # Implement irreducibility inspection
        if self.Irreducibility_Inspection:
            is_irreducible = self._irreducibility_inspection( X = X, feature_idx = feature_idx )
            if not is_irreducible:
                pass_ = False
                cause_ = "Irreducibility inspection failed!"
                return pass_, cause_, norm, A_opt
            else:
                cause_ = ""
        return pass_, cause_, norm, A_opt
    
    def _fast_path( self, X: np.array, start_support: list, start_norm: float, start_A: list, max_iter: int = 15 ):
        fast_exact_excl, fast_norm_rec, fast_A_rec = [], [], []
        current = sorted( int( i ) for i in start_support )
        norm_c, A_c = start_norm, start_A
        for _ in range( max_iter ):
            A_arr = np.asarray( A_c, dtype = float )
            
            residual = X[:, current] @ A_arr
            others = np.setdiff1d( np.arange( X.shape[1] ), np.asarray( current, dtype = int ), assume_unique = True )
            if others.size == 0: # No more features to add
                return None, None, None, fast_exact_excl, fast_norm_rec, fast_A_rec
            
            scores    = np.abs( X[:, others].T @ residual )
            next_cand = int( others[ int( np.argmax( scores ) ) ] )
            current   = sorted( set( current ) | { next_cand } )
            norm_c, A_c = self._inequality_inspection( X = X, feature_idx = current )
            
            if norm_c < self.norm_threshold:
                return current, norm_c, A_c, fast_exact_excl, fast_norm_rec, fast_A_rec
            else:
                fast_exact_excl.append( current ) # Any rejected relationship should be exactly excluded.
                fast_norm_rec.append( norm_c )
                fast_A_rec.append( A_c )
        return None, None, None, fast_exact_excl, fast_norm_rec, fast_A_rec  # No acceptable support found within max_iter.
    
    def _to_reduced( self, support_orig: list, reduction_idx ):
        if reduction_idx is None:
            return sorted( int( i ) for i in support_orig )
        ridx = np.asarray( reduction_idx )
        sup  = np.asarray( sorted( int( i ) for i in support_orig ), dtype = int )
        if sup.size == 0:
            return []
        pos = np.searchsorted( ridx, sup )
        pos = np.clip( pos, 0, len( ridx ) - 1 )
        if np.all( ridx[pos] == sup ):
            return pos.tolist()
        return None
    
    def _shadow_reduced( self, support_orig: list, reduction_idx ):
        if reduction_idx is None:
            return sorted( int( i ) for i in support_orig )
        ridx_set = set( int( v ) for v in np.asarray( reduction_idx ).tolist() )
        inside   = [ int( i ) for i in support_orig if int( i ) in ridx_set ]
        mapped   = self._to_reduced( inside, reduction_idx )
        return mapped if mapped is not None else []
    
    def _resolve_fast_path( self, X: np.array, seed_orig: list, seed_norm: float, seed_A: list, reduction_idx, accepted_supports: list, col_names: np.ndarray ):
        support, _, _, fast_excl, fast_norm, fast_A = self._fast_path(
            X = X, start_support = seed_orig, start_norm = seed_norm, start_A = seed_A )
        
        # Intermediate dead-ends.
        exact_to_add = []
        for inter_id, inter in enumerate(fast_excl):
            print( Text( f'Fast-path candidate {col_names[ inter ]} rejected ({len( inter )} feat).\n( Cause: Inequality inspection failed! )', style = 'yellow' ) ) if self.show_rejected else None
            
            self._trace_add( {
                'status' : 'fast_path_rejected',
                'support': list( inter ),
                'norm'   : float( fast_norm[inter_id] ),
                'A_opt'  : [ float( a ) for a in fast_A[inter_id] ] if not np.isinf( fast_norm[inter_id] ) else None,
                'gap'    : None,
                'time'   : None,
            }, col_names )
            
            red = self._to_reduced( inter, reduction_idx )
            if red is not None:
                exact_to_add.append( red )
        
        if support is None:
            return None, None, exact_to_add, 'fast_path_failed'
        
        # Confirm irreducibility and non-duplication before recording it.
        pass_, cause_, fp_norm, fp_A_opt = self.Inspection( X = X, feature_idx = support )
        accepted_keys  = { frozenset( int( i ) for i in s ) for s in accepted_supports }
        if ( not pass_ ) or ( frozenset( int( i ) for i in support ) in accepted_keys ):
            reason = cause_ if not pass_ else 'Duplicate of an accepted relationship'
            print( Text( f'Fast-path completion {col_names[ support ]} rejected ({len( support )} feat).\n( Cause: {reason} )', style = 'yellow' ) ) if self.show_rejected else None
            
            self._trace_add( {
                'status' : 'fast_path_rejected',
                'support': list( support ),
                'norm'   : float( fp_norm ),
                'A_opt'  : [ float( a ) for a in fp_A_opt ] if not np.isinf( fp_norm ) else None,
                'gap'    : None,
                'time'   : None,
            }, col_names )
            
            red = self._to_reduced( support, reduction_idx )
            if red is not None:
                exact_to_add.append( red )
            return None, None, exact_to_add, 'fast_path_rejected'
        
        # Confirmed, minimal, new relationship.
        red_full = self._to_reduced( support, reduction_idx )
        if red_full is not None:                       # wholly inside reduced set
            return support, red_full, exact_to_add, 'fast_path_accepted'
        
        # Reaches outside the reduced set -> conservative: exact-forbid the shadow.
        shadow = self._shadow_reduced( support, reduction_idx )
        if len( shadow ) > 0:
            exact_to_add.append( shadow )
        return support, None, exact_to_add, 'fast_path_accepted'
    
    def _ablation_loop( self, work_X: np.array, work_V: np.array, reduction_idx, full_X: np.array,
                        col_names: np.ndarray, target_count: int, z_pos_accepted: list, stage: str ):
        if work_V.size == 0 or work_V.shape[1] == 0:
            return
        
        super_excl, exact_excl = [], []
        if self.total_time_limit is None:
            time_limit = 6000
        else:
            time_limit = self.total_time_limit
        iter_ = 0
        while len( z_pos_accepted ) < target_count:
            # Detect relationship
            if time_limit <= 0:
                print( "Time limit reached. Stopping detection." )
                break
            iter_ += 1
            
            if self.per_solve_time_limit is None:
                solve_time = time_limit
            else:
                solve_time = min( time_limit, self.per_solve_time_limit )
            print( f'Iteration {iter_} | Remaining time: {time_limit:.2f} seconds | This solve capped at: {solve_time:.2f} seconds' )
            
            start_ = time.time()
            z_sol, gap, solve_time = self.Minimum_Support( V = work_V, time = solve_time, super_exclusions = super_excl, exact_exclusions = exact_excl )
            end_ = time.time()
            time_limit = int( time_limit - (end_ - start_) )
            
            if z_sol is None or z_sol is False:   # Infeasible (None) or solver error (False).
                print( Text( "Minimum support detection failed.", style = 'bright_cyan' ) )
                break
            super_excl.append( z_sol )
            
            # Map the solver support back to original-column coordinates.
            if reduction_idx is None:
                correct_idx = [ int( _ ) for _ in z_sol ]
            else:
                correct_idx = [ int( _ ) for _ in reduction_idx[z_sol] ]
            
            # Inspect relationship (honours the inequality / irreducibility switches).
            pass_, cause_, norm, A_opt = self.Inspection( X = work_X, feature_idx = z_sol )
            if pass_:
                if cause_ == "No inspection performed!":
                    print( Text( f'Relationship {col_names[ correct_idx ]} detected ({len( correct_idx )} feat).', style = 'pink1' ) ) if self.show_accepted else None
                    z_pos_accepted.append( correct_idx )
                    
                    self._trace_add( {
                        'status' : f'{stage}_detected',
                        'support': list( correct_idx ),
                        'norm'   : float( norm ) if not np.isinf( norm ) else float( 'inf' ),
                        'A_opt'  : [ float( a ) for a in A_opt ] if not np.isinf( norm ) else None,
                        'gap'    : float( gap ) if gap is not None else None,
                        'time'   : float( solve_time ) if solve_time is not None else None
                    }, col_names )
                else:
                    print( Text( f'Relationship {col_names[ correct_idx ]} detected and confirmed ({len( correct_idx )} feat). | A_opt: {np.round( A_opt, 2 )}', style = 'green' ) ) if self.show_accepted else None
                    z_pos_accepted.append( correct_idx )
                    
                    self._trace_add( {
                        'status' : f'{stage}_accepted',
                        'support': list( correct_idx ),
                        'norm'   : float( norm ) if not np.isinf( norm ) else float( 'inf' ),
                        'A_opt'  : [ float( a ) for a in A_opt ] if not np.isinf( norm ) else None,
                        'gap'    : float( gap ) if gap is not None else None,
                        'time'   : float( solve_time ) if solve_time is not None else None
                    }, col_names )
            else:
                print( Text( f'Relationship {col_names[ correct_idx ]} detected but failed confirmation ({len( correct_idx )} feat).\n( Cause: {cause_} )', style = 'red' ) ) if self.show_rejected else None
                
                self._trace_add( {
                    'status' : f'{stage}_rejected',
                    'support': list( correct_idx ),
                    'norm'   : float( norm ) if not np.isinf( norm ) else float( 'inf' ),
                    'A_opt'  : [ float( a ) for a in A_opt ] if not np.isinf( norm ) else None,
                    'gap'    : float( gap ) if gap is not None else None,
                    'time'   : float( solve_time ) if solve_time is not None else None
                }, col_names )
                
                if self.fastpath and cause_ == "Inequality inspection failed!":
                    accepted_support, super_add, exact_add, fp_status = self._resolve_fast_path(
                        X = full_X, seed_orig = correct_idx, seed_norm = norm, seed_A = A_opt,
                        reduction_idx = reduction_idx, accepted_supports = z_pos_accepted, col_names = col_names )
                    for c in exact_add:
                        exact_excl.append( c )
                    if accepted_support is not None:
                        fp_norm, fp_A = self._inequality_inspection( X = full_X, feature_idx = accepted_support )
                        print( Text( f'Relationship {col_names[ accepted_support ]} recovered via fast-path and confirmed ({len( accepted_support )} feat).', style = 'green' ) ) if self.show_accepted else None
                        z_pos_accepted.append( accepted_support )
                        
                        self._trace_add( {
                            'status' : fp_status,
                            'support': list( accepted_support ),
                            'norm'   : float( fp_norm ),
                            'A_opt'  : [ float( a ) for a in fp_A ] if not np.isinf( fp_norm ) else None,
                            'gap'    : None,
                            'time'   : None
                        }, col_names )
                        
                        if super_add is not None:
                            super_excl.append( super_add )
                    else:
                        print( Text( f'Fast-path from {col_names[ correct_idx ]} did not yield a new relationship.', style = 'red' ) ) if self.show_rejected else None
    
    def Ablation_Detection( self, X: np.array, col_names: np.ndarray ):
        n, p = X.shape
        
        eig_val, eig_vec = np.linalg.eigh( X.T @ X )
        
        # Banner so the tailed log file is readable across multiple calls.
        tag_str = f' | {self.log_tag}' if self.log_tag is not None else ''
        self._write_log_banner(
            f'Ablation_Detection START | n={n} p={p} reduction={self.reduction} reduction_method={self.reduction_method!r} '
            f'inequality={self.Inequality_Inspection} irreducibility={self.Irreducibility_Inspection} '
            f'fastpath={self.fastpath}{tag_str}' )
        
        self._init_trace_file()
        
        z_pos_accepted = []
        
        V_orig = self._construct_small_eigvec( X = X, eig_val = eig_val, eig_vec = eig_vec )
        orig_num_of_relationship =  V_orig.shape[1]
        print( f'Number of relationships to be detected (original matrix): {orig_num_of_relationship}' )
        
        use_screen = ( p >= 100 and self.reduction )
        
        # ---- Stage 1: primary detection ----
        if use_screen:
            # Detect on the dimensionality-reduced matrix.
            if self.reduction_method == 'corr':
                reduce_X, reduction_idx = self.Corr_Dimensionality_Reduction( X = X, outlier_threshold = self.outlier_threshold )
            else:
                reduce_X, reduction_idx = self.Eigvec_Dimensionality_Reduction( X = X, eig_val = eig_val, eig_vec = eig_vec )
            
            if reduce_X is None:                 # nothing crossed the reduction threshold
                reduce_X, reduction_idx = X, np.arange( p )
            reduction_idx = np.asarray( reduction_idx )
            
            print( f'Reduced X shape: {reduce_X.shape}' )
            
            V_red = self._construct_small_eigvec( X = reduce_X )
            self._ablation_loop(
                work_X = reduce_X, work_V = V_red, reduction_idx = reduction_idx, full_X = X,
                col_names = col_names, target_count = V_red.shape[1], z_pos_accepted = z_pos_accepted, 
                stage = 'reduced' )
        else:
            # No screen (reduction off, or p < 100): detect directly on the FULL original matrix.
            self._ablation_loop(
                work_X = X, work_V = V_orig, reduction_idx = None, full_X = X,
                col_names = col_names, target_count = orig_num_of_relationship, z_pos_accepted = z_pos_accepted,
                stage = 'full' )
        
        # Final flush: write any remaining ( < 10 ) un-checkpointed records.
        self._flush_trace( col_names, force = True )
        
        return z_pos_accepted
    
    def SMR_Main( self, X: np.array, col_names: np.ndarray ):
        return self.Ablation_Detection( X = X, col_names = col_names )
    
    def Bertsimas_Minimum_Support( self, V: np.array, delta: float = 10 ** ( -4 ), time: float = 6000, super_exclusions: list = [] ):
        try:
            with gp.Env( empty = True ) as env:
                if self.log_file is not None:
                    env.setParam( 'OutputFlag', 1 )
                    env.setParam( 'LogToConsole', 0 )
                else:
                    env.setParam( 'OutputFlag', 0 )
                
                env.start()
                with gp.Model( "Bertsimas Minimum Support", env = env ) as model:
                    if time is not None:
                        model.Params.TimeLimit = time
                    if self.log_file is not None:
                        model.Params.LogToConsole = 0
                        model.Params.LogFile      = self.log_file
                    
                    p, m = V.shape # m: the number of small eigenvalue
                    
                    M = 1 / math.sqrt( m )
                    theta = model.addVars( m, vtype = GRB.CONTINUOUS, lb = -M, ub = M, name = "theta" ) 
                    theta_abs = model.addVars( m, vtype = GRB.CONTINUOUS, lb = 0, name = "theta_abs" )
                    z = model.addVars( p, vtype = GRB.BINARY, name = "z" )
                    a = model.addVars( p, vtype = GRB.CONTINUOUS, lb = -M, ub = M, name = "a" )
                    
                    model.setObjective( z.sum(), GRB.MINIMIZE ) # Finding the minimum support.
                    
                    # Linear combination of eigenvectors.
                    for i in range( p ):
                        model.addConstr( a[i] == gp.quicksum( V[i, j] * theta[j] for j in range( m ) ), name = 'Linear Combination' )
                    
                    # Superset exclusions (accepted supports).
                    if len( super_exclusions ) >= 1:
                        for c in super_exclusions:
                            model.addConstr(  gp.quicksum( z[ c[j] ]  for j in range( len( c ) ) ) <= len( c ) - 1, name = 'Superset exclusion' )
                    
                    # Big-M constraint to construct the relationship between z and a.
                    for i in range( p ):
                        model.addConstr( a[i] <= M * z[i], name = 'Big-M upper' )
                        model.addConstr( a[i] >= -M * z[i], name = 'Big-M lower' )
                    
                    # Ensuring vector a is non-zero.
                    for j in range( m ): 
                        model.addGenConstrAbs( theta_abs[j], theta[j], name = 'Absolute value' )
                    model.addConstr( theta_abs.sum() >= delta, name = 'Non-zero constraint' )
                    
                    model.optimize()
                    
                    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
                        detected_z = np.where( np.array( [ z[i].x for i in range( p ) ] ) > 0.5 )[0].tolist()
                        return detected_z, model.MIPGap, model.Runtime
                    elif model.status == GRB.INFEASIBLE:
                        model.computeIIS()
                        model.write( "min_support.ilp" )
                        return None, None, None
                    else:
                        return None, None, None
        
        except gp.GurobiError as e:
            print( "Error code" + str( e.errno ) + ": " + str( e ) )
            return False
        
        except AttributeError:
            print( "Encountered an attribute error" )
            model.computeIIS()
            model.write( "min_support.ilp" )
            return False
    
    def Bertsimas_Detection( self, X: np.array, col_names: np.ndarray ):
        n, p = X.shape
        
        z_pos_accepted = []
        self._init_trace_file()
        
        V = self._construct_small_eigvec( X = X, spec_epsilon = 10 ** (-2) )
        num_of_relationship = V.shape[1]
        print( f'Number of relationships to be detected: {num_of_relationship}' )
        if self.total_time_limit is None:
            time_limit = 6000
        else:
            time_limit = self.total_time_limit
        
        iter_ = 0
        while len( z_pos_accepted ) < num_of_relationship:
            # Detect relationship
            if time_limit <= 0:
                print( "Time limit reached. Stopping detection." )
                break
            
            iter_ += 1
            
            if self.per_solve_time_limit is None:
                solve_time = time_limit
            else:
                solve_time = min( time_limit, self.per_solve_time_limit )
            print( f'Iteration {iter_} | Remaining time: {time_limit:.2f} seconds | This solve capped at: {solve_time:.2f} seconds' )
            
            start_ = time.time()
            z_sol, gap, solve_time = self.Bertsimas_Minimum_Support( V = V, time = solve_time, super_exclusions = z_pos_accepted )
            end_ = time.time()
            time_limit = int( time_limit - (end_ - start_) )
            
            if len( z_sol ) != 0:
                z_pos_accepted.append( z_sol )
                print( f'Relationship {col_names[ z_sol ]} detected! ({len( z_sol )} feat)' )
                
                _, _, norm, A_opt = self.Inspection( X = X, feature_idx = z_sol )
                
                self._trace_add( {
                        'status' : 'detected',
                        'support': list( z_sol ),
                        'norm'   : float( norm ) if not np.isinf( norm ) else float( 'inf' ),
                        'A_opt'  : [ float( a ) for a in A_opt ] if not np.isinf( norm ) else None,
                        'gap'    : float( gap ) if gap is not None else None,
                        'time'   : float( solve_time ) if solve_time is not None else None
                    }, col_names )
        self._flush_trace( col_names, force = True )
        
        return z_pos_accepted
