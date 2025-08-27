
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import math 
import scipy.optimize as op
import networkx as nx
import seaborn as sns
import warnings


def spreadsheet_to_pandas(ruta):
    topics = pd.read_excel(ruta, sheet_name="topics")
    modules = pd.read_excel(ruta, sheet_name="modules", dtype={"topics": str})
    professors = pd.read_excel(ruta, sheet_name="professors", dtype={"topics": str})
    students = pd.read_excel(ruta, sheet_name="students", dtype={"Preferred": str, "Disliked": str})

    # Convertir columnas string en listas de strings
    students["Preferred"] = students["Preferred"].str.split(",")
    students["Disliked"] = students["Disliked"].str.split(",")
    professors["topics"] = professors["topics"].str.split(",")
    modules["topics"] = modules["topics"].str.split(",")

    print("Successful pandas upload")
    return topics, modules, professors, students

def pandas_to_numpy(topics, modules, professors, students):

    students_np = students.copy()
    professors_np = professors.copy()
    modules_np = modules.copy()

    # Convertir listas de strings a arrays de enteros
    students_np["Preferred"] = students_np["Preferred"].apply(lambda x: np.array(x, dtype=int))
    students_np["Disliked"] = students_np["Disliked"].apply(lambda x: np.array(x, dtype=int))
    professors_np["topics"] = professors_np["topics"].apply(lambda x: np.array(x, dtype=int))
    modules_np["topics"] = modules_np["topics"].apply(lambda x: np.array(x, dtype=int))

    # Otras columnas
    cap = np.array(professors_np["Capacity"])
    s_stream = np.where(np.array(students_np["stream"]) == 1)[0]
    p_stream = np.where(np.array(professors_np["stream"]) == 1)[0]

    return topics, modules_np, professors_np, students_np, cap, s_stream, p_stream

def upload_validation(ruta):
    # Cargar
    topics, modules, professors, students = spreadsheet_to_pandas(ruta)

    # Convertir
    try:
        topics, modules_np, professors_np, students_np, cap, s_stream, p_stream = pandas_to_numpy(
            topics, modules, professors, students
        )
    except ValueError as e:
        raise ValueError(f"Data contains non integers: {e}")

    # Validación 1: Ver que todos los arrays contengan enteros
    for col in ["Preferred", "Disliked"]:
        if not students_np[col].apply(lambda x: x.dtype.kind == 'i').all():
            raise ValueError(f"Column {col} has non integers")

    for col in ["topics"]:
        if not professors_np[col].apply(lambda x: x.dtype.kind == 'i').all():
            raise ValueError("Professor Column 'topics' has non integers")
        if not modules_np[col].apply(lambda x: x.dtype.kind == 'i').all():
            raise ValueError(" Module Column  'topics' has non integers")

    # Validación 2: Capacidad total debe ser mayor que número de estudiantes
    if cap.sum() < len(students_np):
        raise ValueError("INVALID DATA Less projects than students")
    
    

    # Validación 3: Streams solo deben tener 0 o 1
    if not set(students_np["stream"]).issubset({0, 1}):
        raise ValueError("Columna 'stream' de estudiantes contiene valores distintos de 0 o 1")
    if not set(professors_np["stream"]).issubset({0, 1}):
        raise ValueError("Columna 'stream' de profesores contiene valores distintos de 0 o 1")

    print("✔ Valid data, proceed with optimization")

    # Warning
    if len(s_stream) >  cap[p_stream].sum():
        warnings.warn(
            f"WARNING: More stream students ({len(s_stream)}) than stream projects ({len(p_stream)}). Do not intend to match with Stream restriction",
            UserWarning
        )

    return topics, modules_np, professors_np, students_np, cap, s_stream, p_stream

def cos_sim(a,b):

    return (a.T @ b) / (np.linalg.norm(a)*np.linalg.norm(b))

def sp_preference(s, p, max_rate, mod_num, method = "angle"):

    s = np.unique(s.astype(int)-1)
    p = np.unique(p.astype(int)-1)

    #s = s.astype(int)-1
    #p = p.astype(int)-1

    s_len = len(s)
    p_len = len(p)

    assert s_len <= max_rate, "more rated professors than permited"


    s_rate = np.arange(max_rate, max_rate - s_len, -1)

    s_vec = np.zeros(mod_num)
    p_vec = np.zeros(mod_num)

    s_vec[s] = s_rate
    p_vec[p] = 1

   

    if method == "sum":
        return(np.sum(s_vec*p_vec))
    elif method == "angle":

        #angle = (s_vec.T @ p_vec) / (np.linalg.norm(p_vec)*np.linalg.norm(s_vec))
        angle = cos_sim(s_vec, p_vec)
        
        if angle == 0: ## orthogonal
            return 0.
        else:
            return  angle 
            
        #return((s_vec.T @ p_vec) / (np.linalg.norm(p_vec)*np.linalg.norm(s_vec)))


def preference_matrix(s,p, max_rate, mod_num, s_block= None, method = "angle"):
    """
        Gets the pandas series for students and professors.
    """

    s_len = len(s)
    p_len = len(p)


    mat = np.zeros((s_len, p_len))

    if s_block is not None:
    
        for i in range(0, s_len):

            for j in range(0, p_len):

                ## check incompatibility
                sb = s_block.iloc[i].astype(int)-1
                pb = p.iloc[j].astype(int)-1
                if np.intersect1d(sb, pb).size == 0:
                    mat[i,j] = sp_preference(s.iloc[i], p.iloc[j], max_rate, mod_num, method )
    else:
        for i in range(0, s_len):

            for j in range(0, p_len):

                mat[i,j] = sp_preference(s.iloc[i], p.iloc[j], max_rate, mod_num, method )
    
    return mat


def preference_matrix2(s,p, max_rate, mod_num, method = "angle"):
    """
        Gets the pandas series for students and professors.
    """

    s_len = len(s)
    p_len = len(p)

    mat = np.zeros((s_len, p_len))

    for i in range(0, s_len):
        
        for j in range(0, p_len):
            
            mat[i,j] = sp_preference(s.iloc[i], p.iloc[j], max_rate, mod_num, method )


    return mat



def top_affinities(S, k):
    """Keeps only the top-k values per row, sets the rest to zero."""
    output = np.zeros_like(S)
    row_indices = np.arange(S.shape[0])[:, None]

    # Get indices of top-k values per row
    topk_indices = np.argpartition(-S, kth=k-1, axis=1)[:, :k]

    # Fill output with the top-k values
    output[row_indices, topk_indices] = S[row_indices, topk_indices]

    return output
    

def drop_blocks(S, mask):

    #s_len = len(s_block)
    #p_len = len(p)
    #for i in range(0, s_len):
    #    
    #    for j in range(0, p_len):
    #        sb = s_block.iloc[i].astype(int)-1
    #        pb = p.iloc[j].astype(int)-1
    #        if np.intersect1d(sb, pb).size > 0:
    #            S[i,j] = 0
    if S.shape != mask.shape:
        raise ValueError(f"Shape mismatch: S is {S.shape}, mask is {mask.shape}")
    
    S[mask == 1] = 0

    return S

def extract_matrix(S, s, p ):
    """ 
        Separates matrix in 4:
            stream students with stream professors
            stream students with non stream professors
            no stream students with stream professors
            non stream students with non stream professors

    """
    s = np.asarray(s).astype(int)
    p = np.asarray(p).astype(int)

    s_stream = np.where( np.array(s) == 1 )[0]
    p_stream = np.where( np.array(p) == 1 )[0]
    ns_stream = np.where( np.array(s) == 0 )[0]
    np_stream = np.where( np.array(p) == 0 )[0]


    return S[np.ix_(s_stream, p_stream)], S[np.ix_(s_stream, np_stream)], S[np.ix_(ns_stream, p_stream)], S[np.ix_(ns_stream, np_stream)]
    
def enhance_matrix(S, s, p, eps = 1e-1):
    _, SNP, NSP, NSNP = extract_matrix(S, s, p)

    # Convert s and p into index arrays
    s = np.asarray(s).astype(int)
    p = np.asarray(p).astype(int)
    
    s_stream = np.where(s == 1)[0]
    p_stream = np.where(p == 1)[0]
    
    S2 = S.copy()
    S2[np.ix_(s_stream, p_stream)] = (
        S[np.ix_(s_stream, p_stream)]
        + max(np.max(SNP), np.max(NSP), np.max(NSNP))
        + eps
    )
    
    return S2



def unpack_vars(x, n, m, k):
    U = x[:n*k].reshape(n, k)
    V = x[n*k:].reshape(m, k)

    
    #U = x[:n*k].reshape(n, k)
    #V = x[n*k:n*k+m*k].reshape(m, k)

    return U, V

def weighted_loss(x0, n,m,k, A, w0=.05, l =.01):
    """
        total_loss = sum_{observed} (a_ij - u_i*v_j) + w0 (sum_{ij} u_i*v_j + lambda (norm_fro(U + V))
    """

    U,V = unpack_vars(x0, n, m, k)
    A_hat = U @ V.T
    observed_mask = (A != 0)

    term1 = np.sum((A[observed_mask] - A_hat[observed_mask])**2)
    term2 = w0 * np.sum(A_hat[~observed_mask]**2) 
    term3 = l * (np.sum(U**2) + np.sum(V**2))

    total_loss = term1 + term2 + term3

    return total_loss


def components_suggestion(A, threshold = 1, plot = False):

    _, S_svd, _ = np.linalg.svd(A, full_matrices=False)
    var_explained = (S_svd**2) / np.sum(S_svd**2)
    var_cumulative = np.cumsum(var_explained)

    if plot:
        fig, ax1 = plt.subplots(figsize=(10, 6))
        x = np.arange(1, len(S_svd) + 1)
        ax1.bar(x, var_explained, label='Explained Variance', alpha=0.6, color='#7b68ee')
        ax1.set_xlabel('Component')
        ax1.set_ylabel('Explained Variance')
        ax1.set_ylim(0, 1)
        ax2 = ax1.twinx()
        ax2.plot(x, var_cumulative, label='Accumulated Variacne', color="#40e0d0", marker='o')
        ax2.set_ylabel('Accumulated Variance')
        ax2.set_ylim(0, 1.05)
        fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
        plt.grid(True, axis='y', linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.show()

        if threshold < 1:
            return np.min(np. where(var_cumulative > threshold))
        else:
            return len(S_svd)
            
        

    


def recommendation_affinities(A, k = 20, w0=.05, l=.01, followup = False):

    n,m = A.shape


    U_svd, S_svd, Vt_svd = np.linalg.svd(A, full_matrices=False)
    U_k = U_svd[:, :k]
    S_k = np.sqrt(S_svd[:k])  
    V_k = Vt_svd[:k, :].T

    U0 = np.abs(U_k * S_k)
    V0 = np.abs(V_k * S_k)


    x0 = np.concatenate([U0.ravel(), V0.ravel()])
    noneg = [(0, None)] * len(x0)

    # Optimization
    result = op.minimize(weighted_loss, x0, args=(n, m, k, A, w0, l ), 
                         method='L-BFGS-B', 
                         bounds = noneg, 
                         options={'maxfun': 10000000,  # o más si hace falta
                        'maxiter': 500000,  # no es el principal en L-BFGS-B, pero mantenlo
                        'eps': 1e-4,
                        'gtol': 1e-4,
                        'ftol': 1e-4,
                        'disp': followup}
                        )

    # Reshape results
    U_opt, V_opt = unpack_vars(result.x, n, m, k)

    print(result.message)
    print(result.success)
    
    return U_opt@V_opt.T, U_opt, V_opt

def cold_start_fix(U, V,P, top = None):
    n = len(P)
    aux = np.where (np.sum(V, axis = 1) == 0)[0]

    V2 = V.copy()
    P2 = P.copy()

    np.fill_diagonal(P2, 0)
    if top == None:
        top = n
    

    for k, i in enumerate(aux):
        
        reduced = np.zeros(n)
        index = np.argpartition(P2[i,:], -top)[-top:] 
        reduced[index] = P2[i,:][index]
        reduced = reduced/np.sum(reduced)

        V2[i,:] = reduced @ V2 ## average weight of most similar professors

    return U@V2.T, U, V2

def affinity_correl(A, B):
    return np.corrcoef(A.ravel(), B.ravel())[0, 1]

def affinity_correl_obs(A, B):
    mask = A > 0
    a_vals = A[mask]
    b_vals = B[mask]
    if a_vals.size < 2:
        return np.nan  # No hay suficientes datos para calcular correlación
    return np.corrcoef(a_vals, b_vals)[0, 1]

def prof_relation(p, mod_num):
    """
        Gets the pandas series for students and professors.
    """

 
    p_len = len(p)
    mat = np.zeros((p_len,p_len))

    for i in range(0, p_len):
        
        for j in range(0, p_len):

            p1 = p.iloc[i].astype(int)-1
            p2 = p.iloc[j].astype(int)-1

            p1_vec = np.zeros(mod_num)
            p2_vec = np.zeros(mod_num)

            p1_vec[p1] = 1
            p2_vec[p2] = 1
            mat[i,j] = cos_sim(p1_vec, p2_vec)

    return mat


def recommendation_profs(pref1, P):
    # Producto normal
    #dot_product = pref1 @ P.T
    
    mask = (pref1[:, :, None] * P.T[None, :, :]) != 0  
    contrib = pref1[:, :, None] * P.T[None, :, :]  
    summed = contrib.sum(axis=1) 
    count_nonzero = mask.sum(axis=1) 
    count_nonzero[count_nonzero == 0] = 1
    averaged = summed / count_nonzero
    
    return averaged

def plot_sim(cos_sim_matrix_thresh, weight=5, highlight_edges=None):
    """
    Plot network graph from similarity matrix.
    
    Parameters:
    cos_sim_matrix_thresh: numpy array - similarity matrix
    weight: float - scaling factor for edge widths
    highlight_edges: list of tuples/sets - edges to highlight, e.g., [(0,1), (2,3)] or [{0,1}, {2,3}]
    """
    
    # Set diagonal to zero to remove self-loops
    cos_sim_matrix_no_diag = cos_sim_matrix_thresh.copy()
    np.fill_diagonal(cos_sim_matrix_no_diag, 0)
    
    # Create graph from modified matrix
    G = nx.from_numpy_array(cos_sim_matrix_no_diag)
    
    # Set default highlight_edges if none provided
    if highlight_edges is None:
        highlight_edges = []
    
    # Convert highlight_edges to sets for easier comparison
    highlight_edges_sets = []
    for edge in highlight_edges:
        if isinstance(edge, (tuple, list)):
            highlight_edges_sets.append({edge[0], edge[1]})
        elif isinstance(edge, set):
            highlight_edges_sets.append(edge)
    
    edge_weights = nx.get_edge_attributes(G, 'weight')
    edge_widths = [weight * edge_weights[(u, v)] for u, v in G.edges()]
    
    # Plotting
    pos = nx.circular_layout(G)
    center = np.mean(list(pos.values()), axis=0)
    plt.figure(figsize=(6, 6))
    nx.draw_networkx_nodes(G, pos, node_color='#7b68ee', node_size=400, node_shape="s")
    
    for (u, v), w in zip(G.edges(), edge_widths):
        rad = 0.1 
        midpoint = (pos[u] + pos[v]) / 2
        # Vector from midpoint to center
        to_center = center - midpoint
        # Vector from u to v
        edge_vec = pos[v] - pos[u]
        # Cross product (z-component) to check side
        cross = np.cross(edge_vec, to_center)
        rad = -0.1 if cross > 0 else 0.1
        
        # Check if current edge should be highlighted
        current_edge = {u, v}
        is_highlighted = current_edge in highlight_edges_sets
        
        if is_highlighted:
            nx.draw_networkx_edges(
                G, pos,
                edgelist=[(u, v)],
                width=w,
                connectionstyle=f'arc3,rad={rad}',
                arrows=True,
                edge_color="#40e0d0", 
                alpha=.9
            )
        else:
            nx.draw_networkx_edges(
                G, pos,
                edgelist=[(u, v)],
                width=w,
                connectionstyle=f'arc3,rad={rad}',
                arrows=True,
                edge_color="#708090", 
                alpha=.2
            )
    
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold', font_color="#e6e6fa")
    
    plt.axis('off')
    plt.tight_layout()
    plt.title("Professors complementarity graph")
    plt.savefig("professors_network.jpg", dpi=300, bbox_inches='tight')
    plt.show()

def incompabilities(dislikes, professors, threshold = 0):


    m = len(dislikes)
    n = len(professors)
    S = np.zeros((m,n), dtype = int)

    for i in range(0,m):
        for j in range(0,n):
            sb = dislikes.iloc[i].astype(int)-1
            pb = professors.iloc[j].astype(int)-1
            if (1.0*np.intersect1d(sb, pb).size)/len(pb) > threshold:
                S[i,j] = (1.0*np.intersect1d(sb, pb).size)/len(pb)
                S[i,j] = 1
                

    return S

def restrictions(s_num, p_num, S = None, s_stream = [], p_stream = []):
    #print("hello 1", S)
    #print(len(s_stream))
    extra_rows = 0
    if len(s_stream) > 0:
        extra_rows += 1  # One row for stream constraints
    if S is not None:
        extra_rows += 1  # One row for incompatibility constraints

    if len(s_stream) > 0:
        if S is not None:
            stream_index = -2  # Second to last row when both constraints present
        else:
            stream_index = -1  # Last row when only stream constraint present
    
    students = np.zeros((s_num + extra_rows, s_num*p_num), dtype = int)
    professors = np.zeros((p_num, s_num*p_num), dtype = int)

    #print(students.shape)
    if len(s_stream) > 0:
        ps_vec = np.zeros(p_num)

    for p in range(0, p_num):

        for s in range(0, s_num):

            professors[p, s*p_num + p] = 1

        if len(s_stream) > 0 and p in p_stream:
            ps_vec[p] = 1

    for i in range(0, s_num):

        students[i, i*p_num:i*p_num+p_num] = 1

        if len(s_stream) > 0 and i in s_stream:
            students[stream_index, i*p_num:i*p_num+p_num] = ps_vec

    if S is not None:
        ##print("hello", S)
        students[-1, :] = S.flatten()      
    

    bounds = [(0, 1)] * (s_num*p_num)
    

    return students, professors, bounds

def LPA(S, cap, S_block = None, s_stream = [], p_stream = [], sens = False):

    s_num, p_num = np.shape(S)
    smat, pmat, bounds = restrictions(s_num, p_num, S = S_block, s_stream = s_stream, p_stream = p_stream)

    

    cap2 = list(np.ones(s_num))

    if len(s_stream) > 0:
        cap2.append(len(s_stream))

    if S_block is not None:
        cap2.append(0)

    cap2 = np.array(cap2)

    solution = op.linprog(-S.flatten(),pmat, cap, smat, cap2, bounds,method="highs")


    assert solution.success, "Unsuccesful optimization, review constraints"

    
    assignment_mat =solution.x.reshape(s_num, p_num)
    assignment = np.argmax(assignment_mat, axis=1)
    prof_dict = {}
    for i in range(0,p_num):
        prof_dict[i] = np.array([], dtype = int)

    for i in range(0,s_num):
        prof_dict[assignment[i]] = np.append(prof_dict[assignment[i]],i) 


    if sens == False:
        return assignment, prof_dict
    else:
        return assignment, prof_dict, solution.ineqlin    
    

## SPA

def assignation_check(S_assign, S_work):

    unnasigned = np.where(S_assign == -1)[0]
    S_unnasigned = S_work[unnasigned,:] ## Indices with of student without assigned projects
    S_with_list = np.where(np.sum(S_unnasigned, axis =1) > 0 )[0]

    return unnasigned, S_with_list

def spa_sym(S, capacities, max_iter = 500, followup = True):
    ## this is built for non symmetric potentials, maybe 
    ## If S,P are the same but transposed, we may not need this two arguments.


    assert np.sum(capacities) >= len(S), 'not enough projects for students'
    

    S_num, P_num = S.shape
    ##P_num = len(P)
    ##proj_sum = np.sum(P[:,-1]) 

    
    ##initialize counters and state savers

    P_cap = capacities ## array to count capacities
    S_assign = np.zeros(S_num,dtype = int)-1 ## array to save professor assignation
    S_work = S.copy() ## I think I will need to alter S, but I'm not sure at this point
    unnasigned, S_with_list = assignation_check(S_assign, S_work)

    if followup:
        print("Student assignments initialized", S_assign)

    
    prof_dict = {}

    for i in range(0,P_num):
        prof_dict[i] = np.array([], dtype = int)

    if followup:
        print("Assignments initialized", prof_dict)


    iter = 0

    
    while len(unnasigned)> 0 and len(S_with_list) and iter < max_iter:
        if followup:
            print("\n Iteration = ", iter)
        to_assign_students =  unnasigned[S_with_list].astype(int)       

        ## unnasigned students apply to each professor
        for i, index in enumerate(to_assign_students):
            
            prof = np.argmax(S_work[index,:])
            prof_dict[prof] = np.append(prof_dict[prof],index) 
            S_assign[index] = prof ## propose a provisional match
        
        if followup:
            print("\nnew temporal assignments", prof_dict)

        for i in range(0,P_num):

            if len(prof_dict[i]) > P_cap[i]:

                
                #print(np.argsort(-P[i,prof_dict[i]])  ) ## order based on suitability
                
                order = np.argsort(-S_work.T[i,prof_dict[i]]) 
                remaining_students = prof_dict[i][order[0:P_cap[i]] ]
                rejected_students =  prof_dict[i][order[P_cap[i]:]  ]
                

                for _, index in enumerate(rejected_students):
                    S_work[index, i] = 0 ## block rejected students
                    S_assign[index] = -1 ## break the assignment    
                
                prof_dict[i] = remaining_students

            if len(prof_dict[i]) == P_cap[i]:
                ## professor is already on cap, 
                ## This will always trigger if the previous if is triggered, but may trigger without it.
                
#
                min_score = np.min(S_work.T[i,prof_dict[i]]) ## by definintion S.T[i,prof_dict[i]] is greater than zero
                mask = S_work[:,i] < min_score
                S_work[mask,i] = 0
            ### end of if required to update the assignation
        ##end of for of professors               
        unnasigned, S_with_list = assignation_check(S_assign, S_work)
        
        if followup:
            print("iteration results")
            print("assignment", S_assign)
            print("prof assignment", prof_dict)
            print("unnasigned", unnasigned)
        iter = iter + 1

    return S_assign, prof_dict 


## Auxiliary functions

def envy_matrix(assignment, S):

    s_num, _ = np.shape(S)
    envy = np.zeros((s_num, s_num), dtype = int)


    for i in range(0, s_num):
        pk = assignment[i]
        aik = S[i, pk] ## affinity of student i with assigned professor k

        for j in range(i, s_num):
            pl = assignment[j]
            ajl = S[j,pl] ## affinity of student j with assigned professor l

            ## crossed affinities
            ail = S[i,pl]
            ajk = S[j,pk]

            if ail > aik: ## i would prefer j's professor
                envy[i,j] = 1

            if ajk > ajl: ## j would prefer i's professor
                envy[j,i] = 1

    return envy

def mutual_envy(assignment, S):

    assert len(assignment) == len(S), "lengths of assignments and affinites doesn't match" 
    
    envy_mat = envy_matrix(assignment, S)

    cross = envy_mat + envy_mat.T

    rows, cols = np.where(cross == 2)

    stable = (len(rows) == 0)
    return stable, rows, cols



def optimality_score(assignment_rand, S_rand, fulloptim = False, followup = False):

    assert len(assignment_rand) == len(S_rand), "lengths of assignments and affinites doesn't match"

    S_rand = S_rand.astype(float)
    sum_cap = 0
    sum_al = 0
    n_students = 0
    for i in range(0, len(assignment_rand)):
        
        if fulloptim:
            if followup:
                print(assignment_rand[i], S_rand[i, assignment_rand[i]])
            sum_cap +=  0 if assignment_rand[i] == -1 else S_rand[i, assignment_rand[i]]
            sum_al += np.average(S_rand[i,:])
            n_students += 1
            

        else:
            if assignment_rand[i] != -1:
                sum_cap += S_rand[i, assignment_rand[i]]
                sum_al += np.average(S_rand[i,:])
                n_students += 1

    return sum_cap/n_students, sum_al/n_students


def assignment_differences(ass1, ass2, S):

    assert len(ass1) == len(ass2), "different set of students"
    diff_assign = {} 
    diff_score = {}
    for i in range(0, len(ass1)):

        if ass1[i] != ass2[i]:
            diff_assign[i] = (ass1[i] ,ass2[i] )
            diff_score[i] = (S[i,ass1[i]], S[i,ass2[i]])

    return diff_assign, diff_score

def professor_has_ties(S):

    _,p_num = np.shape(S)
    ties = np.full((p_num, ), False, dtype=bool)

    for i in range(0, p_num):
        if len(np.where(np.unique(S[:,i], return_counts = True)[1] != 1 )[0]) != 0:
            ties[i] = True

    return ties, np.where(ties == True)

def is_stable(assignment, prof_dict, S, caps):

    n_num, p_num = np.shape(S)

    assert len(assignment) == n_num, "lengths of assignments and affinites don't match"
    assert len(prof_dict) == p_num, "lengths of professors and affinites don't match"
    assert len(prof_dict) == len(caps), "lengths of assignments and capacities don't match"

    bps = []

    #the loop will review all possible assignments

    for i in range(0, n_num):
        for j in range(0, p_num):

            if assignment[i] == j:
                continue ## do not check actual matches
                
            sij = S[i,j]
            if assignment[i] == -1:
                sma = 0 ## the student was unnasigned
            else:
                sma = S[i, assignment[i]] ## actual affinity of student match
            
            if sij <= sma:
                continue ## student i wont prefer j over current assignment.
            else:

                prof_list =prof_dict[j]

                if len(prof_list) < caps[j]:
                    ## professor has capacity also, it catches the case where prof_list is empty
                    
                        bps.append((i,j))
                else:
                    ##sij is greater than minimum affinity of professor assigned students. prof list is always non empty here
                    ## With all non negative preferences.
                    ## if sij is greater than the min, is greater than at least one of the assigned students to pj
                    ## if sij is greater than at least on of the assigned students, then is greater than the mean, 
                    ## hence, this condition is equivalent.
                    if sij > np.min(S[prof_list,j]):
                        bps.append((i,j))
    
    return len(bps) == 0, bps


def is_envy_free(assignment, S, print_graph = False,return_cycles=False):

    assert len(assignment) == len(S), "lengths of assignments and affinites doesn't match" 
    
    envy_mat = envy_matrix(assignment, S)

    envy_graph = nx.from_numpy_array(envy_mat, create_using=nx.DiGraph)

    is_free = nx.is_directed_acyclic_graph(envy_graph)

    if print_graph:
        pos = nx.circular_layout(envy_graph)

        # Find cycles if not envy-free
        cycles = list(nx.simple_cycles(envy_graph)) if not is_free else []

        # Collect nodes and edges involved in cycles
        cycle_nodes = set()
        cycle_edges = []
        for cycle in cycles:
            cycle_nodes.update(cycle)
            for i in range(len(cycle)):
                src = cycle[i]
                tgt = cycle[(i + 1) % len(cycle)]
                cycle_edges.append((src, tgt))

        # --- Draw all nodes ---
        node_colors = ['salmon' if node in cycle_nodes else 'lightblue' for node in envy_graph.nodes]
        nx.draw_networkx_nodes(envy_graph, pos, node_color=node_colors)

        # --- Draw all edges ---
        normal_edges = [e for e in envy_graph.edges if e not in cycle_edges]
        nx.draw_networkx_edges(envy_graph, pos, edgelist=normal_edges, edge_color='gray', arrows=True)
        nx.draw_networkx_edges(envy_graph, pos, edgelist=cycle_edges, edge_color='red', arrows=True)

        # --- Draw labels ---
        nx.draw_networkx_labels(envy_graph, pos)

        title = "Envy Graph (No Cycles)" if is_free else "Envy Graph (Cycles Highlighted)"
        plt.title(title)
        plt.show()

    if return_cycles and not is_free:
        cycles = list(nx.simple_cycles(envy_graph))
        return is_free, cycles
    else:
        return is_free,[]
    
def is_restricted_envy_free(ass, S, s, p):

    s = np.array(s)
    p = np.array(p)
    envy_free, envies = is_envy_free(ass,np.round(S, decimals = 7),False, True)


    if not envy_free:
        envy_free = True

        for i in range(0, len(envies)):

            cyc = np.array(envies[i])
            env = np.roll(cyc, -1)
            
            mask = s[cyc] == 1 ## check all the students on the cycle that are part of the stream
            stream_permute = p[ass[env[mask]]]
            if len(stream_permute) == np.sum(stream_permute): ## if making the rotation respects the stream restriction, then, envy is possible

                envy_free = False
                break


    return envy_free
    
 

def critical_professors(sens, caps, increase = 1):

    crit_prof = np.where(sens.marginals != 0.)[0]
    caps_inc =  np.ceil((-1.)*sens.marginals).astype(int)
    caps2 = caps.copy()

    for i, prof in enumerate(crit_prof):
        caps2[prof] = caps2[prof] + increase

    return crit_prof,  caps2, caps_inc


def sensitivity_improvement(S_rand, cap_rand, increase = 1, followup = False, max_iter = 500):
    
    assert increase > 0, "requires a positive increase"
    cap = cap_rand.copy()
    caps_inc = np.ones_like(cap_rand)
    iter = 0
    
    
    while np.sum(caps_inc) > 0 and iter < max_iter:
        
        _, _, sens = LPA(S_rand, cap, True)   
        _, cap_aux, caps_inc = critical_professors(sens,cap,increase)

        if np.array_equal(cap,cap_aux):
            break ## new polytope resulted in the same optimum
        else:
            cap = cap_aux

        if followup:
            print("final caps at tier ", iter, "= ",cap)

        iter += 1

    cap_change = np.clip( cap - sens.residual.astype(int),1,10000000) - cap_rand
    crit_prof = np.where(cap_change != 0)
    
    return crit_prof, np.clip( cap - sens.residual.astype(int),1,10000000), cap_change, sens
    ##cap, cap_change, sens##


def fulfills_stream(s, p, ass):
    """
    Pass the full list, with zeros and ones as gotten from the base spreadsheet.
    """
    
    s_stream = np.where( np.array(s) == 1 )[0]
    ass_stream = ass[s_stream] ## only the assignment of students in the stream 
    
    if np.any(ass_stream == -1):
        print("some stream students were unnasigned")
        ass_stream = ass_stream[ass_stream != -1]
        #return False
    
    aux = np.array(p[ass_stream])

    ##aux = np.array(p[ass] ) 

    return not np.any(aux == 0)
    

def extract_matrix(S, s, p ):
    """ 
        Separates matrix in 4:
            stream students with stream professors
            stream students with non stream professors
            no stream students with stream professors
            non stream students with non stream professors

    """
    s = np.asarray(s).astype(int)
    p = np.asarray(p).astype(int)

    s_stream = np.where( np.array(s) == 1 )[0]
    p_stream = np.where( np.array(p) == 1 )[0]
    ns_stream = np.where( np.array(s) == 0 )[0]
    np_stream = np.where( np.array(p) == 0 )[0]


    return S[np.ix_(s_stream, p_stream)], S[np.ix_(s_stream, np_stream)], S[np.ix_(ns_stream, p_stream)], S[np.ix_(ns_stream, np_stream)]


def bad_assignment(ass, S = []):

    
    if len(S) == 0:
        missassignment = np.where(ass == -1)[0]
    else:
        missassignment = []
        for i, j in enumerate(ass):

            if S[i,j] == 0:
                missassignment.append(i)

    return np.array(missassignment)




def plot_density(S, save_name=None, title=None):
    B = nx.Graph()

    # Add nodes for each set
    n_students, n_profs = S.shape
    students_2 = [f"Student_{i}" for i in range(n_students)]
    professors_2 = [f"Prof_{j}" for j in range(n_profs)]

    B.add_nodes_from(students_2, bipartite=0)
    B.add_nodes_from(professors_2, bipartite=1)

    # Add edges with weights from the matrix
    for i in range(n_students):
        for j in range(n_profs):
            if S[i, j] != 0:
                B.add_edge(students_2[i], professors_2[j], weight=S[i, j])

    # Compute positions
    student_y = np.linspace(0, 1, len(students_2))
    professor_y = np.linspace(0, 1, len(professors_2))

    pos = {}
    pos.update({s: (0, y) for s, y in zip(students_2, student_y)})
    pos.update({p: (1, y) for p, y in zip(professors_2, professor_y)})

    # Identify nodes with and without edges
    student_nodes = set(students_2)
    professor_nodes = set(professors_2)

    connected_nodes = set(n for n in B.nodes if B.degree(n) > 0)
    isolated_students = student_nodes - connected_nodes
    connected_students = student_nodes & connected_nodes

    isolated_profs = professor_nodes - connected_nodes
    connected_profs = professor_nodes & connected_nodes

    # Draw
    plt.figure(figsize=(5, 5))

    # Draw students with and without edges
    nx.draw_networkx_nodes(B, pos,
        nodelist=list(connected_students),
        node_shape='o',
        node_size=10,
        node_color='#7b68ee'  # Connected students
    )
    nx.draw_networkx_nodes(B, pos,
        nodelist=list(isolated_students),
        node_shape='o',
        node_size=10,
        node_color='#ff8c00'  # Isolated students
    )

    # Draw professors with and without edges
    nx.draw_networkx_nodes(B, pos,
        nodelist=list(connected_profs),
        node_shape='s',
        node_size=10,
        node_color='#20b2aa'  # Connected professors
    )
    nx.draw_networkx_nodes(B, pos,
        nodelist=list(isolated_profs),
        node_shape='s',
        node_size=10,
        node_color='#dc143c'  # Isolated professors
    )

    # Draw edges
    nx.draw_networkx_edges(B, pos,
        edge_color="#708090",
        width=0.5,
        alpha=0.1
    )

    plt.axis('off')
    plt.tight_layout()
    if title is not None:
        plt.title(title)
    if save_name is not None:
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.show()

    return B.number_of_edges() / (n_students * n_profs)