

import numpy as np
import copy
import random
import pandas as pd

#---------------------------------Community Detection Using Louvain Method Modified For Correlation Matrices---------------------------------

def create_correlation_matrix(data):
    """
    Create a correlation matrix from a file path or a DataFrame.
    Args:
        data (str or pd.DataFrame): Path to the CSV file or a DataFrame.
    Returns:
        tuple: Correlation matrix (as NumPy array), T, N, and company names.
    """
    if isinstance(data, str):  # If data is a file path
        df_standardized_returns = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):  # If data is already a DataFrame
        df_standardized_returns = data
    else:
        raise ValueError("Input must be a file path or a DataFrame.")
    
    # Set the 'timestamp' column as the index if it's not already
    if 'timestamp' in df_standardized_returns.columns:
        df_standardized_returns.set_index('timestamp', inplace=True)
    
    # Calculate lambda boundaries using RMT
    T = len(df_standardized_returns)
    N = len(df_standardized_returns.columns)

    # Extract the company names
    company_names = df_standardized_returns.columns.to_list()
    
    # Calculate the correlation matrix on the returns data
    correlation_matrix = df_standardized_returns.corr().to_numpy()  # Convert to NumPy array

    return correlation_matrix, T, N, company_names

def calculate_C_g(correlation_matrix, T, N):
    """
    Calculate the denoised correlation matrix using eigenvalue filtering.
    Ensures real values are used throughout the calculation.
    
    Args:
        correlation_matrix (np.ndarray): Input correlation matrix.
        T (int): Number of time periods.
        N (int): Number of variables.
    
    Returns:
        np.ndarray: Filtered (denoised) correlation matrix.
    """
    # Calculate the eigenvalues and eigenvectors of the correlation matrix
    eigenvalues, eigenvectors = np.linalg.eig(correlation_matrix)
    
    # Ensure real values (handle small imaginary parts caused by numerical errors)
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)
    
    lambda_plus = (1 + np.sqrt(N / T))**2
    lambda_min = (1 - np.sqrt(N / T))**2  # Not used in this code but calculated as per RMT
    
    print("Lambda plus:", lambda_plus)
    print(f"Eigenvalues: {eigenvalues}")    

    # Obtain eigenvalues and eigenvectors above lambda_plus
    denoised_eigenvalues = []
    denoised_eigenvectors = []

    
    for index, eigenvalue in enumerate(eigenvalues):
        if eigenvalue > lambda_plus:
            denoised_eigenvalues.append(eigenvalue)
            denoised_eigenvectors.append(eigenvectors[:, index])  # Corresponding eigenvector

    print("denoised_eigenvalues", denoised_eigenvalues)
    
    # Remove the largest eigenvalue (global mode) from denoised values
    if denoised_eigenvalues:
        max_value = max(denoised_eigenvalues)
        max_index = denoised_eigenvalues.index(max_value)
        denoised_eigenvalues.pop(max_index)
        denoised_eigenvectors.pop(max_index)
    
    # Reconstruct the filtered correlation matrix C^(g)
    C_g = np.zeros_like(correlation_matrix)
    for i, eigenvalue in enumerate(denoised_eigenvalues):
        eigenvector = np.array(denoised_eigenvectors[i]).reshape(-1, 1)  # Column vector
        C_g += eigenvalue * (eigenvector @ eigenvector.T)  # Outer product
    
    # Return the filtered correlation matrix
    return C_g


def calculate_modularity_with_potential_move(node,current_index,neighbour_community_index,communities, modularity_matrix, original_corr_matrix):
    
    "Calculate the modularity of a potential move of a node from its current community to a neighbour community"

    # Get the current and neighbor communities
    current_community = communities[current_index]
    neighbour_community = communities[neighbour_community_index]

    #CALCULATE MODULARITY GAIN FROM ADDING NODE TO NEW COMMUNITY
    #---------------------------------------------------

    # Calculate old modularity of the neighbor community
    # old_modularity_new_community = sum(
    #     modularity_matrix[i][j] 
    #     for i in neighbour_community
    #     for j in neighbour_community
    #     if i <= j
    # )

    # Calculate new modularity after adding the node to the neighbor community
    temp_neighbour_community = neighbour_community + [node] #We create a temp variable to avoid modifying the original community
    new_modularity_new_community = sum(  #The new modularity can be computed by adding the modularity without the node to the additional gain after adding the node
        modularity_matrix[node][j] #We only check pairwise correlations with the node we just added to the community
        #for i in temp_neighbour_community
        for j in temp_neighbour_community
        #if i <= j
    )

    #change_from_addition = new_modularity_new_community - old_modularity_new_community
    change_from_addition  = new_modularity_new_community

    #CALCULATE MODULARITY LOSS FROM REMOVING NODE FROM CURRENT COMMUNITY
    #---------------------------------------------------

    old_modularity_current_community = sum(
        modularity_matrix[i][j]
        for i in current_community
        for j in current_community
        if i <= j
    )

    temp_current_community = copy.deepcopy(current_community) #We create a temp variable to avoid modifying the original community
    temp_current_community.remove([node]) #Remove node from old community

    new_modularity_current_communtity = sum(
        modularity_matrix[i][j]
        for i in temp_current_community
        for j in temp_current_community
        if i <= j
    )


    change_from_removal = old_modularity_current_community - new_modularity_current_communtity



    # Normalize the modularity change
    c_norm = np.sum(np.triu(original_corr_matrix))


    modularity_change = (change_from_addition - change_from_removal) / c_norm

    return modularity_change


def phase_1(communities, modularity_matrix, original_corr_matrix):

    "Randomly select a node, evaluate modularity changes, and move it to maximize modularity"
    
    nodes = [node for community in communities for node in community]  # Flatten all nodes into a single list
    community_map = {node: index for index, community in enumerate(communities) for node in community}  # Map nodes to their communities
    
    moved = True
    iteration = 0

    while moved == True:  # Continue until no moves improve modularities

        #print(f"ITERATION {iteration}")
        #print("-----------------")
        
        iteration += 1

        moved = False
        random.shuffle(nodes)  # Randomly shuffle the nodes
        
        for node in nodes:

            #print("COMMUNITY MAP BELOW:")
            #print(community_map)

            current_index = community_map[node]  # Find the node's current community index
            #print("Current index",current_index)
            current_community = communities[current_index]
            
            #print("Node:", node)
            #print("Current Node Communty:", current_community)
            
            max_modularity = 0
            best_community = None
            
            for j, neighbor_community in enumerate(communities):

                if not neighbor_community or neighbor_community == current_community:
                    continue  # Skip empty or identical communities
                
                modularity_change = calculate_modularity_with_potential_move(node, current_index, j, communities, modularity_matrix, original_corr_matrix)
                
                if modularity_change > max_modularity:
                    max_modularity = modularity_change
                    best_community = neighbor_community
            
            # If a modularity improvement is found, move the node
            if max_modularity > 0 and best_community is not None:
                current_community.remove(node)
                best_community.append(node)
                community_map[node] = communities.index(best_community)  # Update the community map
                moved = True  # Indicate that a move occurred
                #print("MOVED NODE", node, "TO COMMUNITY", communities.index(best_community))

                #print("UPDATED COMMUNITY MAP:")
                #print(community_map)

        # Remove empty communities
        communities = [community for community in communities if community]

        #Update community map with new communities
        community_map = {node: index for index, community in enumerate(communities) for node in community}  # Map nodes to their communities
        
        #print(f"End of iteration {iteration}, Communities: {communities},")
    
    return communities

def phase_2(communities,modularity_matrix):

    "Node aggregation phase"

    # Aggregate nodes into hypernodes

    #df_standardized_returns = pd.read_csv("returns_standardized.csv")

    renormalized_modularity_matrix = np.zeros((len(communities),len(communities))) #If there are 5 communities, this will be a 5x5 matrix

    for community_index, community in enumerate(communities):
        
        hypernode_correlations = [] #Stores the correlation between the current community and all other communities (Including self loops)

        for  neighbour_community in communities:
            
            # if community == neighbour_community:
            #     continue
            
            # Calculate the covariance between the current community and the neighbour community
            cov_comm1_comm2 = 0

            for i in community:
                for j in neighbour_community:
                    cov_comm1_comm2 += modularity_matrix[i][j]
            
            hypernode_correlations.append(cov_comm1_comm2)

        renormalized_modularity_matrix[community_index] = hypernode_correlations
        

    return renormalized_modularity_matrix

def flatten_final_communities(final_communities):

    flattened_communities = []
    for item in final_communities:
        if isinstance(item, list) and all(isinstance(subitem, list) for subitem in item):
            # Flatten one level of nesting
            flattened_item = [subitem for nested in item for subitem in nested]
            flattened_communities.append(flattened_item)
        else:
            flattened_communities.append(item)  # Keep as is
            
    return flattened_communities

def map_hypernodes_to_nodes(hypernode_communities, node_communities):

    final_communities = []

    for community in hypernode_communities:

        final_community = []

        for hyper_node in community:

            node_community = node_communities[hyper_node]
            final_community.append(node_community)
            

        final_communities.append(final_community)

    #Flatten final communities so they are in the same shape as the original communities
    final_communities = flatten_final_communities(final_communities)
    
    return final_communities


def modified_louvain(modularity_matrix,original_corr_matrix):

    #Get node indices from matrix
    node_indices = np.arange(modularity_matrix.shape[0])

    #Create initial communities
    initial_pahse1_communities = [[node] for node in node_indices]

    #Phase 1 for nodes
    #--------------------------------

    #Calculate initial communities
    phase1_communities = phase_1(initial_pahse1_communities, modularity_matrix, original_corr_matrix)

    #Phase 2 (aggregation) for hypernodes
    #--------------------------------

    renormalized_modularity_matrix = phase_2(phase1_communities,modularity_matrix)

    hyper_node_indices = np.arange(renormalized_modularity_matrix.shape[0])

    initial_hypernode_communities = [[node] for node in hyper_node_indices]

    #print("Initial hypernode communities", initial_hypernode_communities)

    #Phase 1 for hypernodes
    #--------------------------------

    phase1_hypernode_communities = phase_1(initial_hypernode_communities, renormalized_modularity_matrix,original_corr_matrix)

    #Map back hypernode communities to node communities
    #--------------------------------

    print("Detected", len(phase1_communities), "Initially")

    final_communities = map_hypernodes_to_nodes(phase1_hypernode_communities, phase1_communities)

    print("Final number of communities:", len(final_communities))
    
    return final_communities


def calculate_global_modularity(communities, modularity_matrix, original_corr_matrix):

    #Calculate the global modularity of the communities

    modularity = 0
    #c_norm = np.sum(modularity_matrix)
    c_norm = np.sum(np.triu(original_corr_matrix))

                
    modularity = 0.0

    for community in communities:
        for i in community:
            for j in community:  
                if i<=j:  #Ensure that we only add each pair of nodes once! 
                    modularity += modularity_matrix[i][j]

    modularity = modularity/c_norm

    return modularity

def map_communities_to_company_names(communities, company_names):

    company_communities = []

    for partition in communities:
        company_list = []
        for i in partition:
            company_list.append(company_names[i])
        company_communities.append(company_list)

    return company_communities    

def map_communities_to_company_names(communities, company_names):

    company_communities = []

    for partition in communities:
        company_list = []
        for i in partition:
            company_list.append(company_names[i])
        company_communities.append(company_list)
    
    return company_communities 


def spectral_method(C_g):
    # Perform eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eig(C_g)
    max_eigenvalue_index = np.argmax(eigenvalues)
    leading_eigenvector = eigenvectors[:, max_eigenvalue_index]
    
    community_1 = []
    community_2 = []
    
    # Creating the communities based on the sign of the eigenvector
    for i in range(len(leading_eigenvector)):
        if leading_eigenvector[i] > 0:
            community_1.append(i)  
        else:
            community_2.append(i) 
    
    return [community_1, community_2]


def calculate_modularity(C_g, partitions, corr_matrix):
    # C_norm is the total sum of C_g (Eq.38)
    c_norm = np.sum(np.triu(corr_matrix))

    modularity = 0.0
    
    # Calculate modularity based on the partition
    #Since the nodes have been assigned to communities, and we are summing over these communitiies seperately, there is no need for the (sisj +1)/2 term, shown in the paper
    for community in partitions:
        for i in community:
            for j in community:
                if i <= j:
                    modularity += C_g[i, j]
    
    # Normalize modularity by C_norm
    modularity /= c_norm
    return modularity



def recursive_spectral_method(C_g, corr_matrix, company_names, min_size=2, modularity_threshold=0.00001):
    result_communities = []
    modularities = []

    # Recursive function to split communities
    def split_community(community_nodes):

        # If community is too small, add it directly to result_communities
        if len(community_nodes) <= min_size:
            result_communities.append(community_nodes)
            return

        # Extract the submatrix for the current community
        submatrix = C_g[np.ix_(community_nodes, community_nodes)]

        # Apply spectral method to split into two communities
        communities = spectral_method(submatrix)

        # Map the sub-community indices back to the original indices
        community_1 = [community_nodes[i] for i in communities[0]]
        community_2 = [community_nodes[i] for i in communities[1]]
        # Calculate modularity before and after the split
        initial_modularity = calculate_modularity(C_g, [community_nodes], corr_matrix)
        new_modularity = calculate_modularity(C_g, [community_1, community_2], corr_matrix)
        # Check if the split improves modularity significantly
        if (new_modularity - initial_modularity) > modularity_threshold:
            # Recursively split each resulting community
            split_community(community_1)
            split_community(community_2)
            modularities.append(new_modularity)
        else:
            # If modularity gain is too low, add the original community without splitting
            result_communities.append(community_nodes)
    
    # Start recursive splitting from the entire set of nodes
    all_nodes = list(range(len(C_g)))


    split_community(all_nodes)

    company_communities = []

    for partition in result_communities:
        company_list = []
        for i in partition:
            company_list.append(company_names[i])
        company_communities.append(company_list)

 
    return result_communities, company_communities, modularities

if __name__ == "__main__":

    correlation_matrix,T,N,company_names = create_correlation_matrix('binance_data/top_100_tickers/2024/1m/log_returns_standardized.csv') 
    print(correlation_matrix) 
    C_g = calculate_C_g(correlation_matrix,T,N) #Modularity matrix
    # print(correlation_matrix.shape)
    # correlation_matrix_np = np.array(correlation_matrix)
    # louvain_communities = modified_louvain(C_g,correlation_matrix_np)

    # print("Global modularity of modified Louvain method: ", calculate_global_modularity(louvain_communities, C_g, correlation_matrix_np))


    