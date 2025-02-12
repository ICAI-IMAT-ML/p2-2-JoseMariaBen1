import numpy as np
def get_k_nearest_neighbors(distances: np.ndarray) -> np.ndarray:
    """Get the k nearest neighbors indices given the distances matrix from a point.

    Args:
        distances (np.ndarray): distances matrix from a point whose neighbors want to be identified.

    Returns:
        np.ndarray: row indices from the k nearest neighbors.

    Hint:
        You might want to check the np.argsort function.
    """
    indices_menor_distancia = []
    while len(indices_menor_distancia) !=  len(distances):
        menor_indice = 0
        menor_distancia = np.infty
        for i in range(0,len(distances)):
            if distances[i] < menor_distancia and i not in indices_menor_distancia:
                menor_indice = i
                menor_distancia = distances[i]
                print(i)
                print(distances[i])
        
        indices_menor_distancia.append(menor_indice)
    return indices_menor_distancia

y_true = np.array([1, 0, 1, 0, 1, 0])
y_pred = np.array([1, 0, 1, 0, 0, 1])
positive_label = 1
y_true_mapped = np.array([1 if label == positive_label else 0 for label in y_true])
y_pred_mapped = np.array([1 if label == positive_label else 0 for label in y_pred])
TP = np.sum((y_true_mapped == 1) & (y_pred_mapped == 1))
TN = np.sum((y_true_mapped == 0) & (y_pred_mapped == 0))
FP = np.sum((y_true_mapped == 0) & (y_pred_mapped == 1))
FN = np.sum((y_true_mapped == 1) & (y_pred_mapped == 0))

# Confusion Matrix

# Accuracy
accuracy = (TP+TN)/(TP+TN+FP+FN)

# Precision
precision=TP/(TP+FP)

# Recall (Sensitivity)
recall = TP/(TP+FN)

# Specificity
specificity = TN/(TN+FP)

# F1 Score
f1 = 2 *(precision*recall)/(precision + recall)

y_true = np.array([1, 0, 1, 0, 1, 0, 1, 1, 0, 0])
y_probs = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.2, 0.1])
positive_label = 1
n_bins = 5

y_true_mapped = np.array([1 if label == positive_label else 0 for label in y_true])
y_probs = np.array(y_probs)

cte = 1/n_bins
bins = np.array([[cte*(i-1), cte*i] for i in range(1,n_bins+1)])
bin_centers = np.array([(cte*(i-1)+cte*i)/2 for i in range(1,n_bins+1)])
true_proportions = []

for bin in bins:
    print(bin)
    digitos = []
    for i in range(len(y_probs)):
        if y_probs[i] >= bin[0] and y_probs[i] < bin[1]:
            print(y_true_mapped[i])
            digitos.append(y_true_mapped[i])
    digitos = np.array(digitos)
    if np.sum(digitos==1) == 0:
        prob = 0
    else:
        prob = np.sum(digitos==1)/np.sum(digitos)
    true_proportions.append(prob)


#predictivos correspondan con el valor del intervalo 
#dum encoding 


print({"bin_centers": bin_centers, "true_proportions": true_proportions})

def plot_calibration_curve(y_true, y_probs, positive_label, n_bins=10):
    """
    Plot a calibration curve to evaluate the accuracy of predicted probabilities.

    This function creates a plot that compares the mean predicted probabilities
    in each bin with the fraction of positives (true outcomes) in that bin.
    This helps assess how well the probabilities are calibrated.

    Args:
        y_true (array-like): True labels of the data. Can be binary or categorical.
        y_probs (array-like): Predicted probabilities for the positive class (positive_label).
                            Expected values are in the range [0, 1].
        positive_label (int or str): The label that is considered the positive class.
                                    This is used to map categorical labels to binary outcomes.
        n_bins (int, optional): Number of bins to use for grouping predicted probabilities.
                                Defaults to 10. Bins are equally spaced in the range [0, 1].

    Returns:
        dict: A dictionary with the following keys:
            - "bin_centers": Array of the center values of each bin.
            - "true_proportions": Array of the fraction of positives in each bin.
    """

    # Map labels to binary values (0 or 1)
    y_true_mapped = np.array([1 if label == positive_label else 0 for label in y_true])
    y_probs = np.array(y_probs)

    cte = 1/n_bins
    bins = np.array([[cte*(i-1), cte*i] for i in range(1,n_bins+1)])
    bin_centers = np.array([(cte*(i-1)+cte*i)/2 for i in range(1,n_bins+1)])
    true_proportions = []

    for bin in bins:
        digitos = []
        for i in range(len(y_probs)):
            if y_probs[i] >= bin[0] and y_probs[i] <= bin[1]:
                digitos.append(y_true_mapped[i])
        digitos = np.array(digitos)
        if np.sum(digitos==1) == 0.0:
            prob = 0.0
        else:
            prob = np.mean(digitos)
        true_proportions.append(prob)
        

    #predictivos correspondan con el valor del intervalo 
    #dum encoding 

    
    return {"bin_centers": bin_centers, "true_proportions": true_proportions}

y_probs_uniform = np.array([0.5] * len(y_true))
result_uniform = plot_calibration_curve(
    y_true, y_probs_uniform, positive_label, n_bins=1
)

print(result_uniform["true_proportions"][0])

print(np.mean(y_true == positive_label))