"""
NumPy Text Classifier from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - clean_text
def clean_text(text: str) -> str:
    # Lowercase text and replace non-alphabetic chars with spaces
    text = text.lower()
    return "".join(c if c.isalpha() else " " for c in text).strip()

# Step 2 - tokenize
def tokenize(text: str) -> list:
    # Split text on any whitespace into non-empty word tokens
    return text.split()

# Step 3 - tokenize_corpus
def tokenize_corpus(texts: list) -> list:
    # TODO: Apply clean_text and tokenize to every document so the full corpus becomes a list of token lists.
    return [tokenize(clean_text(t)) for t in texts]

# Step 4 - split_train_val_test_indices
import numpy as np

def split_train_val_test_indices(n_samples: int, val_fraction: float, test_fraction: float, seed: int = 0) -> tuple:
    # Set seed and create shuffled indices
    np.random.seed(seed)
    indices = np.random.permutation(n_samples)

    # Calculate split sizes
    n_val = int(n_samples * val_fraction)
    n_test = int(n_samples * test_fraction)
    n_train = n_samples - n_val - n_test 

    # Partition indices
    train = indices[:n_train]
    val = indices[n_train : n_train + n_val]
    test = indices[n_train + n_val :]
    
    return train, val, test

# Step 5 - count_word_frequencies
def count_word_frequencies(tokenized_docs: list) -> dict:
    # TODO: Return a dict mapping each unique token to its total count...
    word_counts = {}
    for doc in tokenized_docs:
        for token in doc:
            word_counts[token] = word_counts.get(token, 0) + 1
    return word_counts

# Step 6 - build_vocabulary
def build_vocabulary(word_counts: dict, max_size: int) -> dict:
    if not word_counts or max_size == 0:
        return {}

    # Sort by count descending (-count), then word string ascending (lexicographic tie-breaker)
    sorted_words = sorted(word_counts.items(), key=lambda pair: (-pair[1], pair[0]))
    
    # Keep top max_size words and assign zero-based indices
    return {word: i for i, (word, _) in enumerate(sorted_words[:max_size])}

# Step 7 - tokens_to_bow
def tokens_to_bow(tokens: list, vocab: dict) -> np.ndarray:
    # TODO: Convert one document's token list into a bag-of-words count vector...
    V = len(vocab)
    bow = np.zeros(V, dtype=float)
    for token in tokens:
        if token in vocab:
            idx = vocab[token]
            bow[idx] += 1
    return bow

# Step 8 - corpus_to_bow_matrix
def corpus_to_bow_matrix(tokenized_docs: list, vocab: dict) -> np.ndarray:
    # TODO: Stack per-document BoW vectors into a 2-D count matrix for a whole corpus.
    N, V = len(tokenized_docs), len(vocab)

    vector_list = []
    for i in range(N):
        bow = tokens_to_bow(tokenized_docs[i], vocab)
        vector_list.append(bow)
    if vector_list == []:
        return np.zeros((0, V))
    else:
        return np.stack(vector_list, axis=0)

# Step 9 - compute_document_frequencies
def compute_document_frequencies(bow_matrix: np.ndarray) -> np.ndarray:
    # TODO: Count docs where each term appears at least once (df, shape (V,))
    mask = bow_matrix > 0
    return np.sum(mask, axis = 0)

# Step 10 - compute_idf
def compute_idf(df: np.ndarray, n_docs: int) -> np.ndarray:
    # TODO: Compute smoothed IDF idf_j = log((n_docs + 1) / (df_j + 1)) + 1
    return np.log((n_docs+1)/(df+1))+1

# Step 11 - transform_tfidf
def transform_tfidf(bow_matrix: np.ndarray, idf: np.ndarray) -> np.ndarray:
    # TODO: Multiply BoW counts by the fitted IDF vector to produce TF-IDF features.
    return bow_matrix * idf

# Step 12 - fit_tfidf
def fit_tfidf(bow_train: np.ndarray) -> np.ndarray:
    # TODO: Fit IDF on the training BoW matrix by chaining DF and IDF.
    df = compute_document_frequencies(bow_train)
    n_docs = bow_train.shape[0]
    return compute_idf(df, n_docs)

# Step 13 - sigmoid
import numpy as np

def sigmoid(z: np.ndarray) -> np.ndarray:
    # Convert input to float array
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    
    # Split into positive and negative branches to avoid overflow (exp(large_positive))
    positive = z >= 0
    negative = ~positive
    
    # Positive branch: 1 / (1 + exp(-z))
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    
    # Negative branch: exp(z) / (1 + exp(z))
    exp_z = np.exp(z[negative])
    out[negative] = exp_z / (1.0 + exp_z)
    
    return out

# Step 14 - logistic_predict_proba
def logistic_predict_proba(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    # TODO: Return P(y=1|x) for each row via linear scores and sigmoid
    z = X @ w + b
    return sigmoid(z)

# Step 15 - binary_cross_entropy
def binary_cross_entropy(y_true: np.ndarray, y_prob: np.ndarray, w: np.ndarray, l2_lambda: float) -> float:
    # TODO: Compute mean binary cross-entropy plus L2 penalty on the weights.
    prob = np.clip(y_prob, 1e-15, 1.0 - 1e-15)
    t1 = y_true * np.log(prob)
    t2 = (1-y_true)*np.log(1-prob)
    BCE = -np.mean(t1+t2)
    return BCE + l2_lambda*np.sum(w**2)/2

# Step 16 - logistic_gradients
def logistic_gradients(X: np.ndarray, y_true: np.ndarray, y_prob: np.ndarray, w: np.ndarray, l2_lambda: float) -> tuple:
    """Compute gradients of BCE+L2 w.r.t. weights and bias for one full batch.

    Args:
        X: Feature matrix of shape (N, D).
        y_true: Binary labels of shape (N,).
        y_prob: Predicted probabilities of shape (N,).
        w: Weight vector of shape (D,).
        l2_lambda: L2 regularization strength.

    Returns:
        Tuple (dw, db) with dw shape (D,) and db a float.
    """
    # TODO: Compute gradients of BCE+L2 w.r.t. weights and bias for one full batch.
    N = len(y_true)
    r = y_prob - y_true 
    avg = (X.T @ r)/N
    dw = avg + l2_lambda*w 
    db = r.mean()
    return dw, db

# Step 17 - initialize_logistic_params
def initialize_logistic_params(n_features: int) -> tuple:
    # TODO: Return a zero weight vector of shape (n_features,) and bias 0.0
    return np.zeros(n_features), 0.0

# Step 18 - gradient_descent_step
def gradient_descent_step(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, lr: float, l2_lambda: float) -> tuple:
    # TODO: Run one full-batch gradient descent update; return (w_new, b_new, loss).
    y_prob = logistic_predict_proba(X, w, b)
    loss = binary_cross_entropy(y, y_prob, w, l2_lambda)
    dw, db = logistic_gradients(X, y, y_prob, w, l2_lambda)
    w_new = w - lr*dw 
    b_new = b -lr*db 
    return w_new, b_new, loss

# Step 19 - train_logistic_regression
def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float, l2_lambda: float, n_epochs: int) -> tuple:
    # TODO: Initialize params and run n_epochs of full-batch GD, recording loss...
    n_features = X.shape[1]
    w, b = initialize_logistic_params(n_features)
    losses = []
    for _ in range(n_epochs):
        w, b, loss = gradient_descent_step(X, y, w, b, lr, l2_lambda)
        losses.append(loss)
    return w, b, losses

# Step 20 - predict_labels
def predict_labels(proba: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Convert predicted probabilities into hard binary labels.

    Args:
        proba: 1-D array of probabilities in [0, 1], shape (N,).
        threshold: Decision threshold; proba >= threshold maps to 1.

    Returns:
        Integer array of shape (N,) with values in {0, 1}.
    """
    # TODO: Convert probabilities to hard binary labels via the threshold...
    mask = proba >= threshold 
    return mask.astype(int)

# Step 21 - confusion_counts
def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    # TODO: Return the four confusion-matrix counts (tp, fp, tn, fn) as Python ints
    
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    return int(TP), int(FP), int(TN), int(FN)

# Step 22 - metrics_from_counts
def metrics_from_counts(tp: int, fp: int, tn: int, fn: int) -> dict:
    # TODO: Derive precision, recall, F1, and accuracy from confusion counts...
    out = {}
    if tp + fp + tn + fn == 0:
        acc = 0.0
    else:
        acc = (tp+tn)/(tp + fp + tn + fn)
    out["accuracy"] = acc 
    if tp + fp == 0:
        prec = 0.0 
    else:
        prec = tp / (tp+fp)
    out["precision"] = prec 
    if tp + fn == 0:
        rec = 0.0
    else:
        rec = tp / (tp+fn)
    out["recall"] = rec 
    if prec + rec == 0.0:
        f1 = 0.0 
    else:
        f1 = 2*prec*rec/(prec+rec)
    out["f1"] = f1 
    return out

# Step 23 - tune_decision_threshold
def tune_decision_threshold(y_true: np.ndarray, proba: np.ndarray, thresholds: np.ndarray = None) -> tuple:
    # TODO: Find the decision threshold that maximizes F1 on validation data.
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)
    best_f1, best_threshold = -1.0, thresholds[0]
    for t in thresholds:
        y_pred = predict_labels(proba, threshold=float(t))
        tp, fp, tn, fn = confusion_counts(y_true, y_pred)
        metrics = metrics_from_counts(tp, fp, tn, fn)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = t 
    return float(best_threshold), float(best_f1)

# Step 24 - evaluate_predictions
def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    # TODO: Bundle confusion counts and classification metrics into one report dict
    tp, fp, tn, fn = confusion_counts(y_true, y_pred)
    metrics = metrics_from_counts(tp, fp, tn, fn)
    out = metrics.copy()
    out["tp"], out["fp"], out["tn"], out["fn"] = tp, fp, tn, fn 
    return out

# Step 25 - vectorize_texts (not yet solved)
# TODO: implement

# Step 26 - predict_text (not yet solved)
# TODO: implement

# Step 27 - collect_prediction_errors (not yet solved)
# TODO: implement

