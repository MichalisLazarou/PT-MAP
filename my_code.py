import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import faiss
import scipy as sp
from scipy.stats import t
import torch.nn.functional as F

def update_plabels(support, support_ys, query):
    max_iter = 20
    no_classes = support_ys.max() + 1
    k = 15
    alpha = 0.7
    X = np.concatenate((support, query), axis=0)
    labels = np.zeros(X.shape[0])
    labels[:support_ys.shape[0]]= support_ys
    labeled_idx = np.arange(support.shape[0])
    unlabeled_idx = np.arange(query.shape[0]) + support.shape[0]

    # kNN search for the graph
    d = X.shape[1]
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = int(torch.cuda.device_count()) - 1
    index = faiss.GpuIndexFlatIP(res, d, flat_config)  # build the index

    faiss.normalize_L2(X)
    index.add(X)
    N = X.shape[0]
    Nidx = index.ntotal

    D, I = index.search(X, k + 1)

    # Create the graph
    D = D[:, 1:] ** 3
    I = I[:, 1:]
    row_idx = np.arange(N)
    row_idx_rep = np.tile(row_idx, (k, 1)).T
    W = sp.sparse.csr_matrix((D.flatten('F'), (row_idx_rep.flatten('F'), I.flatten('F'))), shape=(N, N))
    W = W + W.T

        # Normalize the graph
    W = W - sp.sparse.diags(W.diagonal())
    S = W.sum(axis=1)
    S[S == 0] = 1
    D = np.array(1. / np.sqrt(S))
    D = sp.sparse.diags(D.reshape(-1))
    Wn = D * W * D

        # Initiliaze the y vector for each class (eq 5 from the paper, normalized with the class size) and apply label propagation
    Z = np.zeros((N, no_classes))
    A = sp.sparse.eye(Wn.shape[0]) - alpha * Wn
    for i in range(no_classes):
        cur_idx = labeled_idx[np.where(labels[labeled_idx] == i)]
        y = np.zeros((N,))
        y[cur_idx] = 1.0 #/ cur_idx.shape[0]
        f, _ = sp.sparse.linalg.cg(A, y, tol=1e-6, maxiter=max_iter)
        Z[:, i] = f

    # Handle numberical errors
    Z[Z < 0] = 0

        # Compute the weight for each instance based on the entropy (eq 11 from the paper)
    probs_l1 = F.normalize(torch.tensor(Z), 1).numpy()
    probs_l1[probs_l1 < 0] = 0
    p_labels = np.argmax(probs_l1, 1)
    p_probs = np.amax(probs_l1,1)

    p_labels[labeled_idx] = labels[labeled_idx]
    return p_labels[support.shape[0]:]

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), sp.stats.sem(a)
    h = se * t._ppf((1+confidence)/2., n-1)
    return m, h

def label_denoising(support, support_ys, query, query_ys_pred):
    input_size = support.shape[1]
    all_embeddings = np.concatenate((support, query), axis=0)
    X = torch.tensor(all_embeddings, dtype=torch.float32, requires_grad=True)
    all_ys = np.concatenate((support_ys, query_ys_pred), axis=0)
    Y = torch.tensor(all_ys, dtype=torch.long)
    output_size = support_ys.max() + 1
    start_lr = 0.1
    end_lr = 0.00
    cycle = 20 #number of epochs
    step_size_lr = (start_lr - end_lr) / cycle
    lambda1 = lambda x: start_lr - (x % cycle)*step_size_lr
    o2u = nn.Linear(input_size, output_size)
    #torch.save(o2u.state_dict(), 'loss_statistics/o2u_linear')
    o2u.load_state_dict(torch.load('/home/michalislazarou/PhD/rfs_baseline/loss_statistics/o2u_linear'))
    optimizer = optim.SGD(o2u.parameters(), 1, momentum=0.9, weight_decay=5e-4)
    scheduler_lr = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda1)
    criterion = nn.CrossEntropyLoss(reduction = 'none')
    loss_statistics = torch.zeros(all_ys.shape, requires_grad=True)
    lr_progression =[]
    total_iterations = 2000
    for epoch in range(total_iterations):
        output = o2u(X)
        optimizer.zero_grad()
        loss_each = criterion(output, Y)
        loss_all = torch.mean(loss_each)
        loss_all.backward()
        loss_statistics = loss_statistics + loss_each/(total_iterations)
        optimizer.step()
        scheduler_lr.step()
        lr_progression.append(optimizer.param_groups[0]['lr'])
    return loss_statistics, lr_progression