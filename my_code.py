import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
import faiss
import scipy as sp
import umap
from sklearn.decomposition import PCA, KernelPCA, FastICA
from sklearn.manifold import LocallyLinearEmbedding, SpectralEmbedding, Isomap
from scipy.stats import t
import torch.nn.functional as F

def update_plabels(support, support_ys, query, i = 0):
    max_iter = 20
    no_classes = support_ys.max() + 1

    X = np.concatenate((support, query), axis=0)
    k = 30
    alpha = 0.2
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
    #entropy = sp.stats.entropy(probs_l1.T)
    #weights = 1 - entropy / np.log(no_classes)
    #weights = weights / np.max(weights)
    weights = probs_l1

    p_labels[labeled_idx] = labels[labeled_idx]
    return p_labels[support.shape[0]:], weights[support.shape[0]:]

def weight_imprinting(X, Y, model):
    no_classes = Y.max()+1
    imprinted = torch.zeros(no_classes, X.shape[1])
    for i in range(no_classes):
        idx = np.where(Y == i)
       # print(idx)
        tmp = torch.mean(X[idx], dim=0)
        #print(tmp.norm(p=2), torch.sum(tmp))
        tmp = tmp/tmp.norm(p=2)
        #tmp = tmp / torch.sum(tmp)
        imprinted[i, :] = tmp
   # print(model.weight.shape, imprinted.shape)
    model.weight.data = imprinted
    return model

def update_plabels_Y(support, query, Y):
    max_iter = 20
    no_classes = Y.shape[1]
    k = 10
    alpha = 0.8
    X = np.concatenate((support, query), axis=0)

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
        f, _ = sp.sparse.linalg.cg(A, Y[:, i], tol=1e-6, maxiter=max_iter)
        Z[:, i] = f

    # Handle numberical errors
    Z[Z < 0] = 0

        # Compute the weight for each instance based on the entropy (eq 11 from the paper)
    probs_l1 = F.normalize(torch.tensor(Z), 1).numpy()
    probs_l1[probs_l1 < 0] = 0
    p_labels = np.argmax(probs_l1, 1)
    p_probs = np.amax(probs_l1,1)
    #entropy = sp.stats.entropy(probs_l1.T)
    #weights = 1 - entropy / np.log(no_classes)
    #weights = weights / np.max(weights)
    weights = probs_l1

    return p_labels[support.shape[0]:], weights[support.shape[0]:]

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), sp.stats.sem(a)
    h = se * t._ppf((1+confidence)/2., n-1)
    return m, h

def label_denoising(support, support_ys, query, query_ys_pred):
   # pca = PCA(n_components=10)
    all_embeddings = np.concatenate((support, query), axis=0)
    #all_embeddings = pca.fit_transform(all_embeddings)
    input_size = all_embeddings.shape[1]
    X = torch.tensor(all_embeddings, dtype=torch.float32, requires_grad=True)
    all_ys = np.concatenate((support_ys, query_ys_pred), axis=0)
    Y = torch.tensor(all_ys, dtype=torch.long)
    output_size = support_ys.max() + 1
    start_lr = 0.1
    end_lr = 0.00
    cycle = 50 #number of epochs
    step_size_lr = (start_lr - end_lr) / cycle
    lambda1 = lambda x: start_lr - (x % cycle)*step_size_lr
    o2u = nn.Linear(input_size, output_size)
    o2u = weight_imprinting(torch.Tensor(all_embeddings[:support_ys.shape[0]]), support_ys, o2u)
    #torch.save(o2u.state_dict(), 'loss_statistics/o2u_linear')
    #o2u.load_state_dict(torch.load('/home/michalislazarou/PhD/rfs_baseline/loss_statistics/o2u_linear'))
    #optimizer = optim.Adam(o2u.parameters(), 0.01)#, momentum=0.9, weight_decay=5e-4)
    optimizer = optim.SGD(o2u.parameters(), 1, momentum=0.9, weight_decay=5e-4)
    scheduler_lr = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda1)
    criterion = nn.CrossEntropyLoss(reduction = 'none')
    loss_statistics = torch.zeros(all_ys.shape, requires_grad=True)
    lr_progression =[]
    total_iterations = 200
    for epoch in range(total_iterations):
        output = o2u(X)
        optimizer.zero_grad()
        loss_each = criterion(output, Y)
        loss_all = torch.mean(loss_each)
        loss_all.backward()
        #if epoch>200:
        loss_statistics = loss_statistics + loss_each/(total_iterations)
        optimizer.step()
        scheduler_lr.step()
        lr_progression.append(optimizer.param_groups[0]['lr'])
    return loss_statistics, lr_progression

def update_embeddings(support, support_ys, query):
    max_iter = 20
    no_classes = support_ys.max() + 1
    k = 40
    alpha = 0.2
    X = np.concatenate((support, query), axis=0)
    labels = np.zeros(X.shape[0])
    labels[:support_ys.shape[0]]= support_ys
    #print(labels.shape)
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
    Z = np.zeros((N, query.shape[1]))
    A = sp.sparse.eye(Wn.shape[0]) - alpha * Wn
    for i in range(query.shape[1]):
        cur_idx = labeled_idx[np.where(labels[labeled_idx] == i)]
        y = np.zeros((N,))
        #print(y.shape, X[:,i].shape)
        y = X[:,i] #/ cur_idx.shape[0]
        f, _ = sp.sparse.linalg.cg(A, y, tol=1e-6, maxiter=max_iter)
        Z[:, i] = f
    #print(X.dtype, Z.dtype)
    Z = np.float32(Z)
    #Z = F.normalize(torch.tensor(Z), 1).numpy()
    #print(X.dtype, Z.dtype)
    # Handle numberical errors
    #Z[Z < 0] = 0

    # Compute the weight for each instance based on the entropy (eq 11 from the paper)
    # probs_l1 = F.normalize(torch.tensor(Z), 1).numpy()
    # probs_l1[probs_l1 < 0] = 0
    # p_labels = np.argmax(probs_l1, 1)
    # p_probs = np.amax(probs_l1,1)
    # weights = p_probs
    #
    # p_labels[labeled_idx] = labels[labeled_idx]

    return Z[:support_ys.shape[0]], Z[support_ys.shape[0]:]

def rank_probs_per_class(no_class, probs, no_keep =1):
    list_indices = []
    for i in range(no_class):
        index = np.argmax(probs[:,i])
        list_indices.append(index)
    #idxs = np.concatenate(list_indices, axis=0)
    #print(idxs)
    return list_indices

def rank_per_class(no_cls, rank, ys_pred, no_keep):
    list_indices = []
    list_ys = []
    for i in range(no_cls):
        cur_idx = np.where(ys_pred == i)
        #print(cur_idx[0])
        y = np.ones((no_cls,))*i
        class_rank = rank[cur_idx]
        #print(class_rank.shape)
        class_rank_sorted = sp.stats.rankdata(class_rank, method='ordinal')
        class_rank_sorted[class_rank_sorted > no_keep] = 0
        indices = np.nonzero(class_rank_sorted)
       # print(y.shape, cur_idx[0][indices[0]].shape)
        list_indices.append(cur_idx[0][indices[0]])
        list_ys.append(y)
    idxs = np.concatenate(list_indices, axis=0)
    ys = np.concatenate(list_ys, axis = 0)
    #print(idxs.shape, ys.shape)
    #print(idxs, ys)
    return idxs, ys

def update_class_embeddings(X):
    max_iter = 20
    no_features = X.shape[1]
    k=5#X.shape[0]-1
    if X.shape[0]<= k:
        k = X.shape[0] - 1
    alpha = 0.85

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

    Z = np.zeros((N, no_features))
    A = sp.sparse.eye(Wn.shape[0]) - alpha * Wn
    for i in range(no_features):
        y = X[:,i] #/ cur_idx.shape[0]
        f, _ = sp.sparse.linalg.cg(A, y, tol=1e-6, maxiter=max_iter)
        Z[:, i] = f
    Z = np.float32(Z)
    return Z


def selectively_update_embeddings(support_features, support_ys, query_features, query_ys):
    #--------------------when input was support, query-----------------------------------------------------------------
    no_classes = support_ys.max() + 1
    X = np.concatenate((support_features, query_features), axis=0)
    labels = np.zeros(X.shape[0])
    labels[:support_ys.shape[0]]= support_ys
    labels[support_ys.shape[0]:]= query_ys
    #-------------------------------------------------------------------------------------------------------------------
    #no_classes = labels.max() + 1
    labeled_idx = np.arange(X.shape[0])
    Z = np.zeros((X.shape))

    for j in range(no_classes):
        cur_idx = labeled_idx[np.where(labels[labeled_idx] == j)]
        y = np.zeros((cur_idx.shape[0],))
        y = X[cur_idx, :] #/ cur_idx.shape[0]
        #print(y.shape)
        Z[cur_idx, :] = update_class_embeddings(y)
        #Z[cur_idx, :] = cluster_embeddings(y)
    Z = np.float32(Z)
    return Z[:support_ys.shape[0]], Z[support_ys.shape[0]:]

def iter_balanced(support_features, support_ys, query_features, query_ys, labelled_samples):
    query_ys_updated = query_ys
    new_support_features, new_query_features = support_features, query_features
    iterations = int(query_ys.shape[0] / 15)
    for j in range(iterations):
        query_ys_pred, probs = update_plabels(support_features, support_ys, query_features, i=j)
        P, query_ys_pred, indices = compute_optimal_transport(torch.Tensor(probs), T=3)
        #I, query_ys_pred, indices = greedy_selection(P, int(P.shape[0]/P.shape[1]))
        loss_statistics, _ = label_denoising(support_features, support_ys, query_features, query_ys_pred)

        #query_ys_pred, _ = update_plabels(new_support_features, support_ys, new_query_features)
        #loss_statistics, _ = label_denoising(new_support_features, support_ys, new_query_features, query_ys_pred)

        un_loss_statistics = loss_statistics[support_ys.shape[0]:]
        rank = sp.stats.rankdata(un_loss_statistics.detach().numpy(), method='ordinal')
        indices, ys = rank_per_class(support_ys.max() + 1, rank, query_ys_pred, 1)

        if len(indices)<5:
            #print(ys)
            break;
        pseudo_mask = np.in1d(np.arange(query_features.shape[0]), indices)
        pseudo_features, query_features = query_features[pseudo_mask], query_features[~pseudo_mask]
        pseudo_ys, query_ys_pred = query_ys_pred[pseudo_mask], query_ys_pred[~pseudo_mask]
        query_ys_concat, query_ys_updated = query_ys_updated[pseudo_mask], query_ys_updated[~pseudo_mask]
        support_features = np.concatenate((support_features, pseudo_features), axis=0)
        support_ys = np.concatenate((support_ys, pseudo_ys), axis=0)
        query_ys = np.concatenate((query_ys, query_ys_concat), axis=0)
        #new_support_features, new_query_features = selectively_update_embeddings(support_features, support_ys, query_features, query_ys_pred)
        # print(support_features.shape, support_ys.shape, query_features.shape, query_ys_pred.shape)
    support_ys = np.concatenate((support_ys, query_ys_pred), axis=0)
    query_ys = np.concatenate((query_ys, query_ys_updated), axis=0)
    query_ys_pred = support_ys[labelled_samples:]
    query_ys = query_ys[query_ys_pred.shape[0]:]
   # print(query_ys.shape, query_ys_pred.shape)
    return query_ys, query_ys_pred

def iter_balanced_augmented(support_features, support_ys, query_features, query_ys, labelled_samples):
    query_ys_updated = query_ys
    no_augmented_support = support_features.shape[0]

    for j in range(int(query_ys.shape[0] / 5)):
        Y = np.zeros((query_ys_updated.shape[0], support_ys.shape[1]))
        Y = np.concatenate((support_ys, Y), axis=0)
        query_ys_pred, _ = update_plabels_Y(support_features, query_features, Y)
        loss_statistics, _ = label_denoising(support_features[:labelled_samples], np.argmax(support_ys[:labelled_samples], 1), query_features, query_ys_pred)
        un_loss_statistics = loss_statistics[support_ys[:labelled_samples].shape[0]:]
        rank = sp.stats.rankdata(un_loss_statistics.detach().numpy(), method='ordinal')
        indices, ys = rank_per_class(support_ys.shape[1], rank, query_ys_pred, 1)
        if len(indices) < 5:
            break;
        pseudo_mask = np.in1d(np.arange(query_features.shape[0]), indices)
        pseudo_features, query_features = query_features[pseudo_mask], query_features[~pseudo_mask]
        pseudo_ys, query_ys_pred = query_ys_pred[pseudo_mask], query_ys_pred[~pseudo_mask]
        query_ys_concat, query_ys_updated = query_ys_updated[pseudo_mask], query_ys_updated[~pseudo_mask]
        support_features = np.concatenate((support_features, pseudo_features), axis=0)
        pseudo_ys = one_hot(pseudo_ys)
        support_ys = np.concatenate((support_ys, pseudo_ys), axis=0)
        query_ys = np.concatenate((query_ys, query_ys_concat), axis=0)
    support_ys = np.argmax(np.concatenate((support_ys, one_hot(query_ys_pred, 5)), axis=0), 1)
    query_ys = np.concatenate((query_ys, query_ys_updated), axis=0)
    query_ys_pred = support_ys[no_augmented_support:]
    query_ys = query_ys[query_ys_pred.shape[0]:]
    #print(query_ys.shape, query_ys_pred.shape)
    return query_ys, query_ys_pred


def my_method(X, Y, labelled_samples):
    acc = []
    X = X.detach().cpu().numpy()
    Y = Y.detach().cpu().numpy()
    #pca = umap.UMAP(n_neighbors=10, metric='euclidean', verbose=False)
    for i in range(X.shape[0]):
        #print(X[i].shape)
        #X_pca = pca.fit_transform(X[i])
        support_features, query_features =  X[i,:labelled_samples], X[i,labelled_samples:] # X_pca[:labelled_samples], X_pca[labelled_samples:] #
        support_ys, query_ys = Y[i,:labelled_samples], Y[i,labelled_samples:]

        #query_ys_pred, probs = update_plabels(support_features, support_ys, query_features)
        #P, query_ys_pred, _ = compute_optimal_transport(torch.Tensor(probs), T=3)
        #I, query_ys_pred, _ = greedy_selection(P, int(probs.shape[0] / probs.shape[1]))
        #support_features, query_features = update_embeddings(support_features, support_ys, query_features)

        query_ys, query_ys_pred = iter_balanced(support_features, support_ys, query_features, query_ys, labelled_samples)

        #-----------------------------------------------using augmentations---------------------------------------------
        #---------------------------------------------------------------------------------------------------------------
        # support_features, support_ys = augment_support(support_features, support_ys, 5)
        #query_ys, query_ys_pred = iter_balanced_augmented(support_features, support_ys, query_features, query_ys, labelled_samples)
        # Y_matrix = np.zeros((query_ys.shape[0], support_ys.shape[1]))
        # Y_matrix = np.concatenate((support_ys, Y_matrix), axis=0)
        # query_ys_pred, _ = update_plabels_Y(support_features, query_features, Y_matrix)
        #-----------------------------------------------using augmentations---------------------------------------------
        #---------------------------------------------------------------------------------------------------------------
        acc.append(metrics.accuracy_score(query_ys, query_ys_pred))
        if i%50==0:
            a, h = mean_confidence_interval(acc)
            print("Iteration: ", i, a*100)
    return mean_confidence_interval(acc)

def one_hot(Y, dimension=5):
    b = np.zeros((Y.size, dimension))
    b[np.arange(Y.size), Y] = 1
    return b

def augment_support(X, Y, aug_no, alpha=0.75):
    new_X = []
    new_Y = []
    Y = one_hot(Y)
    #print(Y)
    new_X.append(X)
    new_Y.append(Y)
    for i in range(aug_no):
        l = np.random.beta(alpha, alpha)
        l = max(l, 1 - l)

        #print(X.shape[0])
        idx = np.random.permutation(X.shape[0])

        input_a, input_b = X, X[idx]
        target_a, target_b = Y, Y[idx]

        mixed_input = l * input_a + (1-l) * input_b
        mixed_target = l * target_a + (1-l) * target_b

        new_X.append(mixed_input)
        new_Y.append(mixed_target)
        #print(mixed_target)
    new_X = np.concatenate((new_X), axis=0)
    new_Y = np.concatenate((new_Y), axis=0)
    #print(new_Y)
    return new_X, new_Y

def greedy_selection(probs, constraint):
    #1. create the 3-d vector
    vector_list = []
    constraint_classes = []
    for i in range(probs.shape[1]):
        tmp= []
        class_per_index = np.argmax(probs, axis=1)
        #c = np.ones((probs.shape[0], 1))*i
        indices = np.expand_dims(np.arange(probs.shape[0]), axis = 1)
        value = np.expand_dims(np.amax(probs, axis=1), axis = 1)*-1
        probs[np.arange(probs.shape[0]), class_per_index] = -1
        class_per_index = np.expand_dims(class_per_index, axis = 1)
        temp_vector = np.concatenate((value,indices,class_per_index), axis=1)
        vector_list.append(temp_vector)
        constraint_classes.append(tmp)
    vector_list = np.concatenate(vector_list, axis=0)
    indx = np.argsort(vector_list[:, 0])
    vector_list = vector_list[indx]

    # 2. fill the index matrix
    labels = np.zeros(probs.shape[0])
    for j in range(vector_list.shape[0]):
        sample_index = int(vector_list[j,1])
        sample_class =int(vector_list[j,2])
        if len(constraint_classes[sample_class])< constraint:
            #print(sample_class)
            constraint_classes[sample_class].append(int(sample_index))
            labels[sample_index] = sample_class
    best_per_class = np.zeros(probs.shape[1])
    for x in range(len(constraint_classes)):
         best_per_class[x] = constraint_classes[x][0]
    #print(labels)
    return constraint_classes, labels.astype(int), best_per_class.astype(int)



def compute_optimal_transport(M, T=15,  epsilon=1e-6):
    #r is the P we discussed r.shape = n_runs x total_queries, all entries = 1
    M = M.cuda()
    r = torch.ones(1, M.shape[0])
    c = torch.ones(1, M.shape[1]) * (M.shape[0]/M.shape[1])
    r = r.cuda()
    # c is the q we discussed c.shape = n_runs x n_ways, all entries = 15
    c = c.cuda()
    M = torch.unsqueeze(M, dim=0)
    n_runs, n, m = M.shape
    P = M
    # doing the temperature T exponential here, M is distances


    u = torch.zeros(n_runs, n).cuda()
    maxiters = 1000
    iters = 1
    # normalize this matrix
   # for i in range(100):
    #P = torch.exp(T * P)#.cuda()
   # for i in range(20):
    P = torch.pow(P, T)
        #P = torch.exp(T * P)
        # print(P.view((n_runs, -1)).shape, P.shape,)
        # sums up the whole matrix P and divides each element with the sum
    P /= P.view((n_runs, -1)).sum(1).unsqueeze(1).unsqueeze(1)
    while torch.max(torch.abs(u - P.sum(2))) > epsilon:
        u = P.sum(2)
        P *= (r / u).view((n_runs, -1, 1))
        P *= (c / P.sum(1)).view((n_runs, 1, -1))
        if iters == maxiters:
            break
        iters = iters + 1
    P = torch.squeeze(P).detach().cpu().numpy()
    best_per_class = np.argmax(P, 0)
   # print(best_per_class.shape)
    labels = np.argmax(P, 1)
    return P, labels, best_per_class