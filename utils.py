import torch
import warnings
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from variable import *
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
from torchmetrics.functional import r2_score
import ot


def tsne2d(*tensors, labels=None, title="", perplexity=30, n_iter=1000):
    """
    Projects multiple high-dim tensors into 2D space using t-SNE.
    
    Args:
        *tensors: Variable number of PyTorch tensors (Batch, Dim)
        labels: List of strings for the legend
        title: Plot title
        perplexity: t-SNE perplexity (related to number of nearest neighbors)
        n_iter: Number of iterations for optimization
    """
    # 1. Prepare Data
    np_data = [t.detach().cpu().numpy() for t in tensors]
    sizes = [d.shape[0] for d in np_data]
    
    # 2. Stack all data
    combined = np.vstack(np_data)
    
    # 3. Normalization (Still recommended for t-SNE distance calculations)
    combined_norm = StandardScaler().fit_transform(combined)
    
    # 4. t-SNE Reduction
    # init='pca' is generally more stable and faster than random initialization
    tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=n_iter, init='pca', learning_rate='auto', random_state=42)
    projected = tsne.fit_transform(combined_norm)
    
    # 5. Plotting
    plt.figure(figsize=(10, 7))
    
    # Custom colors: Purple, Blue, Orange, Green
    my_cmap = ListedColormap(['purple', 'blue', 'orange', 'green', 'magenta', 'cyan', 'gold', 'midnightblue'])
    colors = [my_cmap(i % my_cmap.N) for i in range(len(np_data))]
    
    current_idx = 0
    for i, size in enumerate(sizes):
        start, end = current_idx, current_idx + size
        label = labels[i] if labels and i < len(labels) else f"Tensor {i+1}"
        
        plt.scatter(
            projected[start:end, 0], 
            projected[start:end, 1], 
            color=colors[i], 
            label=label, 
            alpha=0.6,
            edgecolors='white',
            linewidths=0.2,
            s=8
        )
        current_idx = end
    plt.axis('off')
    plt.title(title)
    plt.legend()
    plt.show()


def pca2d(*tensors, labels=None, title=""):
    """
    Projects multiple high-dim tensors into 2D space.
    
    Args:
        *tensors: Variable number of PyTorch tensors (Batch, Dim)
        labels: List of strings for the legend
        title: Plot title
    """
    np_data = [t.detach().cpu().numpy() for t in tensors]
    # Record the sizes to split them later
    sizes = [d.shape[0] for d in np_data]
    # 2. Stack all data to normalize and fit PCA together
    combined = np.vstack(np_data)
    # 3. Normalization (Crucial for PCA)
    combined_norm = StandardScaler().fit_transform(combined)
    # 4. PCA Reduction
    pca = PCA(n_components=2)
    projected = pca.fit_transform(combined_norm)
    # 5. Plotting
    plt.figure(figsize=(10, 7))
    # colors = plt.cm.rainbow(np.linspace(0, 1, len(np_data)))
    # Define the custom map
    my_cmap = ListedColormap(['purple', 'blue', 'orange', 'green', 'magenta', 'cyan', 'gold', 'midnightblue'])
    colors = [my_cmap(i % my_cmap.N) for i in range(len(np_data))]
    current_idx = 0
    for i, size in enumerate(sizes):
        start, end = current_idx, current_idx + size
        label = labels[i] if labels and i < len(labels) else f"Tensor {i+1}"
        plt.scatter(
            projected[start:end, 0], 
            projected[start:end, 1], 
            color=colors[i], 
            label=label, 
            alpha=0.6,
            edgecolors='white',
            linewidths=0.2,
            s=8
        )
        current_idx = end
    # Calculate variance explained for the axes
    var_exp = pca.explained_variance_ratio_ * 100
    plt.xlabel(f"X-feature ({var_exp[0]:.1f}% variance)")
    plt.ylabel(f"Y-feature ({var_exp[1]:.1f}% variance)")
    plt.title(title)
    plt.legend()
    plt.show()

def prepare(tensor: torch.Tensor, number, device, max_iter=100, random_state: int = 999):
    # Handle the zero-cluster case
    if number <= 0:
        return None, torch.zeros((0, tensor.size(1))).to(device)
    
    # 1) convert to numpy for sklearn
    data = tensor.cpu().numpy()
    # 2) cluster into k groups
    kmeans = KMeans(n_clusters=number, max_iter=max_iter, random_state=random_state).fit(data)
    # labels = kmeans.fit_predict(data)       # array of length s1
    centroids = kmeans.cluster_centers_     # shape (k, s2)
    centroids = (torch.tensor(centroids)).to(device)
    return kmeans, centroids

def assign_prototype(kmeans, data, device):
    if kmeans is None:
        # Return a zero tensor of the same batch size and embedding dim
        return torch.zeros_like(data).to(device), None
    data = data.cpu().numpy()
    labels = kmeans.predict(data)       # array of length s1
    predict = kmeans.cluster_centers_[labels]     # shape (k, s2)
    return (torch.tensor(predict)).to(device), (torch.tensor(labels)).to(device)

def save_output(save_weight, train, test):
    with open(save_weight + 'train-test.txt', 'w') as f:
        f.write("=== Output from training ===\n")
        f.write(str(train) + "\n")
        
        f.write("=== Output from testing ===\n")
        f.write(str(test) + "\n")

def save_overlay_output(save_weight, train, test):
    with open(save_weight + 'overlay-train-test.txt', 'w') as f:
        f.write("=== Output from training ===\n")
        f.write(str(train) + "\n\n")
        
        f.write("=== Output from testing ===\n")
        f.write(str(test) + "\n\n")
        
def draw_train(train_losses, val_losses):
    iters = [iter * ITERS_PER_EVAL for iter in range(len(train_losses))]
    valid_iters = [iter * ITERS_PER_EVAL for iter in range(len(val_losses))]
    plt.plot(iters, train_losses, label='train')
    plt.plot(valid_iters, torch.Tensor.cpu(torch.tensor(val_losses)), label='validation')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.title('training and validation loss curves')
    plt.legend()
    plt.show()

def generate_paired_data(sfinal_user, tfinal_user, device, neg_num=1):
    """
    生成所有源嵌入和目标嵌入的配对输入，以及对应的标签（仅对应索引为1，其余为0）
    参数：
        sfinal_user: 源嵌入，shape [batch_size, src_emb_dim]
        tfinal_user: 目标嵌入，shape [batch_size, tgt_emb_dim]
    返回：
        paired_inputs: 所有配对的输入，shape [batch_size×batch_size, src_emb_dim + tgt_emb_dim]
        paired_labels: 对应标签，shape [batch_size×batch_size, 1]（1表示匹配，0表示不匹配）
    """
    batch_size = sfinal_user.shape[0]
    
    # ========== 步骤1：生成N个正例（GPU上直接拼接） ==========
    pos_inputs = torch.cat([sfinal_user, tfinal_user], dim=-1)  # [N, src_dim+tgt_dim]，GPU张量
    pos_labels = torch.ones(batch_size, 1, device=device)      # 直接在GPU上创建标签

    # ========== 步骤2：生成N个负例（GPU向量化操作，无循环） ==========
    # 核心：生成每个样本的负例索引（GPU上完成，替代CPU的random.choice）
    # 1. 生成基础索引 [0,1,2,...,N-1]
    base_indices = torch.arange(batch_size, device=device)  # [N]，GPU张量
    # 2. 为每个索引生成排除自身的随机负例索引（GPU并行）
    # 方法：对每个位置i，随机打乱索引后取第一个≠i的索引
    shuffled_indices = torch.stack([torch.randperm(batch_size, device=device) for _ in range(batch_size)])  # [N, N]
    # 过滤掉等于自身的索引，取第一个作为负例索引
    mask = (shuffled_indices != base_indices.unsqueeze(1))  # [N, N]，True表示索引≠自身
    # 取每个行第一个True对应的索引（即每个i的第一个非自身索引）
    neg_indices = shuffled_indices[mask].reshape(batch_size, -1)[:, 0]  # [N]，GPU张量

    # 3. 用向量化方式生成所有负例（替代for循环，GPU并行）
    neg_src = sfinal_user[base_indices]  # [N, src_dim]（等价于原循环的sfinal_user[i]）
    neg_tgt = tfinal_user[neg_indices]  # [N, tgt_dim]（每个i对应随机负例的目标嵌入）
    neg_inputs = torch.cat([neg_src, neg_tgt], dim=-1)  # [N, src_dim+tgt_dim]，GPU张量
    neg_labels = torch.zeros(batch_size, 1, device=device)  # GPU上创建负例标签

    # ========== 步骤3：合并并打乱（全程GPU操作） ==========
    paired_inputs = torch.cat([pos_inputs, neg_inputs], dim=0)  # [2N, src_dim+tgt_dim]
    paired_labels = torch.cat([pos_labels, neg_labels], dim=0)  # [2N, 1]

    # 随机打乱（GPU上完成）
    shuffle_idx = torch.randperm(paired_inputs.shape[0], device=device)
    paired_inputs = paired_inputs[shuffle_idx]
    paired_labels = paired_labels[shuffle_idx]
    
    return paired_inputs, paired_labels

def overlap(overlap_path):
    """
    USAGE:
    for s_uid, t_uid in self.overlap_src2tgt.items():
        self.user_emb_tgt.weight.data[t_uid] = self.user_emb_src.weight.data[s_uid]
    """
    # Build mapping from source_uid -> target_uid for overlapping users
    overlap_src2tgt = {}
    with open(overlap_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: 
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"Bad format in overlap file: {line}")
            src_uid = int(parts[0])
            tgt_uid = int(parts[1])
            overlap_src2tgt[src_uid] = tgt_uid
    return overlap_src2tgt

def mini_batch_iterator(train_uid: torch.Tensor,
                        train_iid: torch.Tensor,
                        train_rates: torch.Tensor,
                        batch_size: int,
                        shuffle: bool = True):
    """
    Generator that yields mini-batches from the training tensors.
    Args:
        train_uid (torch.Tensor): Tensor of user IDs, shape [N].
        train_iid (torch.Tensor): Tensor of item IDs, shape [N].
        train_rates (torch.Tensor): Tensor of ratings, shape [N].
        batch_size (int): Size of each mini-batch.
        shuffle (bool): Whether to shuffle the data before each epoch.
    Yields:
        (batch_uid, batch_iid, batch_rates): Mini-batch tensors.
    """
    assert len(train_uid) == len(train_iid) == len(train_rates), \
        "All input tensors must have the same length."
    n = len(train_uid)
    indices = torch.arange(n, device=train_uid.device)
    if shuffle:
        indices = indices[torch.randperm(n, device=train_uid.device)]
    for start_idx in range(0, n, batch_size):
        end_idx = min(start_idx + batch_size, n)
        batch_idx = indices[start_idx:end_idx]
        yield train_uid[batch_idx], train_iid[batch_idx], train_rates[batch_idx]

def sample_mini_batch(train_uid: torch.Tensor,
                      train_iid: torch.Tensor,
                      train_rates: torch.Tensor,
                      batch_size: int):
        unique_uids = torch.unique(train_uid)
        perm = torch.randperm(len(unique_uids))
        sampled_uids = unique_uids[perm[:batch_size]]
        mask = torch.isin(train_uid, sampled_uids)
        batch_idx   = mask.nonzero(as_tuple=True)[0]
        batch_uid   = train_uid[batch_idx]
        batch_iid   = train_iid[batch_idx]
        batch_rates = train_rates[batch_idx]
        return batch_uid, batch_iid, batch_rates, sampled_uids

def sample_mini_batch_sequential(test_uid: torch.Tensor,
                               test_iid: torch.Tensor,
                               test_rates: torch.Tensor,
                               batch_size: int,
                               current_idx: int = 0):
    total_inter = len(test_uid)
    if current_idx >= total_inter:
        return None, None, None, 0
    # Calculate the end index for this batch
    end_idx = min(current_idx + batch_size, total_inter)
    batch_uid = test_uid[current_idx:end_idx]
    batch_iid = test_iid[current_idx:end_idx]
    batch_rates = test_rates[current_idx:end_idx]
    # Return both the batch data and the next starting index
    return batch_uid, batch_iid, batch_rates, end_idx

def r2(predicted_ratings, ttest_rates):
    result = r2_score(predicted_ratings, ttest_rates)
    # print(f"r2 score: {round(result.item(), 5)}")
    return result

def rmse(predicted_ratings, true_ratings):
    squared_difference = (true_ratings - predicted_ratings)**2
    result = torch.sqrt(squared_difference.mean())
    # print(f"rmse: {round(result.item(), 5)}")
    return result

def mae(predicted_ratings, true_ratings):
    squared_difference = true_ratings - predicted_ratings
    result = torch.abs(squared_difference).mean()
    # print(f"mae: {round(result.item(), 5)}")
    return result

def MSELOSS(predicted_ratings, true_ratings):
    squared_difference = (true_ratings - predicted_ratings)**2
    return squared_difference.mean()

def WDLOSS(xs, xt, lambda_e=0.01, numItermax=100, device='cpu'):
    warnings.filterwarnings("ignore", category=UserWarning, module="ot")
    a = (torch.tensor(ot.utils.unif(xs.size(0)), dtype=torch.float)).to(device)
    b = (torch.tensor(ot.utils.unif(xt.size(0)), dtype=torch.float)).to(device)
    # Compute ground cost matrix 
    M = ot.dist(xs, xt, metric='cosine').to(device)
    # ot.sinkhorn returns the optimal transport matrix, the distance is the final cost
    # WD = ot.sinkhorn2(a, b, M, lambda_e, numItermax=numItermax, method='sinkhorn_log')
    plan = ot.sinkhorn(a, b, M, reg=lambda_e, method='sinkhorn_log', numItermax=numItermax)
    WD = torch.sum(plan * M)
    return WD.to(device), plan.to(device)

def load_main_pt(folder_path, device):
    data = torch.load(folder_path)
    # Access the overall domain statistics
    n_user = data['n_user']
    n_item = data['n_item']
    # Access the overall domain statistics
    tn_user = data['tn_user']
    tn_item = data['tn_item']
    # Access the training interaction tensors
    train_uid = data['train_uid']
    train_iid = data['train_iid']
    train_rates = data['train_rates']
    # Access the testing interaction tensors
    test_uid = data['test_uid']
    test_iid = data['test_iid']
    test_rates = data['test_rates']
    return train_uid.to(device), train_iid.to(device), train_rates.to(device), test_uid.to(device), test_iid.to(device), test_rates.to(device)\
        , n_user, n_item, tn_user, tn_item

def get_pretrain_data(isFull: bool, pt_path, movie_json_path, movie_user_map_path, movie_item_map_path, device):
    # loading data
    ttrain_uid, ttrain_iid, ttrain_rates, ttest_uid, ttest_iid, ttest_rates, n_user, n_item, tn_user, tn_item\
    = load_main_pt(pt_path, 'cpu')
    # Convert ttest_uid tensor to a Python set for fast O(1) filtering
    test_overlap_users = set(ttest_uid.cpu().numpy())

    # Map format: raw string ID -> processed integer ID
    user_map_df = pd.read_csv(movie_user_map_path, sep=' ', header=None, names=['reviewerID', 'user_id'])
    user_map = dict(zip(user_map_df['reviewerID'], user_map_df['user_id']))
    item_map_df = pd.read_csv(movie_item_map_path, sep=' ', header=None, names=['asin', 'item_id'])
    item_map = dict(zip(item_map_df['asin'], item_map_df['item_id']))

    # Load the raw JSON file
    df = pd.read_json(movie_json_path, lines=True)
    df = df[['reviewerID', 'asin', 'overall']]
    # Apply mappings to convert strings to integer IDs
    df['uid'] = df['reviewerID'].map(user_map)
    df['iid'] = df['asin'].map(item_map)
    # Drop any unmapped/NaN rows (safety check)
    df = df.dropna(subset=['uid', 'iid'])
    df['uid'] = df['uid'].astype(int)
    df['iid'] = df['iid'].astype(int)
    
    if isFull is False:
        # FILTER OUT TEST USERS (Avoid Data Leakage)
        print(f"Total interactions before filtering: {len(df)}")
        # Keep only rows where 'uid' is NOT in the test_overlap_users set
        df_filtered = df[~df['uid'].isin(test_overlap_users)]
        print(f"Total interactions after excluding test overlap users: {len(df_filtered)}")
    else:
        df_filtered = df

    # Convert the filtered interactions to PyTorch Tensors
    train_uid = torch.tensor(df_filtered['uid'].values, dtype=torch.long)
    train_iid = torch.tensor(df_filtered['iid'].values, dtype=torch.long)
    train_rates = torch.tensor(df_filtered['overall'].values, dtype=torch.float32)

    tn_user = df['uid'].nunique()
    tn_item = df['iid'].nunique()

    return train_uid.to(device), train_iid.to(device), train_rates.to(device), tn_user, tn_item