import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
import torch
from torch.utils.data import DataLoader
from clustering import clustering
from scipy.optimize import linear_sum_assignment

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C)
    return feat_mean, feat_std

def reassign(y_before, y_pred):
    assert y_before.size == y_pred.size
    D = max(y_before.max(), y_pred.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_before.size):
        w[y_before[i], y_pred[i]] += 1
    row_ind, col_ind= linear_sum_assignment(w.max() - w)
    return col_ind

def compute_features(dataloader, model, N, device):
    model.eval()
    # discard the label information in the dataloader
    for i, (input_tensor, _, _) in enumerate(dataloader):
        with torch.no_grad():
            input_var = input_tensor.to(device)
            aux = model.domain_features(input_var).data.cpu().numpy()
            if i == 0:
                features = np.zeros((N, aux.shape[1])).astype('float32')

            if i < len(dataloader) - 1:
                features[i * dataloader.batch_size: (i + 1) * dataloader.batch_size] = aux.astype('float32')
            else:
                # special treatment for final batch
                features[i * dataloader.batch_size:] = aux.astype('float32')
    return features

def compute_instance_stat(dataloader, model, N, device):
    model.eval()
    for i, (input_tensor, _, _) in enumerate(dataloader):
        with torch.no_grad():
            input_var = input_tensor.to(device)
            conv_feats = model.conv_features(input_var)
            for j, feats in enumerate(conv_feats):
                feat_mean, feat_std = calc_mean_std(feats)
                if j == 0:
                    aux = torch.cat((feat_mean, feat_std), 1).data.cpu().numpy()
                else:
                    aux = np.concatenate((aux, torch.cat((feat_mean, feat_std), 1).data.cpu().numpy()), axis=1)
            if i == 0:
                features = np.zeros((N, aux.shape[1])).astype('float32')
            if i < len(dataloader) - 1:
                features[i * dataloader.batch_size: (i + 1) * dataloader.batch_size] = aux.astype('float32')
            else:
                # special treatment for final batch
                features[i * dataloader.batch_size:] = aux.astype('float32')
    print(features.shape)
    return features

def arrange_clustering(images_lists):
    pseudolabels = []
    image_indexes = []
    for cluster, images in enumerate(images_lists):
        image_indexes.extend(images)
        pseudolabels.extend([cluster] * len(images))
    indexes = np.argsort(image_indexes)
    return np.asarray(pseudolabels)[indexes]
            

def domain_split(dataset, model, device, cluster_before, filename, epoch, nmb_cluster=3, method='Kmeans', pca_dim=256, batchsize=128, num_workers=4, whitening=False, L2norm=False, instance_stat=True):
    cluster_method = clustering.__dict__[method](nmb_cluster, pca_dim, whitening, L2norm)

    dataset.set_transform('val')
    dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=False, num_workers=num_workers)

    if instance_stat:
        features = compute_instance_stat(dataloader, model, len(dataset), device)
    else:
        features = compute_features(dataloader, model, len(dataset), device)

    clustering_loss = cluster_method.cluster(features, verbose=False)
    cluster_list = arrange_clustering(cluster_method.images_lists)

    class_nmi = normalized_mutual_info_score(
        cluster_list, dataloader.dataset.labels, average_method='geometric')
    domain_nmi = normalized_mutual_info_score(
        cluster_list, dataloader.dataset.domains, average_method='geometric')
    before_nmi = normalized_mutual_info_score(
        cluster_list, cluster_before, average_method='arithmetic')
    
    log = 'Epoch: {}, NMI against class labels: {:.3f}, domain labels: {:.3f}, previous assignment: {:.3f}'.format(epoch, class_nmi, domain_nmi, before_nmi)
    print(log)
    if filename:
        with open(filename, 'a') as f:
            f.write(log + '\n')
        
    mapping = reassign(cluster_before, cluster_list)
    cluster_reassign = [cluster_method.images_lists[mapp] for mapp in mapping]
    dataset.set_transform(dataset.split)
    return arrange_clustering(cluster_reassign)
