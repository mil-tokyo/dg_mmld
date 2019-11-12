from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import torch
from copy import deepcopy
from clustering.domain_split import calc_mean_std
from sklearn.decomposition import PCA

def plot_embedding(X, y, d, dir_name, img_name):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    # Plot colors numbers
    plt.figure(figsize=(12,10))
    ax = plt.subplot(111)
#     color=['r', 'g', 'b', 'c']
    color = ['crimson', 'limegreen', 'royalblue', 'c']
    shape = ['s', 'o', '^', 'D']
    label=['Photo', 'Art Painting', 'Cartoon', 'Sketch']
    # label=['Photo', 'Cartoon', 'Sketch']
    flag=[0, 0, 0, 0]
    k=0
    for i in range(X.shape[0]):
        # plt.text(X[i, 0], X[i, 1], str(int(y[i])), color=color[d[i]])#, fontdict={'weight': 'bold', 'size': 15})
        if flag[d[i]]==0 and k==d[i]:
            print('label')
            plt.plot(X[i, 0], X[i, 1], 'o', color=color[d[i]], label=label[d[i]])
            flag[d[i]]=1
            k+=1
        else:
            plt.plot(X[i, 0], X[i, 1], 'o', color=color[d[i]])
        # plt.plot(X[i, 0], X[i, 1], shape[int(y[i])], color=color[d[i]])#, fontdict={'weight': 'bold', 'size': 15})
        # plt.plot(X[i, 0], X[i, 1], 'o', color='blue', markersize=6)
    plt.xticks([]), plt.yticks([])
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.legend(loc='upper center', frameon=True, fontsize=44, ncol=2, bbox_to_anchor=(0.5, 0.0), columnspacing=0.0, handletextpad=0.0)
#    plt.title(title)
    # Check extension in case.
    plt.savefig(dir_name + '/' + img_name + '.png', transparent=True, bbox_inches="tight", pad_inches=0.0)
    plt.savefig(dir_name + '/' + img_name+  '.eps', transparent=True, bbox_inches="tight", pad_inches=0.0)
    plt.savefig(dir_name + '/' + img_name+  '.pdf', transparent=True, bbox_inches="tight", pad_inches=0.0)
    plt.show()
    plt.close()
    
def VisualizePerformance(model, dataloader_list, device, dir_name, img_name, model_name, conv=False):
    model.eval()
    embeddings, classes, domains = [], [], []
    for i, dataloader in enumerate(dataloader_list):
        for inputs, labels in dataloader:
            with torch.no_grad():
                classes.extend(labels.numpy())
                domains.extend([i for _ in range(inputs.shape[0])])
                inputs = inputs.to(device)
                labels = labels.to(device)
                # features =  model.domain_features(inputs)
                if model_name == 'caffenet':
                    features = model.features(inputs*57.6)
                    features = features.view(features.size(0), -1)
                    features= model.classifier(features)
                    features = features.cpu().numpy()
                    
                elif model_name == 'resnet':
                    x = model.conv1(inputs)
                    x = model.bn1(x)
                    x = model.relu(x)
                    x = model.maxpool(x)
                    x = model.layer1(x)
                    x = model.layer2(x)
                    x = model.layer3(x)
                    x = model.layer4(x)
                    x = model.avgpool(x)
                    x = x.view(x.size(0), -1)
                    features = x.cpu().numpy()
                
                    
                elif model_name == 'DGDC_caffenet' or 'DGDC_resnet': 
                    if conv:
                        conv_feats = model.conv_features(inputs)
                        for j, feats in enumerate(conv_feats):
                            feat_mean, feat_std = calc_mean_std(feats)
                            if j == 0:
                                features = torch.cat((feat_mean, feat_std), 1).data.cpu().numpy()
                                # features = feat_mean.data.cpu().numpy()
                            else:
                                features = np.concatenate((features, 
                                                           torch.cat((feat_mean, feat_std), 1).data.cpu().numpy()), axis=1)
                    else: 
                        features = model.features(inputs)
                        features = features.cpu().numpy()

                print(features.shape)
                embeddings.extend(features)
            break
    # pca = PCA(n_components=2)
    # comp = pca.fit_transform(embeddings) 
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=1000)
    comp= tsne.fit_transform(embeddings)
    plot_embedding(comp, classes, domains, dir_name, img_name)
    
def VisualizeClustering(model, dataloader, device, dir_name, img_name, model_name, conv=False):
    model.eval()
    embeddings, classes, domains, clusters = [], [], [], []
    # for i, dataloader in enumerate(dataloader_list):
    for inputs, labels, domain, cluster in dataloader:
        with torch.no_grad():
            classes.extend(labels.numpy())
            domains.extend(domain.numpy())
            clusters.extend(cluster.numpy())

            inputs = inputs.to(device)
            labels = labels.to(device)
            # features =  model.domain_features(inputs)
            if model_name == 'caffenet':
                features = model.features(inputs*57.6)
                features = features.view(features.size(0), -1)
                features= model.classifier(features)
                features = features.cpu().numpy()

            elif model_name == 'resnet':
                x = model.conv1(inputs)
                x = model.bn1(x)
                x = model.relu(x)
                x = model.maxpool(x)
                x = model.layer1(x)
                x = model.layer2(x)
                x = model.layer3(x)
                x = model.layer4(x)
                x = model.avgpool(x)
                x = x.view(x.size(0), -1)
                features = x.cpu().numpy()


            elif model_name == 'DGDC_caffenet' or 'DGDC_resnet': 
                if conv:
                    conv_feats = model.conv_features(inputs)
                    for j, feats in enumerate(conv_feats):
                        feat_mean, feat_std = calc_mean_std(feats)
                        if j == 0:
                            features = torch.cat((feat_mean, feat_std), 1).data.cpu().numpy()
                            # features = feat_mean.data.cpu().numpy()
                        else:
                            features = np.concatenate((features, 
                                                       torch.cat((feat_mean, feat_std), 1).data.cpu().numpy()), axis=1)
                else: 
                    features = model.features(inputs)
                    features = features.cpu().numpy()

            print(features.shape)
            embeddings.extend(features)
            break
#     pca = PCA(n_components=2)
#     comp = pca.fit_transform(embeddings) 
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=1000)
    comp= tsne.fit_transform(embeddings)
    plot_embedding(comp, clusters, domains, dir_name, img_name)