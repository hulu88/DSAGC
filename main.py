from collections import Counter

from sklearn.cluster import KMeans

import utils
from model import GAT
from model_GAT import GAT as GAT_self_supervised
from evaluation import eva

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim


def trainer(dataset):
    #  模型实例化
    model_GAT = GAT(
        num_features=args.input_dim,
        hidden_size=args.hidden_size,
        embedding_size=args.embedding_size,
        alpha=args.alpha,
    ).to(device)
    print(model_GAT)
    # 加载预训练权重模型
    model_GAT.load_state_dict(torch.load(args.pretrain_path, map_location='cpu'))

    # data process
    dataset = utils.data_preprocessing(dataset)
    adj = dataset.adj.to(device)
    adj_label = dataset.adj_label.to(device)
    M = utils.get_M(adj).to(device)

    # data and label
    data = torch.Tensor(dataset.x).to(device)
    y = dataset.y.cpu().numpy()

    with torch.no_grad():
        _, z = model_GAT(data, adj, M)
    features = z.data.cpu().numpy()

    # get kmeans and pretrain cluster result
    # # 谱聚类
    # sc = SpectralClustering(n_clusters=args.n_clusters, affinity="nearest_neighbors")
    # y_pred = sc.fit_predict(features)
    # acc, nmi, ari, f1 = eva(y, y_pred, 'pretrain-谱聚类')

    # K-means聚类
    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(features)  # (2708,)
    acc, nmi, ari, f1 = eva(y, y_pred, 'pretrain-Kmeans', show=True)

    # 聚类中心获取
    cluster = kmeans.cluster_centers_  # 获取聚类中心  # (7, 16)
    # result = [acc, nmi, ari, f1]

    # 计算可信赖样本的精确度
    # 验证标签和聚类中心之间的关联度
    for i in range(y_pred.shape[0]):
        dis = []
        for j in range(cluster.shape[0]):
            dist = np.sqrt(np.sum(np.square(features[i] - cluster[j])))
            dis.append(dist)
        if not dis.index(min(dis))==y_pred[i]:
            print("error")
        dis.clear()

    # 获取到簇中心最小的距离
    distances_samples = []  # 每个样本到该簇类中心的距离
    for i in range(features.shape[0]):
        dist_temp = np.sum(np.square(features[i] - cluster[y_pred[i]]))  # 欧式距离
        distances_samples.append(dist_temp)
    distances_samples = np.array(distances_samples)

    # 可信赖样本的阈值
    alpha_k = 0.36  # 0.3
    print("可信赖样本的个数", np.sum(np.array(distances_samples) < alpha_k))
    realiable_sample = np.argwhere(distances_samples < alpha_k)
    # print(distances_samples[realiable_sample])  # 输出这些距离


    # 测试
    y_new = [int(y[i]) for i in realiable_sample]
    y_pred_new = [int(y_pred[i]) for i in realiable_sample]
    eva(y_new, y_pred_new, 'pretrain-可信赖样本', show=True)
    print(Counter(y_pred_new))
    print(Counter(y))

    # 自监督学习  # x, y, adj, y_pre, sample_index)
    train_self_supervised(x=data, y=y, adj=adj, y_pred=y_pred, sample_index=realiable_sample, epoch=400)


def train_self_supervised(x, y, adj, y_pred, sample_index, epoch=200):
    # 初始化模型
    sample_index = [i[0] for i in sample_index]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # x, y, adj, y_pred = x.to(device), torch.LongTensor(y).to(device), adj.to(device), torch.LongTensor(y_pred).to(device)
    x, adj, y_pred = x.to(device), adj.to(device), torch.LongTensor(y_pred).to(device)
    print("自监督训练开始,  GAT模型结构图：")

    model = GAT_self_supervised(nfeat=x.shape[1], nhid=8, nclass=int(y.max()) + 1, dropout=0.18, nheads=8, alpha=0.2)
    print(model)
    # summary(model, input_size=(), device='cpu')
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-5)
    acc_best = 0;
    for i in range(epoch):
        model.train()
        optimizer.zero_grad()
        output = model(x, adj)
        loss_train = F.nll_loss(output[sample_index], y_pred[sample_index])
        loss_train.backward()
        optimizer.step()

        # 评估指标
        y_pre_GAT = np.argmax(output.cpu().detach().numpy(), axis=1)
        y_pre_kmeans = y_pred.cpu().detach().numpy()

        acc_train, nmi_train, ari_train, f1_train = eva(y_pre_GAT[sample_index], y_pre_kmeans[sample_index], '{}'.format(i))
        acc_test, nmi_test, ari_test, f1_test = eva(y_pre_GAT, y, 'Test: {}'.format(i))
        if acc_test > acc_best:
            acc_best = acc_test
            torch.save(model.state_dict(), f"pretrain/self_GCN_{args.name}_{i}_{acc_best}.pkl")
            print(f"epoch - Train {i}:acc {acc_train:.4f}, nmi {nmi_train:.4f}, ari {ari_train:.4f}, f1 {f1_train:.4f}")
            print(f"epoch {i}:acc {acc_test:.4f}, nmi {nmi_test:.4f}, ari {ari_test:.4f}, f1 {f1_test:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default='Cora')  # Citeseer, Cora
    # parser.add_argument('--epoch', type=int, default=30)  # 预训练模型中的epoch
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--n_clusters', default=6, type=int)
    parser.add_argument('--update_interval', default=1, type=int)  # [1,3,5]
    parser.add_argument('--hidden_size', default=256, type=int)
    parser.add_argument('--embedding_size', default=16, type=int)
    parser.add_argument('--weight_decay', type=int, default=5e-3)
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    args = parser.parse_args()

    # 判断文件夹(用于保存模型)是否存在; 不存在，就创建
    if os.path.exists('pretrain/') == False:  #
        os.makedirs('pretrain/')

    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")

    datasets = utils.get_dataset(args.name)
    dataset = datasets[0]

    if args.name == 'Citeseer':
        args.lr = 0.0001
        args.k = None
        args.n_clusters = 6
    elif args.name == 'Cora':
        args.lr = 0.0001
        args.k = None
        args.n_clusters = 7
    elif args.name == "Pubmed":
        args.lr = 0.001
        args.k = None
        args.n_clusters = 3
    else:
        args.k = None

    args.pretrain_path = f'pretrain/GAE_{args.name}.pkl'
    args.input_dim = dataset.num_features

    print(args)
    trainer(dataset)



