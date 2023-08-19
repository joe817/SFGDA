# coding=utf-8
import os

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser
from gnn.cached_gcn_conv import CachedGCNConv
from gnn.dataset.DomainData import DomainData
from gnn.ppmi_conv import PPMIConv
from network import *
import random
import numpy as np
import torch
import torch.functional as F
from torch import nn
import torch.nn.functional as F
import itertools
import time
import warnings
import math
from draw import dual_lineplot


warnings.filterwarnings("ignore", category=UserWarning)


def index2dense(edge_index,nnode=2708):
    indx = edge_index.numpy()
    adj = np.zeros((nnode,nnode),dtype = 'int8')
    adj[(indx[0],indx[1])]=1
    new_adj = torch.from_numpy(adj).float()
    return new_adj


def data_load(args):
    dataset = DomainData("data/{}".format(args.source), name=args.source)
    source_data = dataset[0]
    print(args.source)
    print(source_data)

    dataset = DomainData("data/{}".format(args.target), name=args.target)
    target_data = dataset[0]
    print(args.target)
    print(target_data)

    args.num_classes = dataset.num_classes
    args.num_features = source_data.x.size(1)

    target_data.num_classes = dataset.num_classes

    ppr_file = "{}/{}_ppr.pt".format('tmp', args.target)

    if os.path.exists(ppr_file):
        target_data.ppr = torch.load(ppr_file)
    else:
        print("Processing Pagerank...")
        pr_prob = 1 - args.pagerank_prob
        A = index2dense(target_data.edge_index, target_data.num_nodes)
        A_hat   = A + torch.eye(A.size(0))# add self-loop
        D       = torch.diag(torch.sum(A_hat,1))
        D       = D.inverse().sqrt()
        A_hat   = torch.mm(torch.mm(D, A_hat), D)
        target_data.ppr = pr_prob * ((torch.eye(A.size(0)) - (1 - pr_prob) * A_hat).inverse())
        torch.save(target_data.ppr, ppr_file) 
        print("Done Pagerank.")
        
    source_data = source_data.to(args.device)
    target_data = target_data.to(args.device)

    source_size = source_data.size(0)
    train_index, test_index = train_test_split(range(source_size),test_size=args.label_rate, train_size=args.label_rate)
    
    print('<------------------------------------------------------>')
    print('train_index: ', len(train_index))
    print('test_index: ', len(test_index))
    print('<------------------------------------------------------>')
    
    train_mask = np.zeros(source_size, dtype=bool)
    test_mask = np.zeros(source_size, dtype=bool)
    train_mask[train_index] = True
    test_mask[test_index] = True
    train_mask = torch.tensor(train_mask).to(args.device)
    test_mask = torch.tensor(test_mask).to(args.device)

    data_loader = {}
    data_loader['source_data'] = source_data
    data_loader['target_data'] = target_data
    data_loader['source_train_mask'] = train_mask
    data_loader['source_test_mask'] = test_mask

    return data_loader


def get_node_central_weight(args, target_data, seudo_label):
    ppr_matrix = target_data.ppr
    
    gpr_matrix = []
    
    for iter_c in range(target_data.num_classes):
        
        iter_gpr = torch.mean(ppr_matrix[seudo_label == iter_c],dim=0).squeeze()
        
        gpr_matrix.append(iter_gpr)
        
    
    gpr_matrix = torch.stack(gpr_matrix,dim=0).transpose(0,1)
    
    
    target_size = target_data.size(0)
    
    gpr_sum = torch.sum(gpr_matrix,dim=1)
    gpr_rn  = gpr_sum.unsqueeze(1) - gpr_matrix
    rn_matrix = torch.mm(ppr_matrix, gpr_rn)
    
    label_matrix = F.one_hot(seudo_label, gpr_matrix.size(1)).float() 
    rn_matrix = torch.sum(rn_matrix * label_matrix,dim=1)
    
    totoro_list   = rn_matrix.tolist()
    id2totoro     = {i:totoro_list[i] for i in range(len(totoro_list))}
    sorted_totoro = sorted(id2totoro.items(),key=lambda x:x[1],reverse=False)
    id2rank       = {sorted_totoro[i][0]:i for i in range(target_size)}
    totoro_rank   = [id2rank[i] for i in range(target_size)]
    
    rn_weight = [(args.low + 0.5 * (args.high - args.low) * (1 + math.cos(x*1.0*math.pi/(target_size-1)))) for x in totoro_rank]
    rn_weight = torch.from_numpy(np.array(rn_weight)).type(torch.FloatTensor)

    return rn_weight



# 特征提取器 和 分类器   
def encoder_and_cls(args):

    encoder = GNN(args, type="gcn").to(args.device)
    if args.PPMI:
        encoder = GNN(args, base_model=encoder, type="ppmi", path_len=10).to(args.device)
    
    encoder.load_state_dict(torch.load(os.path.join(args.output_dir, "source_E_" + args.source + ".pt"), map_location='cuda:0'))
    
    cls_model = nn.Sequential(
            nn.Linear(args.encoder_dim, args.num_classes),
        ).to(args.device)

    cls_model.load_state_dict(torch.load(os.path.join(args.output_dir, "source_C_" + args.source + ".pt"), map_location='cuda:0'))

    return encoder, cls_model


# 获得分类器的权重
def get_classifier_weights(cls_model):

    return cls_model[0].weight


# 获得目标域伪标签
def get_pseudo_label(args, encoded_target, cls_model):

    target_logits = cls_model(encoded_target)
    softmax_out = nn.Softmax(dim=1)(target_logits)
    pseudo_label = softmax_out.argmax(dim=1)
    
    return pseudo_label
    

# 计算目标域特征的均值和方差
def collect_target_feature_mean_std(args, target_feature, cls_model):
    
    target_data = args.data_loader['target_data']
    
    # plabels
    plabels = get_pseudo_label(args, target_feature, cls_model)
    
    assert target_feature.shape[0] == plabels.shape[0]

    class_mean = torch.zeros(target_data.num_classes, target_feature.shape[-1])
    class_std = torch.zeros(target_data.num_classes, target_feature.shape[-1])
    
    
    for c in range(target_data.num_classes):
        index = torch.where(plabels == c)[0]
        _std, _mean = torch.std_mean(target_feature[index], dim=0, unbiased=True)
        class_std[c] = _std
        class_mean[c] = _mean
    
    return class_mean.cpu(), class_std.cpu()


def construct_surrogate_feature_sampler(args, target_feature, cls_model):

        # 用于调整伪标签生成过程中的高斯分布方差的倍数
        variance_mult = 1.
        target_mean, target_std = collect_target_feature_mean_std(args, target_feature, cls_model)

        normal_sampler = {}
        
        # 锚点
        source_anchors = F.normalize(get_classifier_weights(cls_model), dim=1).cpu()

        estimated_source_mean_all = torch.zeros(args.num_classes, target_mean.shape[-1])
        estimated_source_std_all = torch.zeros(args.num_classes, target_std.shape[-1])
        
        # 估计源域分布
        for i in range(args.num_classes):
            
            cur_target_norm = target_mean[i].norm(p=2)

            estimated_source_mean = source_anchors[i] * cur_target_norm
            estimated_source_std = target_std[i]
            estimated_source_std[estimated_source_std == 0] = 1e-4
            estimated_source_std = estimated_source_std * variance_mult

            estimated_source_mean_all[i] = estimated_source_mean
            estimated_source_std_all[i] = estimated_source_std

            
            # 多元分布
            estimated_source_covariance = torch.diag_embed(estimated_source_std**2)
            normal_sampler[i] = torch.distributions.multivariate_normal.MultivariateNormal(
                loc=estimated_source_mean,
                covariance_matrix=estimated_source_covariance
            )

        
        return normal_sampler, estimated_source_mean_all, estimated_source_std_all

    
# 采样
def sample_source_features(args,target_feature,cls_model):
    
    labels = args.data_loader['target_data'].y

    source_features_cls = []
    cur_labels = []

    normal_sampler, estimated_source_mean_all, estimated_source_std_all = construct_surrogate_feature_sampler(args, target_feature, cls_model)
    for cur_label in range(args.num_classes):

        count = int((labels == cur_label).sum().item() * args.rate1)
        cur_labels += [cur_label] * count        
        samples = normal_sampler[cur_label].sample(sample_shape=(count,))
        source_features_cls += [samples]
    
    return torch.cat(source_features_cls, dim=0).to(args.device), cur_labels, estimated_source_mean_all, estimated_source_mean_all
                                                                

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        
        rate = min((args.target_epochs + 1) / args.target_epochs, 0.05)
        
        grad_output = grad_output.neg() * rate
        return grad_output, None

    
class GRL(nn.Module):
    def forward(self, input):
        return GradReverse.apply(input)

    
domain_model = nn.Sequential(
    GRL(),
    nn.Linear(512, 40),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(40, 2),
).to('cuda:0')
    

def evaluate(preds, labels):
    
    corrects = preds.eq(labels)
    accuracy = corrects.float().mean().cpu().detach()
    macro_f1 = f1_score(labels.cpu().detach(), preds.cpu().detach(), average='macro')
    micro_f1 = f1_score(labels.cpu().detach(), preds.cpu().detach(), average='micro')
    return accuracy, macro_f1, micro_f1


def source_train(args):
    
    # print('----------------------------------------------------------')
    # print('                      Source Train                        ')
    # print('----------------------------------------------------------')
    
    interval_iter = 1
    ## setting models
    encoder = GNN(args, type="gcn").to(args.device)
    if args.PPMI:
        encoder = GNN(args, base_model=encoder, type="ppmi", path_len=10).to(args.device)

    cls_model = nn.Sequential(
        nn.Linear(args.encoder_dim, args.num_classes),
    ).to(args.device)

    models = [encoder, cls_model]

    ## training
    params = itertools.chain(*[model.parameters() for model in models])
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params, lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, lr=args.learning_rate)
        optimizer = op_copy(optimizer)

    loss_func = CrossEntropyLabelSmooth(num_classes=args.num_classes, epsilon=0.1)

    source_data = args.data_loader['source_data']
    train_mask = args.data_loader['source_train_mask']
    test_mask = args.data_loader['source_test_mask']

    best_acc = (0,0,0)
    best_epoch = 0
    for model in models:
        model.train()
    for epoch in range(1, args.source_epochs):
        if args.optimizer == 'sgd':
            lr_scheduler(optimizer, iter_num = epoch, max_iter = args.source_epochs)

        encoded_source = encoder(source_data.x, source_data.edge_index, args.source)
        source_logits = cls_model(encoded_source)
        
        cls_loss = loss_func(args.device, source_logits[train_mask], source_data.y[train_mask])

        optimizer.zero_grad()
        cls_loss.backward()
        optimizer.step()

        if epoch % interval_iter == 0 or epoch == args.source_epochs:
            for model in models:
                model.eval()

            encoded_source = encoder(source_data.x, source_data.edge_index, args.source)
            encoded_source = encoded_source[test_mask]
            source_logits = cls_model(encoded_source)
            preds = source_logits.argmax(dim=1)
            labels = source_data.y[test_mask]
            accuracy, macro_f1, micro_f1 = evaluate(preds, labels)

            
            if accuracy > best_acc[0]:
                best_epoch = epoch
                best_acc = (accuracy, macro_f1, micro_f1)
                best_encoder = encoder.state_dict()
                best_cls_model = cls_model.state_dict()

            for model in models:
                model.train()
    line1 = "\nBest: epoch: {}, \nsource_acc: {:.5f}, \nmacro_f1: {:.5f}, \nmicro_f1: {:.5f}\n".format(
                best_epoch, best_acc[0], best_acc[1], best_acc[2])
    print (line1)


    torch.save(best_encoder, os.path.join(args.output_dir, "source_E_" + args.source + ".pt"))
    torch.save(best_cls_model, os.path.join(args.output_dir, "source_C_" + args.source + ".pt"))
    
    # print('----------------------------------------------------------')
    # print('                 Source Train Finished!                   ')
    # print('----------------------------------------------------------')

    return best_epoch, best_acc[0], best_acc[1], best_acc[2]




def target_train(args):
    
    # print('----------------------------------------------------------')
    # print('                     Target Train                         ')
    # print('----------------------------------------------------------')
    
    
    encoder = GNN(args, type="gcn").to(args.device)
    if args.PPMI:
        encoder = GNN(args, base_model=encoder, type="ppmi", path_len=10).to(args.device)

    cls_model = nn.Sequential(
        nn.Linear(args.encoder_dim, args.num_classes),
    ).to(args.device)
    
    encoder.load_state_dict(torch.load(os.path.join(args.output_dir, "source_E_" + args.source + ".pt"), map_location='cuda:0'))
    cls_model.load_state_dict(torch.load(os.path.join(args.output_dir, "source_C_" + args.source + ".pt"), map_location='cuda:0'))

    target_data = args.data_loader['target_data']
    encoded_target = encoder(target_data.x, target_data.edge_index, args.target)
    target_logits = cls_model(encoded_target)
    preds = target_logits.argmax(dim=1)
    
    initial_results = evaluate(preds, target_data.y)


    ## training
    interval_iter = 1

    models = [encoder, cls_model, domain_model]
    params = itertools.chain(*[model.parameters() for model in models])
    #params = encoder.parameters()
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params, lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, lr=args.learning_rate)
        optimizer = op_copy(optimizer)
    
    encoder.train()
    cls_model.train()

    best_acc = (0,0,0)
    best_epoch = 0
    for epoch in range(1, args.target_epochs):
        if args.optimizer == 'sgd':
            lr_scheduler(optimizer, iter_num = epoch, max_iter = args.source_epochs)
        
        # target
        encoded_target = encoder(target_data.x, target_data.edge_index, args.target)
        target_logits = cls_model(encoded_target)
        softmax_out = nn.Softmax(dim=1)(target_logits)
        seudo_label = softmax_out.argmax(dim=1)
        rn_weight = get_node_central_weight(args, target_data, seudo_label).to(args.device)

        
        # source
        sampled_source_features, cur_labels, estimated_source_mean_all, estimated_source_mean_all = sample_source_features(args, encoded_target, cls_model)
        
        
        # entropy_loss
        entropy_loss = torch.mean(rn_weight * Entropy(softmax_out))
        msoftmax = softmax_out.mean(dim=0)
        entropy_loss -= args.rate4 * torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))
        e_loss = entropy_loss
        
        # vae loss
        similarity = torch.mm(sampled_source_features, sampled_source_features.t())
        label_similarity = (torch.tensor(cur_labels).unsqueeze(1)==torch.tensor(cur_labels).unsqueeze(0)).float() 
        loss_vae = torch.nn.functional.binary_cross_entropy_with_logits(similarity.to(args.device), label_similarity.to(args.device), weight=(1-torch.eye(similarity.shape[0])).to(args.device)).to(args.device)   
        mu, std = estimated_source_mean_all, estimated_source_mean_all      
        std = F.softplus(std)
        loss_vae += (-0.01)*(1+2*std.log()-mu.pow(2)-std.pow(2)).sum(1).mean().div(math.log(2))
        v_loss = loss_vae
        
        # grl loss
        loss_func = nn.CrossEntropyLoss().to(args.device)
        source_domain_preds = domain_model(sampled_source_features)
        target_domain_preds = domain_model(encoded_target)
        source_domain_cls_loss = loss_func(
                source_domain_preds,
                torch.zeros(source_domain_preds.size(0)).type(torch.LongTensor).to(args.device)
            )
        target_domain_cls_loss = loss_func(
                target_domain_preds,
                torch.ones(target_domain_preds.size(0)).type(torch.LongTensor).to(args.device)
            )
        loss_grl = source_domain_cls_loss + target_domain_cls_loss
        g_loss = loss_grl

        loss = e_loss + args.rate2 * v_loss  + args.rate3 * g_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % interval_iter == 0 or epoch == args.source_epochs:
            encoder.eval()
            encoded_target = encoder(target_data.x, target_data.edge_index, args.target)
            target_logits = cls_model(encoded_target)
            preds = target_logits.argmax(dim=1)
            accuracy, macro_f1, micro_f1 = evaluate(preds, target_data.y)
            print("Epoch: {}, target_acc: {:.5f}, macro_f1: {:.5f}, micro_f1: {:.5f}".format(
                  epoch, accuracy, macro_f1, micro_f1))
            
            if accuracy > best_acc[0]:
                best_epoch = epoch
                best_acc = (accuracy, macro_f1, micro_f1)
                best_encoder = encoder.state_dict()
                best_cls_model = cls_model.state_dict()

            encoder.train()
    print()
    print()
    print('{0} =〉{1}'.format(args.source, args.target))
    line1 = "Initial: epoch:  -1, target_acc: {:.5f}, macro_f1: {:.5f}, micro_f1: {:.5f}".format(
        initial_results[0], initial_results[1], initial_results[2]) 
    line2 = "  Best : epoch: {:0>3d}, target_acc: {:.5f}, macro_f1: {:.5f}, micro_f1: {:.5f}".format(
                best_epoch, best_acc[0], best_acc[1], best_acc[2])
    print (line1)
    print (line2)
    print()
    print()
    if args.write_log:
        with open("log/text.log", 'a') as f:
            line = "{}-{}:".format(args.source, args.target) + "\n" + line1 + "\n" + line2 + "\n"
            f.write(line)
    
    # print('----------------------------------------------------------')
    # print('                 Target Train Finished!                   ')
    # print('----------------------------------------------------------')
            
    return best_epoch, best_acc[0], best_acc[1], best_acc[2]



def print_args(args):
    s = "\n==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    s += "==========================================\n"
    return s

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--source", type=str, default='acmv9')
    parser.add_argument("--target", type=str, default='citationv1')
    
    parser.add_argument("--output_dir", type=str, default='output')
    parser.add_argument("--write_log", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--PPMI", type=bool, default=False)
    parser.add_argument("--pagerank_prob", type=float, default=0.85)
    parser.add_argument("--label_rate", type=float, default=0.05)
    parser.add_argument("--encoder_dim", type=int, default=512)
    parser.add_argument("--optimizer", type=str, default='adam') #adam, sgd
    
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--source_epochs", type=int, default=100)
    parser.add_argument("--target_epochs", type=int, default=100)

    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-3)
    parser.add_argument("--drop_out", type=float, default=5e-1)

    parser.add_argument("--high", type=float, default=0.5)
    parser.add_argument("--low", type=float, default=0.4)
    
    parser.add_argument("--rate1", type=float, default=1)
    parser.add_argument("--rate2", type=float, default=0.5)
    parser.add_argument("--rate3", type=float, default=0.5)
    parser.add_argument("--rate4", type=float, default=2)

    
    
    args = parser.parse_args()
    args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print(print_args(args))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark = False

    args.data_loader = data_load(args)

    # source 
    source_train(args)

    # target
    seeds = [1, 2, 3, 4, 5]
        
    best_macro_f1_list = []
    best_micro_f1_list = []
    for seed in seeds:

        args.seed = seed

        print('--------- seed_{0} ---------'.format(args.seed))
        _, target_acc, macro_f1, micro_f1 = target_train(args)
            
        best_macro_f1_list.append(macro_f1)
        best_micro_f1_list.append(micro_f1)
    
    print('<------------------------------------------------------------>')
    print("{}=>{}:".format(args.source, args.target))
    print("ave_micro_f1:{:.4f}(+-{:.4f})".format(np.mean(best_micro_f1_list)*100, np.std(best_micro_f1_list)*100))
    print("ave_macro:{:.4f}(+-{:.4f})".format(np.mean(best_macro_f1_list)*100, np.std(best_macro_f1_list)*100))
    print('<------------------------------------------------------------>')



