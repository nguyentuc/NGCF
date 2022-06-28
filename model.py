'''
Created on July 24, 2021

@author: Tuc Nguyen Van (nguyentuc1003@gmail.com)
'''

from time import time
import torch
import torch.nn as nn
import torch.nn.functional as F

CUDA_LAUNCH_BLOCKING=1

class NGCF(nn.Module):
    def __init__(self, n_user, n_item, norm_adj, args):
        super(NGCF, self).__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.device = args.device
        self.emb_size = args.embed_size
        self.batch_size = args.batch_size
        self.node_dropout = args.node_dropout[0]
        self.mess_dropout = args.mess_dropout
        self.batch_size = args.batch_size

        self.norm_adj = norm_adj

        self.layers = eval(args.layer_size)
        self.decay = eval(args.regs)[0]

        """
        *********************************************************
        Init the weight of user-item.
        """
        self.embedding_dict, self.weight_dict = self.init_weight()

        """
        *********************************************************
        Get sparse adj.
        """
        t0 = time()
        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.norm_adj).to(self.device)
        print("Time get sparse norm adj:",time() - t0)

    def init_weight(self):
        # xavier init
        initializer = nn.init.xavier_uniform_

        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.n_user,
                                                 self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.n_item,
                                                 self.emb_size)))
        })

        weight_dict = nn.ParameterDict()
        layers = [self.emb_size] + self.layers # [64, 64, 64, 64]
        
        for k in range(len(self.layers)):
            weight_dict.update({'W_gc_%d'%k: nn.Parameter(initializer(torch.empty(layers[k],
                                                                      layers[k+1])))})
            weight_dict.update({'b_gc_%d'%k: nn.Parameter(initializer(torch.empty(1, layers[k+1])))})

            weight_dict.update({'W_bi_%d'%k: nn.Parameter(initializer(torch.empty(layers[k],
                                                                      layers[k+1])))})
            weight_dict.update({'b_bi_%d'%k: nn.Parameter(initializer(torch.empty(1, layers[k+1])))})

        return embedding_dict, weight_dict

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def sparse_dropout(self, x, rate, noise_shape):
        random_tensor = 1 - rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

        maxi = nn.LogSigmoid()(pos_scores - neg_scores)

        mf_loss = -1 * torch.mean(maxi)

        # cul regularizer
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.decay * regularizer / self.batch_size

        return mf_loss + emb_loss, mf_loss, emb_loss

    def rating(self, u_g_embeddings, pos_i_g_embeddings):
        return torch.matmul(u_g_embeddings, pos_i_g_embeddings.t())

    def forward(self, users, pos_items, neg_items, drop_flag=True):
        
        A_hat = self.sparse_dropout(self.sparse_norm_adj,
                                    self.node_dropout,
                                    self.sparse_norm_adj._nnz()) if drop_flag else self.sparse_norm_adj

        # sum messages of neighbors
        embeddings = torch.cat([self.embedding_dict['user_emb'],
                                    self.embedding_dict['item_emb']], 0)

        # get smaller side_embedding (message from neighbors)
        side_embeddings = torch.sparse.mm(A_hat, embeddings)
        u_embedding = side_embeddings[:self.n_user, :]
        i_embedding = side_embeddings[self.n_user: , :]
        u_side_embeddings = u_embedding[users, :]

        items = list(pos_items) + list(neg_items)
        pos_i_embedding = i_embedding[pos_items, :]
        neg_i_embedding = i_embedding[neg_items, :]
        i_side_embeddings = torch.cat([pos_i_embedding, neg_i_embedding], 0)

        small_side_embeddings = torch.cat([u_side_embeddings,i_side_embeddings], 0) 

        # get smaller ego_embedding
        small_ego_embeddings = torch.cat([self.embedding_dict['user_emb'][users , :],
                                    self.embedding_dict['item_emb'][items, :]], 0)

        all_embeddings = [small_ego_embeddings]

        
        for k in range(len(self.layers)):
            
            # transformed sum messages of neighbors with W and bias: [70839, 64] x [64, 64] + [1, 64]
            #  sum_embeddings = self.gcn_layers[k](side_embeddings)
            sum_embeddings = torch.matmul(small_side_embeddings, self.weight_dict['W_gc_%d' % k]) \
                                             + self.weight_dict['b_gc_%d' % k]

            # bi messages of neighbors.
            bi_embeddings = torch.mul(small_ego_embeddings, small_side_embeddings)
           
            # transformed bi messages of neighbors.
            bi_embeddings = torch.matmul(bi_embeddings, self.weight_dict['W_bi_%d' % k]) \
                                            + self.weight_dict['b_bi_%d' % k]
        
            # non-linear activation.
            small_ego_embeddings = nn.LeakyReLU(negative_slope=0.2)(sum_embeddings + bi_embeddings)

            # message dropout.
            small_ego_embeddings = nn.Dropout(self.mess_dropout[k])(small_ego_embeddings)

            # normalize the distribution of embeddings.
            norm_embeddings = F.normalize(small_ego_embeddings, p=2, dim=1)

            all_embeddings += [norm_embeddings] # initialization embedding and after process embedding

        all_embeddings = torch.cat(all_embeddings, 1)

        u_g_embeddings = all_embeddings[:len(users), :]

        i_g_embeddings = all_embeddings[len(users):, :]
        pos_i_g_embeddings = i_g_embeddings[:len(pos_items), :]
        neg_i_g_embeddings = i_g_embeddings[len(neg_items):, :]
    
        return u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings
