import math
import sys

import h5py

from modules import fprint
from modules.models.transformer import TransformerEncoder

sys.path.append("/mnt/d/zhouLab/CodeRepository/deepEPI")
import numpy as np
import torch
from torch import nn,optim
# torch.optim.optimizer
from torch.utils.data import Dataset,DataLoader


from modules.datasets.datasets import EPIDatasets


def init_weights(model, method, mean=0,sd=1,a=0,b=1,gain=math.sqrt(2),verbose=True):
    # if diff:
    #     if isinstance(model,nn.Linear):
    #         nn.init.(model.weight)
    if method.lower() == "zeros":
        if isinstance(model, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d)):
            nn.init.zeros_(model.weight)
            nn.init.constant_(model.bias,0)
    elif method.lower() == "normal":
        if isinstance(model, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d)):
            nn.init.normal_(model.weight,mean=mean,std=sd)
            nn.init.constant_(model.bias,0)
    elif method.lower() == "uniform":
        if isinstance(model, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d)):
            nn.init.uniform_(model.weight,a=a,b=b)
            nn.init.constant_(model.bias,0)
    elif method.lower() == "xavier_normal":
        if isinstance(model, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d)):
            nn.init.xavier_normal_(model.weight,gain=gain)
            nn.init.constant_(model.bias, 0)
    elif method.lower() == "xavier_uniform":
        if isinstance(model, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d)):
            nn.init.xavier_uniform_(model.weight,gain=gain)
            nn.init.constant_(model.bias, 0)
    elif method.lower() == "kaiming_normal":
        if isinstance(model, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d)):
            nn.init.kaiming_normal_(model.weight,a=a)
            nn.init.constant_(model.bias, 0)
    elif method.lower() == "kaiming_uniform":
        if isinstance(model, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d)):
            nn.init.kaiming_uniform_(model.weight,a=a)
            nn.init.constant_(model.bias, 0)
    else:
        if verbose:
            fprint("WARNING","There is not this initialization method: {}! So the parameters will be initialized with default method in Pytorch!!".format(method))
        pass

class CustomConcatLayer(nn.Module):
    def __init__(self, features1, features2):
        super(CustomConcatLayer, self).__init__()
        self.features1 = features1
        self.features2 = features2

    def forward(self, x1, x2):
        # 在第二维度上连接 x1 和 x2
        concatenated = torch.cat([x1, x2], dim=2)
        return concatenated


class SimpleCNN(nn.Module):

    """
    This simpleCNN is based on the fundamental architecture from this literature (Zhuang et al., 2019. Bioinformatics).
    """
    def __init__(self, encode_method="onehot",concat_reverse=False,enhancer_len=4e3,promoter_len=2e3,
                 # vocab_size=None,embedding_dim=None,embedding_matrix=None,
                 init_method="", a=0, b=1, mean=0, sd=1, gain=1,verbose=True):
        # super.__init__()
        super().__init__()
        self.encode_method = encode_method
        self.concat_reverse = concat_reverse

        self.enhancer_len = int(enhancer_len)
        self.promoter_len = int(promoter_len)
        # self.device = torch.device(device)
        in_channels = 4
        if encode_method.lower() == "onehot":

            pass
        elif encode_method.lower() == "dna2vec":
            in_channels = 100
            # self.embedding_enhancer = nn.Embedding(vocab_size,embedding_dim,_weight=embedding_matrix)
            # self.embedding_promoter = nn.Embedding(vocab_size, embedding_dim, _weight=embedding_matrix)
            pass

        if self.concat_reverse:
            in_channels = in_channels * 2

        self.enhancer_conv1 = nn.Sequential(nn.Conv1d(in_channels=in_channels,out_channels=300,kernel_size=40,stride=1, padding="same"),nn.ReLU())
        self.enhancer_maxpool1 = nn.Sequential(nn.MaxPool1d(kernel_size=20,stride=20,padding=0))

        self.promoter_conv1 = nn.Sequential(nn.Conv1d(in_channels=in_channels, out_channels=300, kernel_size=40, stride=1, padding="same"),nn.ReLU())
        self.promoter_maxpool1 = nn.Sequential(nn.MaxPool1d(kernel_size=20, stride=20, padding=0))
        self.merged_layer = CustomConcatLayer(300, 300)
        # self.merged_layer =

        ##
        self.flatten = nn.Flatten()
        # self.flatten
        in_features = int(self.enhancer_len / 20 + self.promoter_len / 20) * 300
        self.linear1 = nn.Sequential(nn.Linear(in_features=in_features, out_features=800), nn.ReLU())
        self.dropout = nn.Dropout(p=0.2)
        self.output_layer = nn.Sequential(nn.Linear(800, 2), nn.Softmax())
        # self.output_layer.
        for m in self.modules():
            # m.
            # m.parameters().shape
            init_weights(m, init_method, mean, sd, a, b, gain,verbose)
        # self.merged_layer()
        # self.merge_layer = self
        # self.conv1 = nn.Conv2d(in_channels=1,out_channels=1,kernel_size=(40,4),stride=,padding=)


    def forward(self,x):
        # x = self
        en_rows = list(range(self.enhancer_len))
        pr_rows = list(range(self.enhancer_len, self.enhancer_len + self.promoter_len))
        # print(en_rows)
        if self.encode_method.lower() == "onehot":
            en_cols = range(1, 5)
            pr_cols = range(1, 5)
            if self.concat_reverse:
                # en_cols = list(range(1,5)) + list(range(10,14))
                # pr_cols = list(range(5,9)) + list(range(15,20))

                en_cols = list(range(1, 5)) + list(range(6, 10))
                pr_cols = list(range(1, 5)) + list(range(6, 10))

            enhancer_x = self.enhancer_conv1(torch.permute(x[:, en_rows, :][:,:,en_cols], (0, 2, 1)))
            promoter_x = self.promoter_conv1(torch.permute(x[:, pr_rows, :][:,:,pr_cols], (0, 2, 1)))
        else:
            en_cols = list(range(0, 100))
            pr_cols = list(range(0, 100))
            if self.concat_reverse:
                # en_cols = list(range(0, 100)) + list(range(200, 300))
                # # print()
                # pr_cols = list(range(100, 200)) + list(range(300, 400))

                en_cols = list(range(0, 200))
                # print()
                pr_cols = list(range(0, 200))
            # print(en_cols)
            # emb_enhancer = self.embedding_enhancer(torch.permute(x[:, :, en_cols], (0, 2, 1)))
            enhancer_x = self.enhancer_conv1(torch.permute(x[:, en_rows, :], (0, 2, 1)))
            # emb_promoter = self.embedding_promoter(torch.permute(x[:, :, pr_cols], (0, 2, 1)))
            promoter_x = self.promoter_conv1(torch.permute(x[:, pr_rows, :], (0, 2, 1)))
            pass

        enhancer_x = self.enhancer_maxpool1(enhancer_x)
        promoter_x = self.promoter_maxpool1(promoter_x)
        # print(promoter_x.shape)
        x = self.merged_layer(enhancer_x,promoter_x)
        # print(x.shape)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.output_layer(x)

        return x
    # def get_params(self):
    #     return self.parameters()
    #     pass
    #
    # def fit(self, train_loader, valid_loader, epochs, criterion, optimizer):
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     self.to(device)
    #
    #     for epoch in range(epochs):
    #         self.train()
    #         for inputs, labels in train_loader:
    #             inputs = inputs.to(device)
    #             labels = labels.to(device)
    #
    #             optimizer.zero_grad()
    #
    #             outputs = self.forward(inputs)  # 前向传播
    #
    #             loss = criterion(outputs, labels)
    #             loss.backward()
    #
    #             optimizer.step()
    #
    #         self.eval()
    #         with torch.no_grad():
    #             total_loss = 0.0
    #             total_correct = 0
    #             total_samples = 0
    #             for inputs, labels in valid_loader:
    #                 inputs = inputs.to(device)
    #                 labels = labels.to(device)
    #
    #                 outputs = self.forward(inputs)
    #
    #                 loss = criterion(outputs, labels)
    #
    #                 total_loss += loss.item() * inputs.size(0)
    #                 _, predicted = torch.max(outputs.data, 1)
    #                 total_correct += (predicted == labels).sum().item()
    #                 total_samples += inputs.size(0)
    #
    #             avg_loss = total_loss / total_samples
    #             accuracy = total_correct / total_samples
    #
    #             print(f"Epoch [{epoch + 1}/{epochs}], Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")


## EPI-Mind model ()
class EPIMind(nn.Module):
    def __init__(self, concat_reverse=False, enhancer_len=4e3, promoter_len=2e3,
                 padding_idx=0,
                 vocab_size=None,
                 embedding_dim=100,
                 num_heads=8, num_layers=4, num_hiddens=72, ffn_num_hiddens=256,
                 init_method="", a=0, b=1, mean=0, sd=1, gain=1, verbose=True, **kwargs):
        super(EPIMind, self).__init__()
        self.enhancer_len = int(enhancer_len)
        self.promoter_len = int(promoter_len)
        self.concat_reverse = concat_reverse
        # self.embedding_enhancer = nn.Embedding(vocab_size, embedding_dim, _weight=kwargs["pretrained_dna2vec"],padding_idx=padding_idx)
        # self.embedding_promoter = nn.Embedding(vocab_size, embedding_dim, _weight=kwargs["pretrained_dna2vec"],padding_idx=padding_idx)
        ##
        in_channels = embedding_dim
        # self.embedding_enhancer = nn.Embedding(vocab_size,embedding_dim,_weight=embedding_matrix)
        # self.embedding_promoter = nn.Embedding(vocab_size, embedding_dim, _weight=embedding_matrix)
        pass

        if self.concat_reverse:
            in_channels = in_channels * 2
        self.enhancer_conv = nn.Sequential(nn.Conv1d(in_channels=in_channels, out_channels=num_hiddens,  # 64
                                                     kernel_size=36,  # 40
                                                     padding="valid"), nn.ReLU())
        self.promoter_conv = nn.Sequential(nn.Conv1d(in_channels=in_channels, out_channels=num_hiddens,  # 64
                                                     kernel_size=36,  # 40
                                                     padding="valid"), nn.ReLU())

        self.enhancer_maxpool = nn.Sequential(nn.MaxPool1d(kernel_size=20, stride=20))
        self.promoter_maxpool = nn.Sequential(nn.MaxPool1d(kernel_size=20, stride=20))
        ##
        self.kqv_size = num_hiddens
        self.transformer_encoder_enhancer = TransformerEncoder(max_len=self.enhancer_len,key_size=self.kqv_size, query_size=self.kqv_size,
                                                               value_size=self.kqv_size,
                                                               d_model=num_hiddens, norm_shape=[198, num_hiddens],
                                                               ffn_num_input=num_hiddens,
                                                               ffn_num_hiddens=ffn_num_hiddens,
                                                               num_heads=num_heads, num_layers=num_layers, dropout=0.1,
                                                               use_bias=False)
        self.transformer_encoder_promoter = TransformerEncoder(max_len=self.promoter_len,key_size=self.kqv_size, query_size=self.kqv_size,
                                                               value_size=self.kqv_size,
                                                               d_model=num_hiddens, norm_shape=[98, num_hiddens],
                                                               ffn_num_input=num_hiddens,
                                                               ffn_num_hiddens=ffn_num_hiddens,
                                                               num_heads=num_heads, num_layers=num_layers, dropout=0.1,
                                                               use_bias=False)

        self.global_maxpool1 = nn.AdaptiveMaxPool1d(1)
        self.global_maxpool2 = nn.AdaptiveMaxPool1d(1)

        self.flatten = nn.Flatten()
        self.dense = nn.Sequential(nn.Linear(in_features=num_hiddens*2, out_features=50),nn.ReLU())
        self.dense1 = nn.Sequential(nn.Linear(in_features=50, out_features=2), nn.Softmax())

        for m in self.modules():
            # m.
            # m.parameters().shape
            init_weights(m, init_method, mean, sd, a, b, gain, verbose)
        pass

    def forward(self, X):
        # enhancer = X[]
        en_rows = list(range(self.enhancer_len))
        pr_rows = list(range(self.enhancer_len, self.enhancer_len + self.promoter_len))
        en_cols = list(range(0, 100))
        pr_cols = list(range(0, 100))
        if self.concat_reverse:
            # en_cols = list(range(0, 100)) + list(range(200, 300))
            # # print()
            # pr_cols = list(range(100, 200)) + list(range(300, 400))

            en_cols = list(range(0, 200))
            # print()
            pr_cols = list(range(0, 200))
        # enhancer = self.embedding_enhancer(enhancer)
        # promoter = self.embedding_promoter(promoter)
        ##
        enhancer = self.enhancer_conv(torch.permute(X[:, en_rows, :][:, :, en_cols], (0, 2, 1)))
        promoter = self.promoter_conv(torch.permute(X[:, pr_rows, :][:, :, pr_cols], (0, 2, 1)))

        enhancer = self.enhancer_maxpool(enhancer)
        promoter = self.promoter_maxpool(promoter)

        ## Transformer
        enhancer = self.transformer_encoder_enhancer(enhancer)
        promoter = self.transformer_encoder_promoter(promoter)

        enhancer = torch.permute(enhancer, dims=(0, 2, 1))
        promoter = torch.permute(promoter, dims=(0, 2, 1))
        enhancer_maxpool = self.global_maxpool1(enhancer)
        promoter_maxpool = self.global_maxpool2(promoter)

        # merge
        # print(enhancer_maxpool.shape)
        # print(promoter_maxpool.shape)
        merge = torch.concat([enhancer_maxpool * promoter_maxpool,
                              torch.abs(enhancer_maxpool - promoter_maxpool),
                              ], -1)
        merge = torch.permute(merge,dims=(0,2,1))
        # print("merge.shape:{}".format(merge.shape))
        merge2 = self.dense(self.flatten(merge))
        preds = self.dense1(merge2)
        return preds
        pass

# class SimpleCN
def cross_entropy_loss(true_labels, predicts):
    loss = - true_labels * torch.log(predicts)
    loss = torch.mean(loss)
    return loss

def train(model,data_loader,device):
    optimizer = torch.optim.SGD(model.parameters(),lr=0.1,momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    for batch, (data, label) in enumerate(data_loader):
        # print(data.shape)
        # print(labels.shape)
        data = data.to(device)
        label = label.to(device)
        label = torch.flatten(label)
        # print(label)
        # label = label.long()
        # print("label: {}".format(label.shape))
        # enhancer_X = torch.permute(data[:, :, 1:5],(0,2,1))
        #
        # promoter_X = torch.permute(data[:, :, 5:9],(0,2,1))
        optimizer.zero_grad()
        # loss = cross_entropy_loss(model(enhancer_X,promoter_X),labels).backward()
        # print(label.shape)
        output = model(data)
        # print("output: {}".format(output.shape))
        loss = criterion(output, label)
        print("batch: {}, loss: {}".format(batch,loss.data))
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    model = SimpleCNN()
    device = "gpu"
    model.to(torch.device(device))
    # torch.backends.cudnn.enable = True
    # torch.backends.cudnn.benchmark = True
    with h5py.File("../../output/prep_datasets_onehot_train.h5","r") as f:
        # print(np.array(f["X"]).shape)
        # print(np.array(f["y"]).shape)
        # dataset = EPIDatasets(data=f["X"],label=f["y"])
        dataset = EPIDatasets(cache_file_path="../../output/prep_datasets_onehot_train.h5")
        data_loader = DataLoader(dataset, batch_size=8,shuffle=False)
        # print(len(data_loader))
        # print(data_loader.__iter__())
        train(model, data_loader, device)
    # a = torch.tensor(np.ones((1, 4, 3000)),dtype=torch.float,device="cuda")
    # b = torch.tensor(np.ones((1, 4, 2000)),dtype=torch.float,device="cuda")
    # c = model(a,b)
    # print(c.shape)
    # print(c)


