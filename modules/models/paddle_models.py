import numpy as np
import paddle
import math
from paddle import nn,optimizer
# paddle.optim.optimizer
from paddle.io import Dataset, DataLoader

from modules import fprint


def init_weights(model, method, mean=0,sd=1,a=0,b=1,gain=math.sqrt(2),verbose=True):
    # if diff:
    #     if isinstance(model,nn.Linear):
    #         nn.init.(model.weight)
    if method.lower() == "zeros":
        if isinstance(model, (paddle.nn.Linear, paddle.nn.Conv1D, paddle.nn.Conv2D)):
            nn.init.zeros_(model.weight)
            nn.init.constant_(model.bias,0)
    elif method.lower() == "normal":
        if isinstance(model, (paddle.nn.Linear, paddle.nn.Conv1D, paddle.nn.Conv2D)):
            nn.init.normal_(model.weight,mean=mean,std=sd)
            nn.init.constant_(model.bias,0)
    elif method.lower() == "uniform":
        if isinstance(model, (paddle.nn.Linear, paddle.nn.Conv1D, paddle.nn.Conv2D)):
            nn.init.uniform_(model.weight,a=a,b=b)
            nn.init.constant_(model.bias,0)
    elif method.lower() == "xavier_normal":
        if isinstance(model, (paddle.nn.Linear, paddle.nn.Conv1D, paddle.nn.Conv2D)):
            nn.init.xavier_normal_(model.weight,gain=gain)
            nn.init.constant_(model.bias, 0)
    elif method.lower() == "xavier_uniform":
        if isinstance(model, (paddle.nn.Linear, paddle.nn.Conv1D, paddle.nn.Conv2D)):
            nn.init.xavier_uniform_(model.weight,gain=gain)
            nn.init.constant_(model.bias, 0)
    elif method.lower() == "kaiming_normal":
        if isinstance(model, (paddle.nn.Linear, paddle.nn.Conv1D, paddle.nn.Conv2D)):
            nn.init.kaiming_normal_(model.weight,a=a)
            nn.init.constant_(model.bias, 0)
    elif method.lower() == "kaiming_uniform":
        if isinstance(model, (paddle.nn.Linear, paddle.nn.Conv1D, paddle.nn.Conv2D)):
            nn.init.kaiming_uniform_(model.weight,a=a)
            nn.init.constant_(model.bias, 0)
    else:
        if verbose:
            fprint("WARNING","There is not this initialization method: {}! So the parameters will be initialized with default method in paddle!!".format(method))
        pass

class CustomConcatLayer(nn.Layer):
    def __init__(self, features1, features2):
        super(CustomConcatLayer, self).__init__()
        self.features1 = features1
        self.features2 = features2

    def forward(self, x1, x2):
        # 在第二维度上连接 x1 和 x2
        concatenated = paddle.concat([x1, x2], axis=2)
        return concatenated

class SimpleCNN(nn.Layer):
    """
        This simpleCNN is based on the fundamental architecture from this literature (Zhuang et al., 2019. Bioinformatics).
        """

    def __init__(self, encode_method="onehot", concat_reverse=False, enhancer_len=4e3, promoter_len=2e3,
                 # vocab_size=None,embedding_dim=None,embedding_matrix=None,
                 init_method="", a=0, b=1, mean=0, sd=1, gain=1, verbose=True):
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

        self.enhancer_conv1 = nn.Sequential(
            nn.Conv1D(in_channels=in_channels, out_channels=300, kernel_size=40, stride=1, padding="same"), nn.ReLU())
        self.enhancer_maxpool1 = nn.Sequential(nn.MaxPool1D(kernel_size=20, stride=20, padding=0), nn.ReLU())

        self.promoter_conv1 = nn.Sequential(
            nn.Conv1D(in_channels=in_channels, out_channels=300, kernel_size=40, stride=1, padding="same"), nn.ReLU())
        self.promoter_maxpool1 = nn.Sequential(nn.MaxPool1D(kernel_size=20, stride=20, padding=0), nn.ReLU())
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
        for m in self.sublayers():
            # m.
            # m.parameters().shape
            init_weights(m, init_method, mean, sd, a, b, gain, verbose)
        # self.merged_layer()
        # self.merge_layer = self
        # self.conv1 = nn.Conv2d(in_channels=1,out_channels=1,kernel_size=(40,4),stride=,padding=)

    def forward(self, x):
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

            enhancer_x = self.enhancer_conv1(paddle.transpose(x[:, en_rows, :][:, :, en_cols], perm=[0,2,1]))
            promoter_x = self.promoter_conv1(paddle.transpose(x[:, pr_rows, :][:, :, pr_cols], perm=[0,2,1]))
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
            # emb_enhancer = self.embedding_enhancer(torch.transpose(x[:, :, en_cols], [0,2,1]))

            enhancer_x = self.enhancer_conv1(paddle.transpose(
                paddle.to_tensor(np.array(x)[:, en_rows, :][:, :, en_cols]), perm=[0,2,1]))
            # emb_promoter = self.embedding_promoter(torch.transpose(x[:, :, pr_cols], [0,2,1]))
            promoter_x = self.promoter_conv1(paddle.transpose(
                paddle.to_tensor(np.array(x)[:, pr_rows, :][:, :, pr_cols]), perm=[0,2,1]))
            pass

        enhancer_x = self.enhancer_maxpool1(enhancer_x)
        promoter_x = self.promoter_maxpool1(promoter_x)
        # print(promoter_x.shape)
        x = self.merged_layer(enhancer_x, promoter_x)
        # print(x.shape)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.output_layer(x)

        return x