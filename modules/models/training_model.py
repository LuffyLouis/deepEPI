import sys
import time
from datetime import datetime
import os
import re

import numpy as np
import thop
import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, precision_recall_curve, auc, f1_score
# from scipy import interp
# from sklearn.model_selection import GridSearchCV
# from skorch import NeuralNetClassifier
from bayes_opt import BayesianOptimization
from torch.utils.data import Dataset, DataLoader
from torch import nn
from tensorboardX import SummaryWriter
# from d2l import torch as d2l
from modules import fprint, check_and_create_dir, format_dict_to_filename, read_config_file, Timer
from modules.datasets.datasets import EPIDatasets
from modules.evaluation.evaluation import ModelEvaluator, PerformanceEvaluator
from modules.models.models import SimpleCNN, init_weights, EPIMind
#
# torch.backends.cudnn.enabled = False
#
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = False
# torch.backends.cudnn.allow_tf32 = True
from modules.models.tools import accuracy

## DDP (DistributedDataParallel)
import torch.multiprocessing as mp
import torch.distributed as dist



def train_each(epoch, log_writer, log_mode, model, data_loader,test_loader, optimizer, device, verbose):
    # optimizer = torch.optim.SGD(model.parameters(),lr=0.1,momentum=0.9)
    # 初始化logWriter

    loss = None
    model.train()
    criterion = nn.CrossEntropyLoss()


    for batch, (data, label) in enumerate(data_loader):
        if device:
            data = data.to(device)
            label = label.to(device)
        else:
            data = data.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
        label = torch.flatten(label)

        # print(data.shape)
        # print(label)
        # label = label.long()
        # print("label: {}".format(label.shape))
        # enhancer_X = torch.permute(data[:, :, 1:5],(0,2,1))
        # promoter_X = torch.permute(data[:, :, 5:9],(0,2,1))
        if epoch == 0 and batch == 0:
            # training_model_perf.to(device)
            # print(data.shape)
            start = time.time()
            # training_model_perf.eval()
            # model = thop.replace_count_softmax(training_model_perf)

            performanceEvaluator = PerformanceEvaluator(model, data)

            # model = EPIMind()
            # performanceEvaluator = PerformanceEvaluator(model, data)
            end = time.time()
            macs, params = performanceEvaluator.clever_format(
                [performanceEvaluator.macs / (end - start), performanceEvaluator.params])
            print("-----------------------------")
            print("FLOPS: {}FLOPS".format(macs))
            print("Params: {}".format(params))
            print("-----------------------------")
            if log_writer:
                log_writer.add_graph(model, data)

        optimizer.zero_grad()
        # loss = cross_entropy_loss(model(enhancer_X,promoter_X),labels).backward()
        # print(label.shape)
        # print("data: {}".format(data.shape))
        output = model(data)
        # print("output: {}".format(output.shape))
        loss = criterion(output, label)
        train_acc = accuracy(output, label) / label.numel()
        # model.eval()
        # test_loss = 0
        # correct = 0
        if test_loader:
            # test_len = len(test_loader)
            with torch.no_grad():
                for index,(data, label) in enumerate(test_loader):

                    if index == 0:
                        if device:
                            data = data.to(device)
                            label = label.to(device)
                        else:
                            data = data.cuda(non_blocking=True)
                            label = label.cuda(non_blocking=True)
                        label = torch.flatten(label)
                        # print(label)
                        # label = label.long()
                        # print("label: {}".format(label.shape))
                        # enhancer_X = torch.permute(data[:, :, 1:5], (0, 2, 1))
                        #
                        # promoter_X = torch.permute(data[:, :, 5:9], (0, 2, 1))
                        output = model(data)
                        # print("output: {}".format(output))
                        test_loss = criterion(output, label).item()
                        pred = output.argmax(dim=1, keepdim=True)
                        # test_accuracy = pred.eq(label.view_as(pred)).sum().item()
                        test_accuracy = accuracy(output, label) / label.numel()
                    # print("index:{}".format(index))
                    # print("correct:{}".format(correct))

                # test_loss /= label.numel()
                # test_accuracy = correct / label.numel()
            if verbose:
                print("Epoch: {}, batch: {}, loss: {:4f}, train acc: {:2f}, test acc: {:2f}".format(epoch, batch,loss.data,train_acc,test_accuracy))
        else:
            if verbose:
                print("Epoch: {}, batch: {}, loss: {:4f}, train acc: {:2f}".format(epoch, batch,loss.data,train_acc))
        loss.backward()
        optimizer.step()
    if log_writer:
        if log_mode.lower() == "train":
            log_writer.add_scalar("train loss", loss.data, epoch)
            log_writer.add_scalar("train accuracy", train_acc, epoch)
            if test_loader:
                log_writer.add_scalar("test accuracy", test_accuracy, epoch)
        else:
            pass
        # log_writer.add
    # log_writer.add_scalar("test accuracy", test_acc, epoch)


def kfold_cross_validation(encode_method, concat_reverse,workers,timer,
                           kfold,  log_writer, epoches, batch_size, model, init_method, optimizer,raw_dataset, metrics,save_path,
                            enhancer_len,promoter_len,heads,num_layers,num_hiddens,ffn_num_hiddens,
                           device,verbose):
    # optimizer = torch.optim.SGD(model.parameters(),lr=0.1,momentum=0.9)
    # model.train()
    criterion = nn.CrossEntropyLoss()
    # roc_curve_data_list = []
    # fpr_data,tpr_data = None,None
    y_true_list = []
    y_pred_list = []
    # 进行K折交叉验证
    # k = len(kfold.split(raw_dataset))
    acc_list = []
    training_model = None
    print("{}, {}, {}".format(encode_method,concat_reverse,init_method))
    for fold, (train_ids, val_ids) in enumerate(kfold.split(raw_dataset)):
        print(f"Fold {fold + 1}")
        if model == "simpleCNN":
            training_model = SimpleCNN(encode_method=encode_method,concat_reverse=concat_reverse,init_method=init_method,verbose=verbose)
        elif model == "EPIMind":
            training_model = EPIMind(concat_reverse=concat_reverse,init_method=init_method,
                                     enhancer_len=enhancer_len, promoter_len=promoter_len,
                                     num_heads=heads, num_layers=num_layers, num_hiddens=num_hiddens, ffn_num_hiddens=ffn_num_hiddens,verbose=verbose)
        training_model.to(device)
        # print("{}, {}".format(len(train_ids),len(val_ids)))
        # 获取当前折的训练数据和验证数据
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

        # print(val_subsampler.indices)
        train_loader = torch.utils.data.DataLoader(raw_dataset, batch_size=batch_size, sampler=train_subsampler,
                                                   num_workers=workers)
        val_loader = torch.utils.data.DataLoader(raw_dataset, batch_size=batch_size, sampler=val_subsampler,
                                                 num_workers=workers)

        # print("train_loader:{}".format(len(train_loader)))
        # print("val_loader:{}".format(len(val_loader)))
        val_len = len(val_ids)
        # model.apply()
        # 训练模型
        for epoch in range(epoches):
            timer.start()

            ##
            train_each(epoch, log_writer, "", training_model, train_loader,None, optimizer, device, verbose)
            ##
            training_model.eval()
            val_loss = 0
            correct = 0
            # val_
            # val_len = len(val_loader)
            with torch.no_grad():
                for index,(data, label) in enumerate(val_loader):
                    data = data.to(device)
                    label = label.to(device)
                    label = torch.flatten(label)
                    # print(label)
                    # label = label.long()
                    # print("label: {}".format(label.shape))
                    # enhancer_X = torch.permute(data[:, :, 1:5], (0, 2, 1))
                    #
                    # promoter_X = torch.permute(data[:, :, 5:9], (0, 2, 1))
                    output = training_model(data)
                    # print("output: {}".format(output))
                    val_loss += criterion(output, label).item()
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(label.view_as(pred)).sum().item()
                    # print("index:{}".format(index))
                    # print("correct:{}".format(correct))
            # print(correct)
            # print(val_len)
            # print(len(val_loader))
            val_loss /= val_len
            accuracy = 100. * correct / val_len
            acc_list.append(correct / val_len)
            if epoch == (epoches - 1):
                # print("sdsds")
                correct_class_probabilities = output[torch.arange(len(label)), label]
                y_true_list.append(label)
                y_pred_list.append(correct_class_probabilities)
                # model_evaluator = ModelEvaluator(label,correct_class_probabilities)
                # fpr,tpr,_ = model_evaluator.calculate_roc_curve()
                # fpr,tpr = np.array(fpr).reshape(1,-1),np.array(tpr).reshape(1,-1)
                # if fold == 0:
                #     fpr_data = fpr
                #     tpr_data = tpr
                # else:
                #     fpr_data = np.r_[fpr_data,fpr]
                #     tpr_data = np.r_[tpr_data,fpr]
                # print(fpr_data)
                # print(model_evaluator.calculate_roc_curve())
                # model_evaluator.plot_roc_curve(save_path="./ROC.pdf")
            timer.stop()
            print(f"Epoch {epoch}: Val Loss: {val_loss:.4f}, Val Accuracy: {accuracy:.2f}%, Cost: {timer.elapsed_time():.3f}s/Epoch,"
                  f" Avg Cost: {timer.final_elapsed_time()/(epoch+1):.3f}s/Epoch")

    model_evaluator = ModelEvaluator(y_true_list=y_true_list, y_pred_list=y_pred_list)
    # model_evaluator.calculate_roc_curve()
    if save_path:
        model_evaluator.plot_mean_roc_curve(save_path)

    if metrics.lower() == "auc":
        mean_auc = model_evaluator.plot_mean_roc_curve()
        return mean_auc
        pass
    elif metrics.lower() == "acc":
        return np.nanmean(acc_list)
        pass
    elif metrics.lower() == "f1":
        return model_evaluator.calculate_f1_score()
    else:
        fprint("Error","Not supported metrics: {}".format(metrics))
        # sys.exit(-1)
        pass
    # mean_tpr = ModelEvaluator.calculate_mean_curve(tpr_data)
    # plt.figure()
    # plt.plot(mean_fpr, mean_tpr, color='darkorange', lw=2,
    #          label='Mean ROC (area = %0.2f)' % auc(mean_fpr, mean_tpr))
    # plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Mean Receiver Operating Characteristic (ROC)')
    # plt.legend(loc="lower right")
    # plt.savefig("./ROC.pdf")
    # plt.show()
    #
    # plt.figure()
    # plt.plot(mean_pr_curve[0], mean_pr_curve[1], color='darkorange', lw=2,
    #          label='Mean PR (area = %0.2f)' % auc(mean_pr_curve[0], mean_pr_curve[1]))
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.title('Mean Precision-Recall Curve')
    # plt.legend(loc="lower left")
    # plt.show()

def save_weights(net, saved_weights_dir='./cache/model', saved_weight_file='weights'):
    if not os.path.exists(saved_weights_dir):
        os.mkdir(saved_weights_dir)
    saved_weight_filename = saved_weight_file + "_" + datetime.now().strftime("%Y%m%d-%H%M%S")
    saved_weight_file_path = os.path.join(saved_weights_dir, saved_weight_filename)
    torch.save(net.state_dict(), saved_weight_file_path)


def load_weigths(saved_weights_dir='./cache/model', saved_weight_file='weights'):
    if os.path.exists(saved_weights_dir):
        latest_time = find_lateset_weight(saved_weights_dir)
        if latest_time:
            latest_file = saved_weight_file + "_" + datetime.strftime(latest_time, "%Y%m%d-%H%M%S")
            latest_file_path = os.path.join(saved_weights_dir, latest_file)
            print("Loading the latest file: {}".format(latest_file_path))
            latest_params = torch.load(latest_file_path)
            return latest_params
        else:
            return False
    else:
        return False

def find_lateset_weight(saved_weights_dir):
    files = os.listdir(saved_weights_dir)
    file_time_list = []
    for idx, file in enumerate(files):
        if 'weights' in file:
            file_time = datetime.strptime(file.split("_")[-1],"%Y%m%d-%H%M%S")
            file_time_list.append(file_time)
    if len(file_time_list) > 0:
        latest_time = max(file_time_list)
        return latest_time
    else:
        return False

## Initialization of a summary writer for logs in each epoch
def init_summary_writer(log_dir='./cache/log'):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir)
    return writer

def check_device():
    if torch.cuda.is_available():
        print("CUDA is available.")
    else:
        print("CUDA is not available.")
    # Get the total number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")

    # Iterate through each GPU and print detailed information
    for gpu_id in range(num_gpus):
        device = torch.device(f"cuda:{gpu_id}")  # Specify the GPU device
        print(f"Device: {device}")
        print(f"Name: {torch.cuda.get_device_name(device)}")
        print(f"Compute Capability: {torch.cuda.get_device_capability(device)}")
        print(f"Total Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.2f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
        print("----------------------")
    pass

#

def remove_null_key(hyperparameters):
    # new_dict = hyperparameters
    keys = []
    for (key, val) in hyperparameters.items():
        if len(val) == 0:
            keys.append(key)
    for key in keys:
        del hyperparameters[key]
    return hyperparameters

def generate_param_combinations(hyperparameters):
    # 检查是否还有未处理的超参数
    if len(hyperparameters) == 0:
        yield {}  # 返回空字典作为最基本的情况
    else:
        # 取出第一个超参数
        param_name, param_values = next(iter(hyperparameters.items()))
        remaining_hyperparameters = dict(hyperparameters)
        del remaining_hyperparameters[param_name]

        # 递归生成剩余超参数的组合
        for param_value in param_values:
            for param_combination in generate_param_combinations(remaining_hyperparameters):
                param_combination[param_name] = param_value
                yield param_combination


class TrainModel:
    def __init__(self, ddp_info, train_dataset_dir, train_dataset_pattern, train_dataset_file,test_dataset_file,workers,
                 model, encode_method,concat_reverse,
                 enhancer_len,promoter_len,heads,num_layers,num_hiddens,ffn_num_hiddens,
                 is_param_optim, param_optim_strategy,params_config,random_size,
                 init_points, n_iter,
                 k_fold,metrics,save_path,
                 init_method, epochs, lr, optimizer, batch_size, momentum,
                 weight_decay, nesterov, betas, eps, lr_decay, initial_accumulator_value, rho, alpha,
                 lambd, t0, max_iter, max_eval,
                 device, random_state,
                 rerun, save_param_dir, save_param_prefix, log_dir, save_model, verbose):


        self.ddp_mode = False
        ## Distributed training
        self.ddp_info = ddp_info
        self.nodes = self.ddp_info[0]
        self.gpus = self.ddp_info[1]
        self.nr = self.ddp_info[2]
        # self.local_rank = self.ddp_info[3]
        self.master = self.ddp_info[3]
        self.start_mode = self.ddp_info[4]
        self.world_size = self.nodes * self.gpus
        if self.master:
            fprint(msg="Initializing the DDP mode...")
            self.ddp_mode = True
        if self.start_mode:
            self.ddp_mode = True
        ##
        self.n_iter = n_iter
        self.init_points = init_points
        self.encode_method = None
        self.train_dataset_dir, self.train_dataset_file, self.model, self.is_param_optim, self.param_optim_strategy, self.k_fold, self.save_param_dir, self.save_model, self.verbose = \
            train_dataset_dir, train_dataset_file, model, is_param_optim, param_optim_strategy, k_fold, save_param_dir, save_model, verbose
        self.workers = workers
        self.encode_method,self.concat_reverse = encode_method,concat_reverse
        ##
        self.params_config = params_config
        self.random_size = random_size
        self.test_dataset_file = test_dataset_file
        self.rerun = rerun
        self.save_param_prefix = save_param_prefix
        self.train_dataset_pattern = train_dataset_pattern
        self.init_method = init_method
        self.training_model = None
        self.train_dataset_files = []

        ##
        self.enhancer_len = enhancer_len
        self.promoter_len = promoter_len
        self.num_heads = heads
        self.num_layers = num_layers
        self.num_hiddens = num_hiddens
        self.ffn_num_hiddens = ffn_num_hiddens
        ##
        self.epochs, self.lr, self.optimizer, self.batch_size = epochs,lr,optimizer,batch_size

        ##
        self.weight_decay, self.nesterov, \
        self.betas, self.eps, self.lr_decay, self.initial_accumulator_value,\
        self.rho, self.alpha, \
        self.lambd, self.t0, \
        self.max_iter, self.max_eval = weight_decay,nesterov,betas,eps,lr_decay,initial_accumulator_value,rho,alpha, \
                                       lambd,t0,max_iter,max_eval
        ##
        self.metrics = metrics
        self.save_path = save_path
        self.random_state = random_state
        self.device = device

        self.momentum = momentum
        self.log_dir = os.path.join(log_dir,self.model)
        ##
        self.timer = Timer()
        check_and_create_dir(self.log_dir)

        ##
        check_device()
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.init(self.init_method,self.optimizer,self.lr,self.momentum,self.weight_decay,self.nesterov,
             self.betas,self.eps,self.lr_decay,self.initial_accumulator_value,
             self.rho,self.alpha,
             self.lambd,self.t0,
             self.max_iter,self.max_eval)

    def init(self,init_method,optim,lr,momentum,weight_decay=0,nesterov=False,
             betas=(0.9, 0.999),eps=1e-08,lr_decay=0,initial_accumulator_value=0,
             rho=0.9,alpha=0.99,
             lambd=0.0001,t0=1000000.0,
             max_iter=20,max_eval=None):
        if self.model == "simpleCNN":
            self.training_model = SimpleCNN(encode_method=self.encode_method, concat_reverse=self.concat_reverse, init_method=init_method,verbose=self.verbose)
            # self.training_model_perf = SimpleCNN(encode_method=self.encode_method, concat_reverse=self.concat_reverse, init_method=init_method,verbose=self.verbose)
        elif self.model == "EPIMind":
            self.training_model = EPIMind(concat_reverse=self.concat_reverse, init_method=init_method,
                                          enhancer_len=self.enhancer_len, promoter_len=self.promoter_len,
                                          num_heads=self.num_heads, num_layers=self.num_layers,
                                          num_hiddens=self.num_hiddens, ffn_num_hiddens=self.ffn_num_hiddens,verbose=self.verbose
                                          )
            # self.training_model_perf = EPIMind(concat_reverse=self.concat_reverse, init_method=init_method,
            #                               enhancer_len=self.enhancer_len, promoter_len=self.promoter_len,
            #                               num_heads=self.num_heads, num_layers=self.num_layers,
            #                               num_hiddens=self.num_hiddens, ffn_num_hiddens=self.ffn_num_hiddens,
            #                               verbose=self.verbose
            #                               )
            pass


        if (not self.rerun) and self.save_param_dir is not None and self.save_param_prefix is not None:
            check_and_create_dir(self.save_param_dir)
            loading_res = load_weigths(self.save_param_dir, self.save_param_prefix)
            # print("loading_res: {}".format(loading_res))
            if loading_res:
                fprint(msg="Loading the latest weights from {} directory based on {} prefix...".format(self.save_param_dir, self.save_param_prefix))
                self.training_model.load_state_dict(loading_res, strict=False)
            else:
                pass
        else:
            fprint(msg="The model will rerun!!!")
            pass
        self.training_model.to(self.device)
        fprint(msg='training on '+self.device)
        # self.training_model.apply(init_weights)
        if optim.lower() == "sgd":
            self.optim = torch.optim.SGD(self.training_model.parameters(),lr=lr,momentum=momentum,weight_decay=weight_decay,nesterov=nesterov)
        elif optim.lower() == "adam":
            self.optim = torch.optim.Adam(self.training_model.parameters(), lr=lr, betas=betas,eps=eps,weight_decay=weight_decay)
        elif optim.lower() == "adagrad":
            self.optim = torch.optim.Adagrad(self.training_model.parameters(), lr=lr,lr_decay=lr_decay,weight_decay=weight_decay,
                                             initial_accumulator_value=initial_accumulator_value,eps=eps)
        elif optim.lower() == "adadelta":
            self.optim = torch.optim.Adadelta(self.training_model.parameters(), lr=lr,rho=rho,eps=eps,weight_decay=weight_decay)
        elif optim.lower() == "rmsprop":
            self.optim = torch.optim.RMSprop(self.training_model.parameters(), lr=lr,alpha=alpha,eps=eps,weight_decay=weight_decay,momentum=momentum)
        elif optim.lower() == "asgd":
            self.optim = torch.optim.ASGD(self.training_model.parameters(), lr=lr,lambd=lambd,alpha=alpha,t0=t0,weight_decay=weight_decay)
        elif optim.lower() == "lbfgs":
            self.optim = torch.optim.LBFGS(self.training_model.parameters(), lr=lr, max_iter=max_iter,max_eval=max_eval)
        else:
            fprint("WARNING","Not supported for {} optimization method, so the default SGD method was used!!!".format(optim))
            self.optim = torch.optim.SGD(self.training_model.parameters(), lr=lr, momentum=momentum)
        # return training_model, optim

    def run(self):
        self.timer.final_start()
        if self.train_dataset_file is None:
            for file in os.listdir(self.train_dataset_dir):
                if re.findall(self.train_dataset_pattern,file):
                    self.train_dataset_files.append(os.path.join(self.train_dataset_dir,file))
        current_log_dir = os.path.join(self.log_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
        log_writer = init_summary_writer(current_log_dir)
        # log_writer = None
        if len(self.train_dataset_files) > 0:
            if self.k_fold:
                fprint("WARNING","The multi-datasets with k-fold CV are only supported by general training mode now!!!")
                pass
            for epoch in range(self.epochs):
                for train_dataset_file in self.train_dataset_files:
                    train_dataset = EPIDatasets(cache_file_path=train_dataset_file)
                    train_dataloader = DataLoader(train_dataset,batch_size=self.batch_size,shuffle=False)
                    test_dataset = EPIDatasets(cache_file_path=self.test_dataset_file)
                    test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
                    self.timer.start()
                    train_each(epoch, log_writer, "Train", self.training_model, train_dataloader,test_dataloader, self.optim, self.device, self.verbose)
                    self.timer.stop()
                    self.timer.final_stop()
                    fprint(msg="Cost: {:3f}s/Epoch, Avg cost: {:.3f}s/Epoch".format(self.timer.elapsed_time(),self.timer.final_elapsed_time()/(epoch+1)))
        else:
            if self.ddp_mode:
                train_dataset = EPIDatasets(cache_file_path=self.train_dataset_file)


                test_dataset = EPIDatasets(cache_file_path=self.test_dataset_file)

                # ddp_train(gpu_rank,self.world_size,self.gpus,self.nr,self.timer,self.epochs,log_writer,
                #           self.training_model,train_dataset,test_dataset,self.optim,
                #             self.batch_size,self.workers,
                #             self.save_param_dir,self.save_param_prefix,self.verbose)
                fprint("WARNING","The ddp mode does not support the tensorboard now!!!")

                # os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'
                if self.start_mode.lower() == "mpspawn":
                    mp.spawn(ddp_train, nprocs=self.gpus,
                             args=(self.world_size, self.gpus, self.nr, self.timer, self.epochs, None,
                                     self.training_model,train_dataset,test_dataset,self.optim,
                                       self.batch_size,self.workers,
                                       self.save_param_dir,self.save_param_prefix,self.verbose,),
                                    daemon=True)
                else:
                    local_rank = int(os.environ["LOCAL_RANK"])
                    ddp_train(local_rank,self.world_size,self.gpus,self.nr,self.timer,self.epochs,None,
                              self.training_model,train_dataset,test_dataset,self.optim,
                                self.batch_size,self.workers,
                                self.save_param_dir,self.save_param_prefix,self.verbose)
                # mp.spawn(ddp_train, nprocs=self.gpus, args=(self.world_size,self.gpus,self.nr,self.timer,self.epochs,None,
                #           self.training_model,train_dataset,test_dataset,self.optim,
                #             self.batch_size,self.workers,
                #             self.save_param_dir,self.save_param_prefix,self.verbose,),
                #          daemon=True)
            else:
                train_dataset_file = self.train_dataset_file
                train_dataset = EPIDatasets(cache_file_path=train_dataset_file)
                train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False,num_workers=self.workers)
                test_dataset = EPIDatasets(cache_file_path=self.test_dataset_file)
                test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False,num_workers=self.workers)
                if self.k_fold is None:
                    for epoch in range(self.epochs):
                        self.timer.start()
                        train_each(epoch, log_writer, "Train", self.training_model, train_dataloader,test_dataloader, self.optim, self.device, self.verbose)
                        self.timer.stop()
                        self.timer.final_stop()
                        fprint(msg="Cost: {:.3f}s/Epoch, Avg cost: {:.3f}s/Epoch".format(self.timer.elapsed_time(),self.timer.final_elapsed_time()/(epoch+1)))
                    fprint(msg="Saving parameters of current model...")
                    save_weights(self.training_model, self.save_param_dir, self.save_param_prefix)
                    fprint(msg="Saved done!")
                else:
                    k_fold = KFold(n_splits=self.k_fold, shuffle=True, random_state=self.random_state)
                    # dataset = EPIDatasets(cache_file_path=self.train_dataset_file)
                    if self.is_param_optim and self.param_optim_strategy:
                        param_dict = read_config_file(self.params_config)
                        if self.param_optim_strategy.lower() == "grid":
                            metrics_data_list = []
                            metrics_list = []
                            # param_grid = {
                            #     # 'batch_size': [4],
                            #     # 'learning_rate': [0.001, 0.01],
                            #     'batch_size': [4, 8, 16, 32, 64],
                            #     'learning_rate': [0.001,0.01,0.1,0.5]
                            #     # 'max_epochs': [5, 10]
                            # }
                            param_combinations = list(generate_param_combinations(remove_null_key(param_dict)))
                            for index,param_dict in enumerate(param_combinations):
                                param_res_dict = {
                                    "params": param_dict,
                                    self.metrics: ""
                                }
                                if "max_epochs" in param_dict.keys():
                                    self.epochs = int(param_dict["max_epochs"])
                                elif "batch_size" in param_dict.keys():
                                    self.batch_size = int(param_dict["batch_size"])
                                elif "learning_rate" in param_dict.keys():
                                    self.lr = param_dict["learning_rate"]

                                print(param_res_dict)
                                self.init(self.init_method, self.optimizer, self.lr, self.momentum,self.weight_decay,self.nesterov,
                                     self.betas,self.eps,self.lr_decay,self.initial_accumulator_value,
                                     self.rho,self.alpha,
                                     self.lambd,self.t0,
                                     self.max_iter,self.max_eval)
                                save_path = "{}_{}.pdf".format(self.save_path,format_dict_to_filename(param_dict))
                                metrics = kfold_cross_validation(self.encode_method, self.concat_reverse,self.workers, self.timer, k_fold, None, self.epochs, self.batch_size,
                                                                 self.model, self.init_method, self.optim, train_dataset, self.metrics,
                                                                 save_path,
                                                                 self.enhancer_len, self.promoter_len, self.num_heads, self.num_layers,  self.num_hiddens,
                                                                 self.
                                                                 ffn_num_hiddens,
                                                                 self.device, self.verbose)
                                param_res_dict[self.metrics] = metrics
                                metrics_data_list.append(param_res_dict)
                                metrics_list.append(metrics)
                                ## add the metrics for each param optimization step
                                log_writer.add_scalar(self.metrics,metrics,index,display_name=format_dict_to_filename(param_dict))
                            index = metrics_list.index(np.nanmax(metrics_list))
                            fprint(msg="The final hyper-parameters is following:")
                            print(metrics_data_list[index])
                        elif self.param_optim_strategy.lower() == "random":
                            metrics_data_list = []
                            metrics_list = []
                            # param_grid = {
                            #     # 'batch_size': [4],
                            #     # 'learning_rate': [0.001, 0.01],
                            #     'batch_size': [4, 8, 16, 32, 64],
                            #     'learning_rate': [0.001, 0.01, 0.1, 0.5]
                            #     # 'max_epochs': [5, 10]
                            # }
                            param_combinations = list(generate_param_combinations(remove_null_key(param_dict)))
                            random_param_combinations = np.random.choice(param_combinations,self.random_size)

                            for index,param_dict in enumerate(random_param_combinations):
                                param_res_dict = {
                                    "params": param_dict,
                                    self.metrics: ""
                                }
                                if "max_epochs" in param_dict.keys():
                                    self.epochs = param_dict["max_epochs"]
                                elif "batch_size" in param_dict.keys():
                                    self.batch_size = param_dict["batch_size"]
                                elif "learning_rate" in param_dict.keys():
                                    self.lr = param_dict["learning_rate"]

                                self.init(self.init_method, self.optimizer, self.lr, self.momentum,self.weight_decay,self.nesterov,
                                     self.betas,self.eps,self.lr_decay,self.initial_accumulator_value,
                                     self.rho,self.alpha,
                                     self.lambd,self.t0,
                                     self.max_iter,self.max_eval)
                                save_path = "{}_{}.pdf".format(self.save_path,format_dict_to_filename(param_dict))
                                metrics = kfold_cross_validation(self.encode_method, self.concat_reverse,self.workers,self.timer, k_fold, None, self.epochs, self.batch_size,
                                                                 self.model, self.init_method, self.optim, train_dataset, self.metrics,
                                                                 save_path,
                                                                 self.enhancer_len, self.promoter_len, self.num_heads,
                                                                 self.num_layers, self.num_hiddens,
                                                                 self.
                                                                 ffn_num_hiddens,
                                                                 self.device, self.verbose)
                                param_res_dict[self.metrics] = metrics
                                metrics_data_list.append(param_res_dict)
                                metrics_list.append(metrics)
                                ## add the metrics for each param optimization step
                                log_writer.add_scalar(self.metrics,metrics,index,display_name=format_dict_to_filename(param_dict))
                            index = metrics_list.index(np.nanmax(metrics_list))
                            fprint(msg="The final hyper-parameters is following:")
                            print(metrics_data_list[index])
                            pass

                        elif self.param_optim_strategy.lower() == "bayes":
                            param_grid = remove_null_key(param_dict)
                            #     {
                            #     # 'batch_size': [4],
                            #     # 'learning_rate': [0.001, 0.01],
                            #     'batch_size': (4, 64),
                            #     'learning_rate': (0.001, 0.5)
                            #     # 'max_epochs': [5, 10]
                            # }
                            def optimize_function(batch_size,learning_rate):
                                self.init(self.init_method, self.optimizer, learning_rate, self.momentum, self.weight_decay,
                                          self.nesterov,
                                          self.betas, self.eps, self.lr_decay, self.initial_accumulator_value,
                                          self.rho, self.alpha,
                                          self.lambd, self.t0,
                                          self.max_iter, self.max_eval)
                                save_path = "{}_batch_size_{}_lr_{}.pdf".format(self.save_path,int(batch_size),learning_rate)
                                metrics = kfold_cross_validation(self.encode_method, self.concat_reverse, self.workers,
                                                                 self.timer, k_fold, None, self.epochs, int(batch_size),
                                                                 self.model, self.init_method, self.optim, train_dataset,
                                                                 self.metrics,
                                                                 save_path,
                                                                 self.enhancer_len, self.promoter_len, self.num_heads,
                                                                 self.num_layers, self.num_hiddens,
                                                                 self.
                                                                 ffn_num_hiddens,
                                                                 self.device, self.verbose)
                                return metrics
                            print(param_grid)
                            optimizer = BayesianOptimization(f=optimize_function,pbounds=param_grid,
                                                 verbose=2,
                                                 # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
                                                 random_state=self.random_state)
                            optimizer.maximize(
                                init_points=self.init_points,  # 执行随机搜索的步数
                                n_iter=self.n_iter,  # 执行贝叶斯优化的步数
                            )
                            print(optimizer.max)

                    else:
                        metrics = kfold_cross_validation(self.encode_method, self.concat_reverse, self.workers, self.timer,k_fold, log_writer, self.epochs, self.batch_size,
                                                         self.model, self.init_method, self.optim, train_dataset, self.metrics,
                                                         self.save_path,
                                                         self.enhancer_len, self.promoter_len, self.num_heads,
                                                         self.num_layers, self.num_hiddens,
                                                         self.
                                                         ffn_num_hiddens,
                                                         self.device, self.verbose)

            self.timer.final_stop()
            fprint(msg="Total time: {}s".format(self.timer.final_elapsed_time()))




def ddp_train(gpu_rank,world_size,gpus,nr,timer,epochs,log_writer,training_model,train_dataset,test_dataset,optim,
            batch_size,workers,
            save_param_dir,save_param_prefix,
          verbose):
    print(gpu_rank)
    rank = nr * gpus + gpu_rank
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    ## prepare the dataloader
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank
    )
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset,
        num_replicas=world_size,
        rank=rank
    )
    ddp_train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False,
                                      pin_memory=True,
                                      num_workers=workers, sampler=train_sampler)
    ddp_test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,
                                     num_workers=workers, sampler=test_sampler)
    ##
    torch.cuda.set_device(gpu_rank)
    training_model.cuda(gpu_rank)

    training_model = nn.parallel.DistributedDataParallel(training_model,
                                                device_ids=[gpu_rank])

    for epoch in range(epochs):
        timer.start()
        train_each(epoch, log_writer, "Train", training_model, ddp_train_dataloader, ddp_test_dataloader, optim,
                   None, verbose)
        timer.stop()
        timer.final_stop()
        fprint(msg="Cost: {:.3f}s/Epoch, Avg cost: {:.3f}s/Epoch".format(timer.elapsed_time(),timer.final_elapsed_time()/(epoch+1)))
    fprint(msg="Saving parameters of current model...")
    save_weights(training_model, save_param_dir, save_param_prefix)
    fprint(msg="Saved done!")