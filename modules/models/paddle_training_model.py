import sys
from datetime import datetime
import os
import re

import numpy as np
import paddle
# from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, precision_recall_curve, auc, f1_score
# from scipy import interp
# from sklearn.model_selection import GridSearchCV
# from skorch import NeuralNetClassifier
from bayes_opt import BayesianOptimization
from paddle.io import Dataset, DataLoader
from paddle import nn
from tensorboardX import SummaryWriter
# from d2l import torch as d2l
from modules import fprint, check_and_create_dir, format_dict_to_filename, read_config_file, Timer
from modules.datasets.paddle_datasets import EPIDatasets
from modules.evaluation.evaluation import ModelEvaluator
from modules.models.paddle_models import SimpleCNN, init_weights
#
# paddle.backends.cudnn.enabled = False
#
# paddle.backends.cuda.matmul.allow_tf32 = True
# paddle.backends.cudnn.benchmark = False
# paddle.backends.cudnn.deterministic = False
# paddle.backends.cudnn.allow_tf32 = True


def train_each(epoch, log_writer, log_mode, model, data_loader,test_loader, optimizer, device,verbose):
    # optimizer = paddle.optim.SGD(model.parameters(),lr=0.1,momentum=0.9)
    # 初始化logWriter

    loss = None
    model.train()
    criterion = nn.CrossEntropyLoss()
    for batch, (data, label) in enumerate(data_loader):

        # print(data.shape)
        # print(labels.shape)
        # data = data.to(device)
        # label = label.to(device)
        label = paddle.flatten(label)
        # print(label)
        # label = label.long()
        # print("label: {}".format(label.shape))
        # enhancer_X = paddle.permute(data[:, :, 1:5],(0,2,1))
        # promoter_X = paddle.permute(data[:, :, 5:9],(0,2,1))
        if epoch == 0 and batch == 0:
            if log_writer:
                pass
                # log_writer.add_graph(model, data)

        optimizer.clear_grad()
        # loss = cross_entropy_loss(model(enhancer_X,promoter_X),labels).backward()
        # print(label.shape)
        # print("data: {}".format(data.shape))
        output = model(data)
        # print("output: {}".format(output.shape))
        loss = criterion(output, label)
        # print(label)
        train_acc = paddle.static.accuracy(output, paddle.reshape(label,shape=[-1,1])) / label.numel()
        # model.eval()
        test_loss = 0
        correct = 0
        if test_loader:
            # test_len = len(test_loader)
            with paddle.no_grad():
                for index,(data, label) in enumerate(test_loader):
                    # data = data.to(device)
                    # label = label.to(device)
                    label = paddle.flatten(label)
                    # print(label)
                    # label = label.long()
                    # print("label: {}".format(label.shape))
                    # enhancer_X = paddle.permute(data[:, :, 1:5], (0, 2, 1))
                    #
                    # promoter_X = paddle.permute(data[:, :, 5:9], (0, 2, 1))
                    output = model(data)
                    # print("output: {}".format(output))
                    test_loss += criterion(output, label).item()
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(label.view_as(pred)).sum().item()
                    # print("index:{}".format(index))
                    # print("correct:{}".format(correct))

                test_loss /= label.numel()
                test_accuracy = correct / label.numel()
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
                           kfold,  log_writer, epoches, batch_size, model, init_method, optimizer,raw_dataset, metrics,save_path, device,verbose):
    # optimizer = paddle.optim.SGD(model.parameters(),lr=0.1,momentum=0.9)
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
            training_model = SimpleCNN(encode_method=encode_method,concat_reverse=concat_reverse,init_method=init_method)

        # training_model.to(device)
        # print("{}, {}".format(len(train_ids),len(val_ids)))
        # 获取当前折的训练数据和验证数据
        train_subsampler = paddle.io.RandomSampler(train_ids)
        val_subsampler = paddle.io.RandomSampler(val_ids)

        # print(val_subsampler.indices)
        # print(list(train_subsampler))
        train_batch_sampler = paddle.io.BatchSampler(sampler=train_subsampler,batch_size=batch_size)
        val_batch_sampler = paddle.io.BatchSampler(sampler=val_subsampler, batch_size=batch_size)
        train_loader = paddle.io.DataLoader(raw_dataset, batch_sampler=train_batch_sampler,
                                                   num_workers=workers)
        val_loader = paddle.io.DataLoader(raw_dataset, batch_sampler=val_batch_sampler,
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
            with paddle.no_grad():
                for index,(data, label) in enumerate(val_loader):
                    # data = data.to(device)
                    # label = label.to(device)
                    label = paddle.flatten(label)
                    # print(label)
                    # label = label.long()
                    # print("label: {}".format(label.shape))
                    # enhancer_X = paddle.permute(data[:, :, 1:5], (0, 2, 1))
                    #
                    # promoter_X = paddle.permute(data[:, :, 5:9], (0, 2, 1))
                    output = training_model(data)
                    # print("output: {}".format(output))
                    val_loss += criterion(output, label).item()
                    pred = output.argmax(axis=1, keepdim=True)
                    # print(pred)
                    correct += pred.equal(label.reshape(pred.shape)).sum().item()
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
                correct_class_probabilities = output[paddle.arange(len(label)), label]
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
            print(f"Epoch {epoch}: Val Loss: {val_loss:.4f}, Val Accuracy: {accuracy:.2f}%, Cost: {timer.elapsed_time():.3f}s/Epoch")

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
    paddle.save(net.state_dict(), saved_weight_file_path)


def load_weigths(saved_weights_dir='./cache/model', saved_weight_file='weights'):
    if os.path.exists(saved_weights_dir):
        latest_time = find_lateset_weight(saved_weights_dir)
        if latest_time:
            latest_file = saved_weight_file + "_" + datetime.strftime(latest_time, "%Y%m%d-%H%M%S")
            latest_file_path = os.path.join(saved_weights_dir, latest_file)
            print("Loading the latest file: {}".format(latest_file_path))
            latest_params = paddle.load(latest_file_path)
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
            file_time = datetime.strptime(file.split("_")[1],"%Y%m%d-%H%M%S")
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
    if paddle.version.cuda():
        print("CUDA is available.")
    else:
        print("CUDA is not available.")
    # Get the total number of available GPUs
    num_gpus = paddle.device.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")

    # Iterate through each GPU and print detailed information
    for gpu_id in range(num_gpus):
        device = f"gpu:{gpu_id}"  # Specify the GPU device
        print(f"Device: {device}")
        print(f"Name: {paddle.device.cuda.get_device_name(device)}")
        print(f"Compute Capability: {paddle.device.cuda.get_device_capability(device)}")
        print(f"Total Memory: {paddle.device.cuda.get_device_properties(device).total_memory / 1e9:.2f} GB")
        print(f"CUDA Version: {paddle.version.cuda()}")
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
    def __init__(self, train_dataset_dir, train_dataset_pattern, train_dataset_file,test_dataset_file,workers,
                 model, encode_method,concat_reverse,
                 is_param_optim, param_optim_strategy,params_config,random_size,
                 init_points,n_iter,
                 k_fold,metrics,save_path,
                 init_method, epochs, lr, optimizer, batch_size, momentum,
                 weight_decay, nesterov, betas, eps, lr_decay, initial_accumulator_value, rho, alpha,
                 lambd, t0, max_iter, max_eval,
                 device, random_state,
                 rerun, save_param_dir, save_param_prefix, log_dir, save_model, verbose):
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
            self.device = paddle.device("cuda" if paddle.cuda.is_available() else "cpu")
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
            self.training_model = SimpleCNN(encode_method=self.encode_method, concat_reverse=self.concat_reverse, init_method=init_method)
            pass
        if (not self.rerun) and self.save_param_dir is not None and self.save_param_prefix is not None:
            # print('撒大声地')
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
        ## set the device to run
        paddle.device.set_device(self.device)
        # self.training_model.to(self.device)
        fprint(msg='training on {}'.format(self.device))
        # self.training_model.apply(init_weights)
        if optim.lower() == "sgd":
            self.optim = paddle.optimizer.SGD(parameters=self.training_model.parameters(),learning_rate=lr,
                                              weight_decay=weight_decay)
        elif optim.lower() == "adam":
            self.optim = paddle.optimizer.Adam(parameters=self.training_model.parameters(),learning_rate=lr, beta1=betas[0],beta2=betas[1],epsilon=eps,weight_decay=weight_decay)
        elif optim.lower() == "adagrad":
            self.optim = paddle.optimizer.Adagrad(parameters=self.training_model.parameters(),learning_rate=lr,lr_decay=lr_decay,weight_decay=weight_decay,
                                             initial_accumulator_value=initial_accumulator_value,epsilon=eps)
        elif optim.lower() == "adadelta":
            self.optim = paddle.optimizer.Adadelta(parameters=self.training_model.parameters(),learning_rate=lr,rho=rho,epsilon=eps,weight_decay=weight_decay)
        elif optim.lower() == "rmsprop":
            self.optim = paddle.optimizer.RMSprop(parameters=self.training_model.parameters(),learning_rate=lr,alpha=alpha,epsilon=eps,weight_decay=weight_decay,momentum=momentum)
        elif optim.lower() == "asgd":
            self.optim = paddle.optimizer.ASGD(parameters=self.training_model.parameters(),learning_rate=lr,lambd=lambd,alpha=alpha,t0=t0,weight_decay=weight_decay)
        elif optim.lower() == "lbfgs":
            self.optim = paddle.optimizer.LBFGS(parameters=self.training_model.parameters(),learning_rate=lr, max_iter=max_iter,max_eval=max_eval)
        else:
            fprint("WARNING","Not supported for {} optimization method, so the default SGD method was used!!!".format(optim))
            self.optim = paddle.optimizer.SGD(parameters=self.training_model.parameters(),learning_rate=lr,weight_decay=weight_decay)
        # return training_model, optim

    def run(self):
        self.timer.final_start()
        if self.train_dataset_file is None:
            for file in os.listdir(self.train_dataset_dir):
                if re.findall(self.train_dataset_pattern,file):
                    self.train_dataset_files.append(os.path.join(self.train_dataset_dir,file))
        current_log_dir = os.path.join(self.log_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
        log_writer = init_summary_writer(current_log_dir)
        if len(self.train_dataset_files) > 0:
            if self.k_fold:
                fprint("WARNING","The multi-datasets with k-fold CV are only supported by general training mode now!!!")
                pass
            for epoch in range(self.epochs):
                for train_dataset_file in self.train_dataset_files:
                    dataset = EPIDatasets(cache_file_path=train_dataset_file)
                    train_dataloader = DataLoader(dataset,batch_size=self.batch_size,shuffle=False)
                    test_dataset = EPIDatasets(cache_file_path=self.test_dataset_file)
                    test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
                    self.timer.start()
                    train_each(epoch, log_writer, "Train", self.training_model, train_dataloader,test_dataloader, self.optim, self.device, self.verbose)
                    self.timer.stop()
                    fprint(msg="Cost: {:3f}s/Epoch".format(self.timer.elapsed_time()))
        else:
            train_dataset_file = self.train_dataset_file
            dataset = EPIDatasets(cache_file_path=train_dataset_file)
            train_dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False,num_workers=self.workers)
            test_dataset = EPIDatasets(cache_file_path=self.test_dataset_file)
            test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False,num_workers=self.workers)
            if self.k_fold is None:

                for epoch in range(self.epochs):
                    self.timer.start()
                    train_each(epoch, log_writer, "Train", self.training_model, train_dataloader,test_dataloader, self.optim, self.device, self.verbose)
                    self.timer.stop()
                    fprint(msg="Cost: {:.3f}s/Epoch".format(self.timer.elapsed_time()))
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
                                                             self.model, self.init_method, self.optim, dataset, self.metrics,
                                                             save_path, self.device, self.verbose)
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
                                                             self.model, self.init_method, self.optim, dataset, self.metrics,
                                                             save_path, self.device, self.verbose)
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
                                                             self.model, self.init_method, self.optim, dataset,
                                                             self.metrics,
                                                             save_path, self.device, self.verbose)
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
                                                     self.model, self.init_method, self.optim, dataset, self.metrics,
                                                     self.save_path, self.device, self.verbose)

        self.timer.final_stop()
        fprint(msg="Total time: {}s".format(self.timer.final_elapsed_time()))



