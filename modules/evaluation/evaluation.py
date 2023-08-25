import numpy as np
import matplotlib.pyplot as plt
from scipy import interp
from scipy.interpolate import interp1d

from sklearn.metrics import roc_curve, precision_recall_curve, auc, f1_score
import pandas as pd


class ModelEvaluator:
    def __init__(self, y_true_list, y_pred_list):
        self.y_true_list = y_true_list
        self.y_pred_list = y_pred_list

    def calculate_roc_curve(self):
        all_fpr = []
        all_tpr = []
        all_roc_auc = []

        for y_true, y_pred in zip(self.y_true_list, self.y_pred_list):
            fpr, tpr, _ = roc_curve(y_true.cpu(), y_pred.cpu())
            # interp_func = interp1d(fpr, tpr, kind='linear')
            roc_auc = auc(fpr, tpr)
            all_fpr.append(fpr)
            all_tpr.append(tpr)
            all_roc_auc.append(roc_auc)

        # mean_fpr = np.mean(all_fpr, axis=0)
        # mean_tpr = np.mean(all_tpr, axis=0)
        mean_fpr, mean_tpr = self.interp_roc_curve(all_fpr,all_tpr)
        mean_roc_auc = auc(mean_fpr, mean_tpr)

        return mean_fpr, mean_tpr, mean_roc_auc

    def interp_roc_curve(self,fpr_list,tpr_list,num=None):
        # 计算所有折中最大的 FPR 和最小的 TPR
        max_fpr = max(max(fpr_fold) for fpr_fold in fpr_list)
        max_fpr_len = max(len(fpr_fold) for fpr_fold in fpr_list)
        # min_tpr = min(min(tpr_fold) for tpr_fold in tpr_list)
        # 定义新的 FPR 范围
        if num:
            new_fpr = np.linspace(0, max_fpr, num=num)
        else:
            new_fpr = np.linspace(0, max_fpr, num=max_fpr_len)
        interp_tpr_list = []
        for fpr_fold, tpr_fold in zip(fpr_list, tpr_list):
            interp_func = interp1d(fpr_fold, tpr_fold, kind='linear')
            interp_tpr = interp_func(new_fpr)
            interp_tpr_list.append(interp_tpr)
        pass
        return new_fpr,np.nanmean(interp_tpr_list, axis=0)


    def calculate_f1_score(self):
        f1_scores = []

        for y_true, y_pred in zip(self.y_true_list, self.y_pred_list):
            f1 = f1_score(y_true.cpu(), np.round(y_pred.cpu()))
            f1_scores.append(f1)

        mean_f1 = np.nanmean(f1_scores)
        return mean_f1

    def plot_mean_roc_curve(self, save_path=None):
        mean_fpr, mean_tpr, mean_roc_auc = self.calculate_roc_curve()
        plt.figure()
        plt.plot(mean_fpr, mean_tpr, color='darkorange', lw=2,
                 label='Mean ROC (area = %0.2f)' % mean_roc_auc)
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Mean Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        if save_path:
            plt.savefig(save_path)  # Save the plot as an image
        plt.show()
        return mean_roc_auc

    def export_to_csv(self, filename):
        # Combine data from all folds into single arrays
        combined_y_true = np.concatenate(self.y_true_list)
        combined_y_pred = np.concatenate(self.y_pred_list)

        results = {
            'True Label': combined_y_true,
            'Predicted Probability': combined_y_pred
        }
        df = pd.DataFrame(results)
        df.to_csv(filename, index=False)