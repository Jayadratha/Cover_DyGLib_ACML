import torch
import torch.nn as nn
import torch.nn.functional as F

############################################### CwR #######################################################
class CwRLoss(nn.Module):
    def __init__(self, c):
        super(CwRLoss, self).__init__()
        self.c = c

    def forward(self, output, target):
        label_rej = 0 * target + 2
        loss_fn = nn.CrossEntropyLoss()
        loss1 = loss_fn(output, target)
        loss2 = loss_fn(output, label_rej)
        loss = (loss1 + ((1 - self.c) * loss2))
        return loss

############################################### Cls_Specific_CwR #######################################################
class CwRLoss_mod(nn.Module):
    def __init__(self, c):
        super(CwRLoss_mod, self).__init__()
        self.c = c

    def forward(self, output, target):
        label_rej = 0 * target + 2
        loss_fn = nn.CrossEntropyLoss(reduction='none')
        loss1 = loss_fn(output, target)
        loss1 = torch.mean(loss1)

        loss2 = (1 - target) * loss_fn(output, label_rej)
        loss2 = torch.mean(loss2)
  
        loss = (loss1 + ((1 - self.c) * loss2)) # only reject if output is 0 (i.e. majority class)
        return loss
############################################### SelectiveNet #######################################################
# emperical coverage
def emp_cov(selective_prob, labels): # emp_cov(output,labels) = fi(g|S_m)= (1/|m|) * sum_{i to m} g(x_i)
    sel = selective_prob
    return torch.mean(sel)

# # true coverage
# def true_cov(output, labels, t=0.5):
#     sel = output[:, -1]
#     sel = torch.where(sel>t, 1, 0).double() # .double is used to convert 0,1 to 0.0,1.0 for torch.mean(float tensor)
#     return torch.mean(sel)


# include coverage in loss
class SelectiveLoss(nn.Module):
    def __init__(self, lambda_val=32):
        super(SelectiveLoss, self).__init__()
        self.lambda_val = lambda_val
        self.loss_fn = nn.BCELoss(reduction='none')
        # self.loss_fn = AUCMLoss()

    def forward(self, input, selective_prob, labels, coverage): # interior point method loss
        pred = input # true prediction
        sel = selective_prob # selection probability g(x_i)
        cov = emp_cov(sel, labels) # emperical coverage = fi(g|S_m)= (1/|m|) * sum_{i to m} g(x_i)

        # Compute the standard BCE loss
        loss_standard = torch.sum(self.loss_fn(pred, labels) * sel) # sum_{i to m} (L_f(x_i,y_i) * g(x_i))
        loss_standard /= (cov * len(sel))
        # loss = emperical selective risk = [(1/|m|) * [sum_{i to m} (L_f(x_i,y_i) * g(x_i))]] / fi(g|S_m)

        # Compute psi=(max((coverage−cov),0))^2
        psi = torch.square(torch.clamp(coverage - cov, min=0))

        # Compute the final loss
        loss = loss_standard + self.lambda_val * psi.sum() # L_(f,g) = emperical selective risk + lambda * psi

        # interior point method loss
        return loss

class SelectiveLoss_Cls_Agnostic(nn.Module):
    def __init__(self, lambda_val=32):
        super(SelectiveLoss, self).__init__()
        self.lambda_val = lambda_val
        self.loss_fn = nn.BCELoss(reduction='none')

    def forward(self, input, selective_prob, labels, coverage): # interior point method loss
        pred = input # true prediction
        sel = selective_prob # selection probability g(x_i)
        cov = emp_cov(sel, labels) # emperical coverage = fi(g|S_m)= (1/|m|) * sum_{i to m} g(x_i)

        # Compute the standard BCE loss
        loss_standard = torch.sum(self.loss_fn(pred, labels) * sel) # sum_{i to m} (L_f(x_i,y_i) * g(x_i))
        loss_standard /= (cov * len(sel))
        # loss = emperical selective risk = [(1/|m|) * [sum_{i to m} (L_f(x_i,y_i) * g(x_i))]] / fi(g|S_m)

        # Compute psi=(max((coverage−cov),0))^2
        psi = torch.square(torch.clamp(coverage - cov, min=0))

        # Compute the final loss
        loss = loss_standard + self.lambda_val * psi.sum() # L_(f,g) = emperical selective risk + lambda * psi

        return loss
    
# find thresholds for respective coverage
def find_tres(selective_prob):
    sel = selective_prob
    sel = sel.tolist()
    sel_min = min(sel)
    sel_max = max(sel)
    sel.sort(reverse=True)
    c = [0.9, 0.8, 0.7, 0.6, 0.5]
    c = [int(x * len(sel)) for x in c]
    return sel[c[0]], sel[c[1]], sel[c[2]], sel[c[3]], sel[c[4]], sel_min, sel_max

class SelectiveLoss_Cls_Specific(nn.Module):
    def __init__(self, minority_lambda_val=32, majority_lambda_val=32):
        super(SelectiveLoss_Cls_Specific, self).__init__()
        self.minority_lambda_val = minority_lambda_val
        self.majority_lambda_val = majority_lambda_val
        self.loss_fn = nn.BCELoss(reduction='none')

    def forward(self, input, selective_prob, labels, minority_coverage, majority_coverage):
        pred = input
        sel = selective_prob

        # for minority class
        sel_minority = sel[labels==1]
        cov_minority = emp_cov(sel_minority, labels[labels==1]) # emperical coverage minority = fi_minority(g|S_m)= (1/|m_minority|) * sum_{i to m_minority} g(x_i_minority)
        pred_minority = pred[labels==1]

        # for majority class
        sel_majority = sel[labels==0]
        cov_majority = emp_cov(sel_majority, labels[labels==0]) # emperical coverage majority = fi_majority(g|S_m)= (1/|m_majority|) * sum_{i to m_majority} g(x_i_majority)
        pred_majority = pred[labels==0]
        # print("\n \n \n")
        # print("number of minority: ", len(sel_minority))


        # Compute the standard BCE loss for minority class
        loss_standard_minority = torch.sum(self.loss_fn(pred_minority, labels[labels==1]) * sel_minority) # sum_{i to m_minority} (L_f(x_i_minority,y_i_minority) * g(x_i_minority))
        loss_standard_minority /= (cov_minority * len(sel_minority)) # L_f(x_i_minority,y_i_minority) * g(x_i_minority) / fi_minority(g|S_m)
        # print("loss_standard_minority: ", loss_standard_minority)
  

        psi_minority = torch.square(torch.clamp(minority_coverage - cov_minority, min=0)) # psi_minority=(max((minority_coverage−cov_minority),0))^2
        # print("psi_minority: ", psi_minority)

        # Compute the final loss for minority class
        loss_minority = loss_standard_minority + self.minority_lambda_val * psi_minority  # L_(f_minority,g_minority) = minority emperical selective risk + lambda * psi_minority
        # print("loss_minority: ", loss_minority)

        # Compute the standard BCE loss for majority class
        loss_standard_majority = torch.sum(self.loss_fn(pred_majority, labels[labels==0]) * sel_majority) # sum_{i to m_majority} (L_f(x_i_majority,y_i_majority) * g(x_i_majority))
        loss_standard_majority /= (cov_majority * len(sel_majority)) # L_f(x_i_majority,y_i_majority) * g(x_i_majority) / fi_majority(g|S_m)
        # print("loss_standard_majority: ", loss_standard_majority)

        # Compute psi=(max((coverage−cov),0))^2 for majority class
        psi_majority = torch.square(torch.clamp(majority_coverage - cov_majority, min=0)) # psi_majority=(max((majority_coverage−cov_majority),0))^2
        # print("psi_majority: ", psi_majority)

        # Compute the final loss for majority class
        loss_majority = loss_standard_majority + self.majority_lambda_val * psi_majority # L_(f_majority,g_majority) = majority emperical selective risk + lambda * psi_majority
        # print("loss_majority: ", loss_majority)


        # Compute the final loss
        if len(sel_minority) == 0:
            loss = loss_majority
        elif len(sel_majority) == 0:
            loss = loss_minority
        else:
            loss = loss_minority + loss_majority            
            # loss = (loss_standard_majority + loss_standard_minority)/  (cov_majority * len(sel_majority) + cov_minority * len(sel_minority)) + self.majority_lambda_val * psi_majority.sum() + self.minority_lambda_val * psi_minority.sum()
            # loss = (loss_standard_majority * len(sel_majority) + loss_standard_minority * len(sel_minority))/  (len(sel_majority) + len(sel_minority)) + self.majority_lambda_val * psi_majority + self.minority_lambda_val * psi_minority
            # L_(f,g) = majority emperical selective risk + minority emperical selective risk + lambda * psi_majority + lambda * psi_minority
        # print("loss: ", loss)
        # exit()

        return loss
    
class SelectiveLoss_Cls_Specific_v2(nn.Module):
    def __init__(self, minority_lambda_val=32, majority_lambda_val=32):
        super(SelectiveLoss_Cls_Specific_v2, self).__init__()
        self.minority_lambda_val = minority_lambda_val
        self.majority_lambda_val = majority_lambda_val
        self.loss_fn = nn.BCELoss(reduction='none')

    def forward(self, input, selective_prob, labels, minority_coverage, majority_coverage):
        pred = input
        sel = selective_prob

        # total emperical coverage
        cov = emp_cov(sel, labels)

        # for minority class
        sel_minority = sel[labels==1]
        cov_minority = emp_cov(sel_minority, labels[labels==1]) # emperical coverage minority = fi_minority(g|S_m)= (1/|m_minority|) * sum_{i to m_minority} g(x_i_minority)
        # print("\n \n \n")
        # print("len(sel_minority): ", len(sel_minority))
        # for majority class
        sel_majority = sel[labels==0]
        cov_majority = emp_cov(sel_majority, labels[labels==0]) # emperical coverage majority = fi_majority(g|S_m)= (1/|m_majority|) * sum_{i to m_majority} g(x_i_majority)
        # print("cov_majority: ", cov_majority)
        # print("cov_minority: ", cov_minority)
        # print("cov: ", cov)

        # Compute the standard BCE loss for all classes
        loss_standard = torch.sum(self.loss_fn(pred, labels) * sel)
        loss_standard /= (cov * len(sel))

        # Compute the IPM loss for minority class
        # if cov_minority == 'none': # if no minority class
        if len(sel_minority) == 0:
            psi_minority = 0
        else:
            psi_minority = torch.square(torch.clamp(minority_coverage - cov_minority, min=0)) # psi_minority=(max((minority_coverage−cov_minority),0))^2
        # print("psi_minority: ", psi_minority)
        # Compute the IPM loss for majority class
        psi_majority = torch.square(torch.clamp(majority_coverage - cov_majority, min=0)) # psi_majority=(max((majority_coverage−cov_majority),0))^2
        # print("psi_majority: ", psi_majority)

        # Compute the final loss
        loss = loss_standard + self.majority_lambda_val * psi_majority + self.minority_lambda_val * psi_minority

        return loss
    


class SelectiveLoss_Cls_Specific_v3(nn.Module):
    def __init__(self, minority_lambda_val=32, majority_lambda_val=32):
        super(SelectiveLoss_Cls_Specific_v3, self).__init__()
        self.minority_lambda_val = minority_lambda_val
        self.majority_lambda_val = majority_lambda_val
        self.loss_fn = nn.BCELoss(reduction='none')

    def forward(self, input, selective_prob, labels, minority_coverage, majority_coverage):
        pred = input
        sel = selective_prob

        # for minority class
        sel_minority = sel[labels==1]
        cov_minority = emp_cov(sel_minority, labels[labels==1]) # emperical coverage minority = fi_minority(g|S_m)= (1/|m_minority|) * sum_{i to m_minority} g(x_i_minority)
        pred_minority = pred[labels==1]

        # for majority class
        sel_majority = sel[labels==0]
        cov_majority = emp_cov(sel_majority, labels[labels==0]) # emperical coverage majority = fi_majority(g|S_m)= (1/|m_majority|) * sum_{i to m_majority} g(x_i_majority)
        pred_majority = pred[labels==0]
        # print("\n \n \n")
        # print("number of minority: ", len(sel_minority))


        # Compute the standard BCE loss for minority class
        loss_standard_minority = torch.sum(self.loss_fn(pred_minority, labels[labels==1]) * sel_minority) # sum_{i to m_minority} (L_f(x_i_minority,y_i_minority) * g(x_i_minority))
        loss_standard_minority /= (cov_minority * len(sel_minority)) # L_f(x_i_minority,y_i_minority) * g(x_i_minority) / fi_minority(g|S_m)
  

        psi_minority = torch.square(torch.clamp(minority_coverage - cov_minority, min=0)) # psi_minority=(max((minority_coverage−cov_minority),0))^2
        # print("psi_minority: ", psi_minority)
        # print("psi_minority* lambda: ", psi_minority * self.minority_lambda_val)

        # Compute the final loss for minority class
        loss_minority = loss_standard_minority + self.minority_lambda_val * psi_minority  # L_(f_minority,g_minority) = minority emperical selective risk + lambda * psi_minority
        # print("loss_minority: ", loss_minority)

        # Compute the standard BCE loss for majority class
        loss_standard_majority = torch.sum(self.loss_fn(pred_majority, labels[labels==0]) * sel_majority) # sum_{i to m_majority} (L_f(x_i_majority,y_i_majority) * g(x_i_majority))
        loss_standard_majority /= (cov_majority * len(sel_majority)) # L_f(x_i_majority,y_i_majority) * g(x_i_majority) / fi_majority(g|S_m)
        # print("loss_standard_majority: ", loss_standard_majority)

        # Compute psi=(max((coverage−cov),0))^2 for majority class
        psi_majority = torch.square(torch.clamp(majority_coverage - cov_majority, min=0)) # psi_majority=(max((majority_coverage−cov_majority),0))^2
        # print("psi_majority: ", psi_majority)
        # print("psi_majority* lambda: ", psi_majority * self.majority_lambda_val)

        # Compute the final loss for majority class
        loss_majority = loss_standard_majority + self.majority_lambda_val * psi_majority # L_(f_majority,g_majority) = majority emperical selective risk + lambda * psi_majority
        # print("loss_majority: ", loss_majority)


        # Compute the final loss
        if len(sel_minority) == 0:
            loss = loss_majority
        elif len(sel_majority) == 0:
            loss = loss_minority
        else:
            # loss = loss_minority + loss_majority            
            # loss = (loss_standard_majority + loss_standard_minority)/  (cov_majority * len(sel_majority) + cov_minority * len(sel_minority)) + self.majority_lambda_val * psi_majority.sum() + self.minority_lambda_val * psi_minority.sum()
            # loss = (loss_standard_majority * len(sel_majority) + loss_standard_minority * len(sel_minority))/  (len(sel_majority) + len(sel_minority)) + self.majority_lambda_val * psi_majority + self.minority_lambda_val * psi_minority
            loss = (loss_standard_majority * len(sel_majority) + loss_standard_minority * len(sel_minority) + self.majority_lambda_val * psi_majority * len(sel_minority) + self.minority_lambda_val * psi_minority * len(sel_majority))/(len(sel_majority) + len(sel_minority))
            # L_(f,g) = majority emperical selective risk + minority emperical selective risk + lambda * psi_majority + lambda * psi_minority
        # print("loss: ", loss)
        # exit()

        return loss
   