import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

# rewrite in pytorch!
def weighting(image, batch_size, num_classes, weight_type='log'):
    """
    The custom class weighing function.
    INPUTS:
    - images: numpy array of shape (batch_size, num_classes, height, width) 
    - batch_size
    - num_classes
    - weight_type(str): 
        -- 'log': 1 / np.log(1.02 + (freq_c / total_freq_c))
        -- 'median': f = Median_freq_c / total_freq_c
    OUTPUTS:
    - class_weights(list): a list of class weights where each index represents each class label
    and the element is the class weight for that label;
    """
    #initialize dictionary with all 0
    
    label_to_frequency = {}
    for i in range(num_classes):
        label_to_frequency[i] = []
    
    #count frequency of each class for images
    
    for n in range(batch_size):
        for i in range(num_classes):
            
            class_mask = np.equal(image[n, i], 1)
            class_mask = class_mask.astype(np.float32)
            class_frequency = (class_mask.sum())

            if class_frequency != 0.0:
                label_to_frequency[i].append(class_frequency)

    
    #applying weighting function and appending the class weights to class_weights
    
    class_weights = np.zeros((num_classes))
    
    total_frequency = 0
    for frequencies in label_to_frequency.values():
        total_frequency += sum(frequencies)
    
    i = 0
    if weight_type == 'log':
     
        for class_, freq_ in label_to_frequency.items():

            class_weight = 1 / np.log(1.02 + (sum(freq_) / total_frequency))
            class_weights[i] = class_weight
            i += 1
        
      
    elif weight_type == 'median':
      
        for i, j in label_to_frequency.items():
            #To obtain the median, we got to sort the frequencies
            j = sorted(j) 

            median_freq_c = np.median(j) / sum(j)
            total_freq_c = sum(j) / total_frequency
            median_frequency_balanced = median_freq_c / total_freq_c
            class_weights[i] = median_frequency_balanced
            i += 1
    # as first goes background
    class_weights[0] = 0.0 
    # normalize weights:
    class_weights /= class_weights.sum()
    return class_weights


class Metric():
    def __init__(self, name='None', params=None):
        self.name = name
        self.params = params
        self.curr_val = 0.0
    def __name__(self):
        return self.name
    def __params__(self):
        return self.params
    def __call__(self, outputs, targets):
        self.curr_val= value
        return value
    
    
class Soft_dice_loss(Metric):
    def __init__(self, smooth=1e-15, weighting=None):
        super(Soft_dice_loss, self).__init__('soft_dice_loss', ['smooth', 'weights'])
        self.smooth = smooth
        self.curr_val = 0.0
        self.weighting = weighting
        self.weights = None

    def __call__(self, outputs, targets):
        num_cl = targets.size(1)
        batch_size = targets.size(0)
        w = targets.size(2)
        h = targets.size(3)
        value = 0

        if self.weighting is not None:
            self.weights = weighting(targets.data.cpu().numpy(), targets.size(0), targets.size(1), weight_type=self.weighting)
        else:
            self.weights = np.ones((num_cl))

        for i in range(num_cl):
            out = outputs[:, i].resize(batch_size, 1, w, h)
            targ = targets[:, i].resize(batch_size, 1, w, h)
            
            iflat = out.view(-1)
            tflat = targ.view(-1)
            intersection = (iflat * tflat).sum()
            value += (1 - ((2. * intersection + self.smooth) /
                  (iflat.sum() + tflat.sum() + self.smooth)))*self.weights[i]
        #value = value / num_cl
        self.curr_val = value
        return value
    def get_weights(self):
        return self.weights
    def get_smooth(self):
        return self.smooth


class Pixel_accuracy_metric(Metric):
    def __init__(self, weighting=None):
        '''
         sum_i(n_ii) / sum_i(t_i), where
         n_ij - the number of pixels of class i
         predicted to belong to class j,
         t_i -  the total number of pixels of class i
        '''
        super(Pixel_accuracy_metric, self).__init__('pixel accuracy')
        self.curr_val = 0.0
      
    def __call__(self, outputs, targets):
        num_cl = targets.shape[1]
        batch_size = targets.size(0)
        w = targets.size(2)
        h = targets.size(3)
        value = 0
        n_ii = 0
        t_i = 0      
        
        #count frequency of each class for images
        for i in range(num_cl):
            out = outputs[:, i].resize(batch_size, 1, w, h).exp()
            targ = targets[:, i].resize(batch_size, 1, w, h)
            
            n_ii += (out.view(-1) * targ.view(-1)).sum()
            t_i += targ.view(-1).sum()
       
        value = n_ii / t_i
        
        self.curr_val = value
        return value

class Mean_accuracy_metric(Metric):
    def __init__(self, weighting=None):
        '''
         (1/n_classes)*sum_i(n_ii / t_i), where
         n_ij - the number of pixels of class i
         predicted to belong to class j,
         t_i -  the total number of pixels of class i
         n_classes - the total number of classes in segmentation
        '''
        super(Mean_accuracy_metric, self).__init__('mean accuracy')
        self.curr_val = 0.0
      
    def __call__(self, outputs, targets):
        num_cl = targets.shape[1]
        batch_size = targets.size(0)
        w = targets.size(2)
        h = targets.size(3)
        value = 0
        # dictionary contains the number of pixels of class i predicted to belong to class i
        n_ = {}
        for i in range(num_cl):
            n_[i] = 0
        # dictionary contains the total number of pixels of each class
        t_ = {}
        for i in range(num_cl):
            t_[i] = 0
        
        accuracy = list([0]) * num_cl
        #count frequency of each class for images
        
        for n in range(batch_size):
            for i in range(num_cl):

                out = outputs[:, i].resize(batch_size, 1, w, h).exp()
                targ = targets[:, i].resize(batch_size, 1, w, h)
            
                n_ii += (out.view(-1) * targ.view(-1)).sum()
                t_i += targ.view(-1).sum()

                n_[i] += n_ii
                t_[i] += t_i

                if (t_i != 0):
                    accuracy[i] = n_ii / t_i

        value = np.mean(accuracy)
        self.curr_val = value
        return value

class IoU(Metric):
    def __init__(self, smooth=1e-15, weighting=None):
        super(IoU, self).__init__('intersection over union', ['smooth', 'weights'])
        self.curr_val = 0.0
        self.weighting = weighting
        self.smooth = smooth
        
    def __call__(self, outputs, targets):
        num_cl = targets.size(1)
        batch_size = targets.size(0)
        w = targets.size(2)
        h = targets.size(3)
        value = 0
        
        if self.weighting is not None:
            weights = weighting(targets.numpy(), targets.size(0), targets.size(1), weight_type=self.weighting)
        else:
            weights = np.ones((num_cl))
        
        for i in range(num_cl):
            out = outputs[:, i].resize(batch_size, 1, w, h)
            targ = targets[:, i].resize(batch_size, 1, w, h)
   
            intersection = (out.view(-1) * targ.view(-1)).sum()
                
            union = out.sum() + targ.sum()
            value += ((intersection + self.smooth) / (union - intersection + self.smooth)) * weights[i]
        
        #value /= num_cl
        self.curr_val = value 
        return value

class LossMulti(Metric):
    def __init__(self, class_weights=None, smooth=1e-15, num_classes=9):
        if class_weights is not None:
            nll_weight = torch.from_numpy(class_weights.astype(np.float32)).cuda(async=True)
        else:
            nll_weight = None
        self.nll_loss = nn.NLLLoss2d(weight=nll_weight)
        self.num_classes=num_classes
        self.smooth = smooth
        self.jaccard_weight = 1.0

    def __call__(self, outputs, targets):
        loss = self.nll_loss(outputs, targets)
        for i in range(self.num_classes):
            cls_weight = self.jaccard_weight / self.num_classes
            out = outputs[:, i].resize(batch_size, 1, w, h)
            targ = targets[:, i].resize(batch_size, 1, w, h)
            intersection = (out.view(-1) * targ.view(-1)).sum()
                
            union = out.sum() + targ.sum() + self.smooth
            loss += (1 - intersection / (union - intersection)) * cls_weight
        loss /= (1 + self.jaccard_weight)
        return loss