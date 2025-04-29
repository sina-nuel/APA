# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from torch.optim import Adam
import torchvision.models as models


ALGORITHMS = [
    'APA'
]


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]



class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Algorithm, self).__init__()
        self.hparams = hparams

    def update(self, minibatches, unlabeled=None):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.

        Admits an optional list of unlabeled minibatches from the test domains,
        when task is domain_adaptation.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError
        
class SwinTClassifier(nn.Module):
    def __init__(self, num_classes):
        super(SwinTClassifier, self).__init__()
        self.swin_t = models.swin_t(pretrained=True)  # Load the pretrained Swin-T model
        
        self.swin_t.head = nn.Linear(self.swin_t.head.in_features, num_classes)

    def forward(self, x):
        return self.swin_t(x)  # Return the logits from the Swin-T model
        
        

class APA(Algorithm):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ADRMX, self).__init__(input_shape, num_classes, num_domains, hparams)
        
        self.num_classes = num_classes
        self.num_domains = num_domains
        
        self.mdl = models.resnet50(pretrained=True)

        num_ftrs = self.mdl.fc.in_features
        self.mdl.fc = nn.Linear(num_ftrs, num_classes) 
        
        
        self.register_buffer('update_count', torch.tensor([0]))
        self.optimizer = Adam(
          self.mdl.parameters(),
          lr=hparams["lr"],
          betas=(hparams['beta1'], 0.9),
          weight_decay=hparams.get("weight_decay", 0)  
          )
        
        
        
        
        
       
      
        
       
        

        
        

        

    def forward(self, x):
        
        
        logits = self.mdl(x)
        return logits   
        
        
    def update(self, minibatches):
        

        self.update_count += 1
        all_x = torch.cat([x for x, _ in minibatches])
        all_y = torch.cat([y for _, y in minibatches])
        
        
        
        
        if self.update_count%self.hparams['mixStep']<self.hparams['normStep'] :
            #Normal training steps
       
            logits = self(all_x)
        
            
            loss = F.cross_entropy(logits, all_y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            return {'loss': loss.item()}
        else:  
            #Augmentation steps
                
            
        
        
            
            
            classifier_remixed_loss = 0
            for i in range(self.num_classes):
                indices = torch.where(all_y == i)[0]
                for _ in range(1):

                    perm = torch.randperm(indices.numel())
                    if len(perm) < 2:
                        continue
                    idx1, idx2 = perm[:2]
                    
                    idx2 = indices[idx2].item()
                    idx1 = indices[idx1].item()
                    
                    remixed_img = self.fft_image_sum(all_x[idx1],all_x[idx2])
                    
                   
                    att_img1 = self(remixed_img)
                   
                    
            
                    classifier_remixed_loss += F.cross_entropy(att_img1.view(1, -1), all_y[idx1].view(-1))
                    
                    remixed_img2 = self.fft_image_sum(all_x[idx2],all_x[idx1])
                    
                    att_img2 = self(remixed_img2)
                    
                  
                    
                    classifier_remixed_loss += F.cross_entropy(att_img2.view(1, -1), all_y[idx2].view(-1))
                    
                    
                    
                   

                    
            classifier_remixed_loss /= (self.num_classes *2)
            self.optimizer.zero_grad()
            classifier_remixed_loss.backward()
            self.optimizer.step()
            
            return {'loss': classifier_remixed_loss.item()}
        

    def predict(self, x):
        with torch.no_grad():
            logits = self(x)
            
        return logits
    


    def fft_image_sum(self, img1, img2):
        device = img1.device
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
        
    
        image1_unnorm = img1 * std + mean
        image2_unnorm = img2 * std + mean
    
        fft_r1 = torch.fft.fft2(image1_unnorm[0, :, :])
        fft_g1 = torch.fft.fft2(image1_unnorm[1, :, :])
        fft_b1 = torch.fft.fft2(image1_unnorm[2, :, :])
    
        fft_r2 = torch.fft.fft2(image2_unnorm[0, :, :])
        fft_g2 = torch.fft.fft2(image2_unnorm[1, :, :])
        fft_b2 = torch.fft.fft2(image2_unnorm[2, :, :])
    
        amp_r1, amp_g1, amp_b1 = torch.abs(fft_r1), torch.abs(fft_g1), torch.abs(fft_b1)
        amp_r2, amp_g2, amp_b2 = torch.abs(fft_r2), torch.abs(fft_g2), torch.abs(fft_b2)
    
        phase_r1, phase_g1, phase_b1 = torch.angle(fft_r1), torch.angle(fft_g1), torch.angle(fft_b1)
    
        new_amp_r = torch.sqrt(amp_r1 * amp_r2) + 1e-8
        new_amp_g = torch.sqrt(amp_g1 * amp_g2) + 1e-8
        new_amp_b = torch.sqrt(amp_b1 * amp_b2) + 1e-8
        

        new_fft_r = new_amp_r * torch.exp(1j * phase_r1)
        new_fft_g = new_amp_g * torch.exp(1j * phase_g1)
        new_fft_b = new_amp_b * torch.exp(1j * phase_b1)
    
        new_image_r = torch.fft.ifft2(new_fft_r).real
        new_image_g = torch.fft.ifft2(new_fft_g).real
        new_image_b = torch.fft.ifft2(new_fft_b).real
    
        new_image = torch.stack([new_image_r, new_image_g, new_image_b])
    
        new_image_min = new_image.view(3, -1).min(dim=1, keepdim=True)[0]
        new_image_max = new_image.view(3, -1).max(dim=1, keepdim=True)[0]
    
        new_image = (new_image - new_image_min.view(3, 1, 1)) / (new_image_max.view(3, 1, 1) - new_image_min.view(3, 1, 1))
    
        
    
        new_image = (new_image - mean) / std
    
        return new_image.unsqueeze(0)

    
        
        
