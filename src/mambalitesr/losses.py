import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights

class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Use modern weights API and eval mode
        vgg = vgg19(weights=VGG19_Weights.DEFAULT).features[:35].eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.mse = nn.MSELoss()
        
    def forward(self, input, target):
        # Ensure VGG is on the same device as inputs
        vgg_device = next(self.vgg.parameters()).device
        if vgg_device != input.device:
            self.vgg = self.vgg.to(input.device)
        # Use AMP to reduce memory during perceptual feature extraction
        with torch.amp.autocast('cuda', enabled=input.is_cuda):
            input_vgg = self.vgg(self.normalize(input))
            target_vgg = self.vgg(self.normalize(target.detach()))
        return self.mse(input_vgg, target_vgg)
    
    def normalize(self, x):
        return (x - 0.5) / 0.5  # Map from [0,1] to [-1,1]

class FeatureDistillationLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.1, gamma=0.01, temperature=1.0):
        """
        Comprehensive knowledge distillation loss
        Args:
            alpha: Weight for feature distillation
            beta: Weight for perceptual loss
            gamma: Weight for adversarial loss
            temperature: Softmax temperature for attention distillation
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.temp = temperature
        self.content_loss = nn.L1Loss()
        self.perceptual_loss = VGGPerceptualLoss() if beta > 0 else None
        
    def forward(self, student_sr, hr, teacher_features=None, student_features=None, discriminator=None):
        # Content loss (pixel-level)
        content_loss = self.content_loss(student_sr, hr)
        
        # Feature distillation loss
        feat_loss = 0
        if teacher_features and student_features:
            for t_feat, s_feat in zip(teacher_features, student_features):
                # Normalize features
                t_feat = F.normalize(t_feat, p=2, dim=1)
                s_feat = F.normalize(s_feat, p=2, dim=1)
                
                # Attention distillation
                t_att = F.softmax(t_feat.flatten(2).mean(-1) / self.temp, dim=1)
                s_att = F.log_softmax(s_feat.flatten(2).mean(-1) / self.temp, dim=1)
                feat_loss += F.kl_div(s_att, t_att.detach(), reduction='batchmean')
        
        # Perceptual loss
        perceptual_loss = self.perceptual_loss(student_sr, hr) if self.perceptual_loss else 0
        
        # Adversarial loss
        adv_loss = 0
        if self.gamma > 0 and discriminator:
            gen_pred = discriminator(student_sr)
            adv_loss = F.binary_cross_entropy_with_logits(
                gen_pred, torch.ones_like(gen_pred)
            )
        
        # Total loss
        total_loss = (1 - self.alpha - self.beta - self.gamma) * content_loss + \
                     self.alpha * feat_loss + \
                     self.beta * perceptual_loss + \
                     self.gamma * adv_loss
        
        return total_loss