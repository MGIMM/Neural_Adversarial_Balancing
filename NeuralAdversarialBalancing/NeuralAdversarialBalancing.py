import torch
from torch import nn

class NeuralAdversarialBalancing():
    def __init__(self,
                 source_sample,
                 target_sample,
                 source_weight = None,
                 target_weight = None):
        """
        Attributes
        ----------
        source_sample: tensor
        Support of the source measure.

        target_sample: tensor
        target of the source measure.

        source_weight: tensor/np.array or None
        Weights of the source weighted empirical measure, i.e.,
        source measure = \frac{1}{n} \sum_{i=1}^{n} source_weight[i] \delta_{source_sample[i]}.
        When not assigned, the weight 1 is being implemented.

        target_weight: tensor/np.array or None
        Weights of the target weighted empirical measure, i.e.,
        $$
        target measure = \frac{1}{m} \sum_{j=1}^{m} target_weight[j] \delta_{target_sample[j]}.
        $$
        When not assigned, the weight 1 is being implemented.
        """
        # use GPU when possible
        if torch.cuda.is_available():
            dev = "cuda:0"
        else:
            dev = "cpu"
        self.dev = torch.device(dev)
        # xi denotes the source measure
        # xi_ring denotes the target measure
        with torch.no_grad():
            self.xi = source_sample.clone().to(self.dev)
            self.xi_ring = target_sample.clone().to(self.dev)
            if source_weight != None:
                self.w = source_weight.clone().to(self.dev)
            else:
                self.w = torch.ones(len(source_sample)).clone().to(self.dev)
            if target_weight != None:
                self.w_ring = target_weight.clone().to(self.dev)
            else:
                self.w_ring = torch.ones(len(target_sample)).to(self.dev)



    def train_loop(self,
                   model_IPM,
                  model_reweighting,
                  optimizer_IPM,
                  optimizer_reweighting,
                  IPM_steps = 1,
                  reweight_steps = 1,
                  lambda_l2_weight = 1e1,
                  lambda_l2_IPM = 1e-3
                  ):
        for t in range(IPM_steps):
            with torch.no_grad():
                weights = model_reweighting(self.xi).clone()
            weights.to(self.dev)
            mean_source = torch.mean(model_IPM(self.xi)*weights*self.w)
            mean_target = torch.mean(model_IPM(self.xi_ring)*self.w_ring)
            loss_IPM = -torch.abs(mean_source - mean_target)
            loss_IPM += lambda_l2_IPM * model_IPM(self.xi).square().mean()
            # Backpropagation
            optimizer_IPM.zero_grad()
            loss_IPM.backward()
            optimizer_IPM.step()
        # optimization for weight function estimation

        for t in range(reweight_steps):
            with torch.no_grad():
                mean_target_ = torch.mean(model_IPM(self.xi_ring)*self.w_ring).clone()
                values = model_IPM(self.xi).clone()
            mean_source_ =  torch.mean(values*model_reweighting(self.xi)*self.w)
            loss_reweighting = torch.abs(mean_source_ - mean_target_)

            loss_reweighting_reg = loss_reweighting + lambda_l2_weight * model_reweighting(self.xi).square().mean()

            optimizer_reweighting.zero_grad()
            loss_reweighting_reg.backward()
            optimizer_reweighting.step()
        return loss_reweighting

    def IPM_descent(self,
                   model_IPM,
                   model_reweighting,
                   optimizer_IPM,
                   optimizer_reweighting,
                   lambda_l2_weight = 1e1,
                   lambda_l2_IPM = 1e-3
                   ):
        with torch.no_grad():
            weights = model_reweighting(self.xi).clone()
        weights.to(self.dev)
        mean_source = torch.mean(model_IPM(self.xi)*weights*self.w)
        mean_target = torch.mean(model_IPM(self.xi_ring)*self.w_ring)
        loss_IPM = -torch.abs(mean_source - mean_target)
        loss_IPM_reg = loss_IPM + lambda_l2_IPM * model_IPM(self.xi).square().mean()
        # Backpropagation
        optimizer_IPM.zero_grad()
        loss_IPM.backward()
        optimizer_IPM.step()
        return -loss_IPM

    def reweighting_descent(self,
                            model_IPM,
                            model_reweighting,
                            optimizer_IPM,
                            optimizer_reweighting,
                            lambda_l2_weight = 1e1,
                            lambda_l2_IPM = 1e-3
                            ):
        with torch.no_grad():
            mean_target_ = torch.mean(model_IPM(self.xi_ring)*self.w_ring).clone()
            values = model_IPM(self.xi).clone()
        mean_source_ =  torch.mean(values*model_reweighting(self.xi)*self.w)
        loss_reweighting = torch.abs(mean_source_ - mean_target_)

        loss_reweighting_reg = loss_reweighting + lambda_l2_weight * model_reweighting(self.xi).square().mean()

        optimizer_reweighting.zero_grad()
        loss_reweighting_reg.backward()
        optimizer_reweighting.step()
        return loss_reweighting


class ParameterClipper(object):
    def __init__(self,
                 w_low = -2,
                 w_high = 2,
                 b_low = -2,
                 b_high = 2):
        self.w_low = w_low 
        self.w_high = w_high 
        self.b_low = b_low 
        self.b_high = b_high 

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'weight'):
            w = module.weight.data
            w = w.clamp(self.w_low,self.w_high)
            module.weight.data = w
        if hasattr(module, 'bias'):
            b = module.bias.data
            b = b.clamp(self.b_low,self.b_high)
            module.bias.data = b
