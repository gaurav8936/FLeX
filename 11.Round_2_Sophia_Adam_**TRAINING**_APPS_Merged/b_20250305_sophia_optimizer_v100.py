import math
import torch
from torch.optim import Optimizer
from typing import List, Optional, Tuple, Dict, Any, Callable


class SophiaG(Optimizer):
    """
    Implements Sophia optimizer with Gauss-Newton-Bartlett (GNB) Hessian estimator.
    
    Based on the paper:
    "Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training"
    by Liu et al. (https://arxiv.org/abs/2305.14342)
    
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): learning rate (default: 1e-4)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its second moment (default: (0.965, 0.99))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-12)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 1e-1)
        k (int, optional): frequency of Hessian estimation (default: 10)
        gamma (float, optional): factor for Hessian scaling and clipping (default: 0.01)
    """

    def __init__(self, params, lr=1e-4, betas=(0.965, 0.99), eps=1e-12, 
                 weight_decay=1e-1, k=10, gamma=0.01):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not isinstance(k, int) or k <= 0:
            raise ValueError(f"Invalid k parameter: {k}, must be a positive integer")
        if not 0.0 <= gamma:
            raise ValueError(f"Invalid gamma parameter: {gamma}")
            
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, 
                        k=k, gamma=gamma)
        super(SophiaG, self).__init__(params, defaults)
        
        # Initialize global step counter for Hessian updates
        self.global_state = {"step": 0}
        
    def step(self, closure=None, gnb_kwargs=None):
        """
        Performs a single optimization step.
        
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            gnb_kwargs (dict, optional): Additional kwargs for the GNB Hessian estimator.
                Should include model and a mini-batch tensor.
        """
        loss = None
        if closure is not None:
            loss = closure()
            
        # Increment global step counter
        self.global_state["step"] += 1
        
        # Check if we need to update Hessian at this step
        update_hessian = (self.global_state["step"] % self.defaults["k"] == 1)
        
        # If we need to update Hessian and no gnb_kwargs provided, use identity fallback
        if update_hessian and gnb_kwargs is None and self.global_state["step"] > 1:
            # Create fallback identity Hessian
            gnb_kwargs = {"hessian_estimates": {}}
            # Set to small constant for all parameters (no need to fill it here)
        
        # Track clipping statistics
        self.global_state["total_params"] = 0
        self.global_state["clipped_params"] = 0
        
        # Track Hessian statistics
        if update_hessian and gnb_kwargs is not None and "hessian_estimates" in gnb_kwargs:
            self.global_state["hessian_sum"] = 0.0
            self.global_state["hessian_count"] = 0
            for h_id, h_val in gnb_kwargs["hessian_estimates"].items():
                self.global_state["hessian_sum"] += torch.mean(h_val).item()
                self.global_state["hessian_count"] += 1
            
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                    
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Sophia does not support sparse gradients")
                    
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Momentum buffer (EMA of gradients)
                    state["m"] = torch.zeros_like(p.data)
                    # Hessian buffer (EMA of diagonal Hessian estimates)
                    state["h"] = torch.zeros_like(p.data)
                    
                # Get hyperparameters for this group
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                gamma = group["gamma"]
                
                # Update step count
                state["step"] += 1
                
                # Decay the first and second moment running averages
                state["m"].mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # Update diagonal Hessian estimate if needed
                if update_hessian and gnb_kwargs is not None:
                    # Get the GNB Hessian estimate for this parameter
                    if hasattr(p, "_optim_id") and gnb_kwargs.get("hessian_estimates") and p._optim_id in gnb_kwargs["hessian_estimates"]:
                        h_estimate = gnb_kwargs["hessian_estimates"][p._optim_id]
                    else:
                        # If specific estimate not provided, use small constant
                        h_estimate = torch.ones_like(p.data) * 1e-4
                    
                    # Update Hessian EMA
                    if state["step"] == 1:
                        state["h"].copy_(h_estimate)
                    else:
                        state["h"].mul_(beta2).add_(h_estimate, alpha=1 - beta2)
                
                # Perform weight decay
                if group["weight_decay"] > 0:
                    p.data.mul_(1 - group["lr"] * group["weight_decay"])
                
                # Compute update: clipped momentum / (gamma * Hessian + eps)
                h_scaled = torch.clamp(gamma * state["h"], min=eps)
                raw_update = state["m"] / h_scaled
                
                # Element-wise clipping of the update to [-1, 1]
                clipped_update = torch.clamp(raw_update, -1.0, 1.0)
                
                # Track clipping statistics
                self.global_state["total_params"] += clipped_update.numel()
                self.global_state["clipped_params"] += torch.sum(torch.abs(raw_update) >= 1.0).item()
                
                # Store update for potential analysis
                state["raw_update"] = raw_update
                state["clipped_update"] = clipped_update
                
                # Apply update
                p.data.add_(clipped_update, alpha=-group["lr"])
                
        return loss
        
    def get_clipping_stats(self):
        """
        Returns the fraction of parameters that were clipped in the last update.
        
        Returns:
            float: Fraction of parameters that were clipped (0.0 to 1.0)
        """
        if "total_params" in self.global_state and self.global_state["total_params"] > 0:
            return self.global_state["clipped_params"] / self.global_state["total_params"]
        return 0.0
    
    def get_hessian_stats(self):
        """
        Returns the average magnitude of Hessian diagonal entries from the last update.
        
        Returns:
            float: Average Hessian diagonal magnitude, or None if not available
        """
        if "hessian_count" in self.global_state and self.global_state["hessian_count"] > 0:
            return self.global_state["hessian_sum"] / self.global_state["hessian_count"]
        return None    
    
    def add_param_optimizer_id(self):
        """
        Assigns a unique ID to each parameter in the optimizer, useful for mapping
        parameters to their Hessian estimates.
        """
        param_id = 0
        for group in self.param_groups:
            for p in group["params"]:
                p._optim_id = param_id
                param_id += 1
                
    def compute_gnb_hessian(self, model, inputs, labels=None, batch_size=32):
        """
        Computes Gauss-Newton-Bartlett Hessian estimate for all parameters.
        
        Arguments:
            model: The model 
            inputs: Mini-batch of inputs
            labels: If None, samples labels from model output
            batch_size: Batch size for estimation
            
        Returns:
            Dict mapping parameter IDs to their Hessian estimates
        """
        # Make sure parameters have IDs
        self.add_param_optimizer_id()
        
        # Store original training mode and switch to eval for Hessian computation
        training = model.training
        model.eval()
        
        hessian_estimates = {}
        
        with torch.no_grad():
            # Get model output
            outputs = model(inputs)
            
            # If no labels provided, sample from model outputs (softmax distribution)
            if labels is None:
                probs = torch.nn.functional.softmax(outputs, dim=-1)
                # Sample new labels from the model's output distribution
                sampled_labels = torch.multinomial(probs, 1).squeeze(-1)
            else:
                sampled_labels = labels
                
            # Compute loss with sampled labels
            loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
            loss = loss_fn(outputs, sampled_labels)
            
            # Compute gradients for this loss
            model.zero_grad()
        
        # Compute gradients
        loss.backward(retain_graph=True)
        
        # Now collect squared gradients as GNB Hessian estimates
        # Multiply by batch size as per the algorithm
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    # GNB = B * g^2 where g is the grad from sampled labels
                    hessian_estimates[p._optim_id] = batch_size * (p.grad.data ** 2)
                else:
                    hessian_estimates[p._optim_id] = torch.zeros_like(p.data)
                    
        # Reset gradients and restore training mode
        model.zero_grad()
        model.train(training)
        
        return hessian_estimates



        
def estimate_lora_hessian(model, batch, tokenizer, lora_layers_only=True):
    """
    Helper function to estimate Hessian for LoRA layers in a model.
    
    Arguments:
        model: The model with LoRA layers
        batch: Batch of data
        tokenizer: Tokenizer for processing inputs
        lora_layers_only: If True, only compute for LoRA parameters
        
    Returns:
        Dictionary of Hessian estimates for each parameter
    """
    inputs = tokenizer(batch['text'], return_tensors='pt', padding=True, truncation=True).to(model.device)
    
    # Forward pass to get logits
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Sample from the model's distribution
        probs = torch.nn.functional.softmax(logits[:, -1, :], dim=-1)
        sampled_next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
    
    # Zero gradients
    model.zero_grad()
    
    # Compute loss with sampled tokens
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = torch.cat([inputs['input_ids'][:, 1:], sampled_next_tokens.unsqueeze(-1)], dim=1)[:, :-1].contiguous()
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    # Use reshape instead of view to handle non-contiguous tensors
    loss = loss_fn(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
    
    # Compute gradients
    loss.backward()
    
    # Get Hessian estimates
    hessian_estimates = {}
    batch_size = inputs['input_ids'].size(0)
    
    for name, param in model.named_parameters():
        # Only compute for LoRA parameters if specified
        if lora_layers_only and 'lora' not in name:
            continue
            
        if hasattr(param, '_optim_id') and param.grad is not None:
            # GNB Hessian estimate: B * (grad)^2
            hessian_estimates[param._optim_id] = batch_size * (param.grad.data ** 2)
        elif hasattr(param, '_optim_id'):
            # If parameter has no gradient, use a small positive value
            hessian_estimates[param._optim_id] = torch.ones_like(param.data) * 1e-4
    
    # Clear gradients
    model.zero_grad()
    
    return hessian_estimates