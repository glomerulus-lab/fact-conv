import torch
import torch.nn as nn

import copy
from pytorch_cifar_utils import  set_seeds

from conv_modules import FactConv2d
#traditional way of calculating svd. can be a bit unstable sometimes tho
def calc_svd(A, name=''):
    u, s, vh = torch.linalg.svd(A, full_matrices=False)  # (C_in_reference, R), (R,), (R, C_in_generated)
    alignment = u  @ vh  # (C_in_reference, C_in_generated)
    return alignment, (u, s, vh)
 

#used in activation cross-covariance calculation
#input align hook
def return_hook():
    def hook(mod, inputs):
        shape = inputs[0].shape
        inputs_permute = inputs[0].permute(1,0,2,3).reshape(inputs[0].shape[1], -1)
        reshape = (mod.input_align@inputs_permute).reshape(shape[1],
                shape[0], shape[2],
                shape[3]).permute(1, 0, 2, 3)
        return reshape 
    return hook


#settings
# our, wa true, aca true (fact, conv)
# our wa false, aca true (fact, conv)
# our wa false aca false (fact, conv) [Just Random]
# our was true aca false (fact, conv)
#
# theirs wa true, aca true (conv)
# theirs wa false, aca true (conv)
# theirs wa false aca false (conv) [Just Random]
# theirs was true aca false (conv)
class rainbow_sampler:
    def __init__(self, net, net_new, args, device, trainloader, num_classes=10):
        self.net = copy.deepcopy(net)
        self.net_new = copy.deepcopy(net_new)
        self.seed = args.seed
        self.sampling = args.sampling
        self.wa = args.wa
        self.in_wa = args.in_wa
        self.aca = args.aca 
        self.device = device
        self.trainloader = trainloader
        self.num_classes = num_classes

    def do_rainbow_sampling(self):
        set_seeds(self.seed)
        self.net.train()
        self.net_new = copy.deepcopy(self.net)
        self.net_new.train()
        print("With seed {}".format(self.seed))
        self.our_rainbow_sampling(self.net, self.net_new)

    def load_state_dicts(self, m1, m2, new_module):
        # reference model state dict
        ref_sd = m1.state_dict()
        # generated model state dict - uses reference model weights. for now
        gen_sd = m2.state_dict()
        
        # module with random init - to be loaded to model
        loading_sd = new_module.state_dict()
        
        # carry over old bias. only matters when we work with no batchnorm networks
        if m1.bias is not None:
            loading_sd['bias'] = ref_sd['bias']
        # carry over old colored covariance. only matters with fact-convs
        if "tri1_vec" in ref_sd.keys():
            loading_sd['tri1_vec'] = ref_sd['tri1_vec']
            loading_sd['tri2_vec'] = ref_sd['tri2_vec']
        
        return ref_sd, gen_sd, loading_sd

    @torch.no_grad()
    def our_rainbow_sampling(self, model, new_model):
        for (n1, m1), (n2, m2) in zip(model.named_children(), new_model.named_children()):
            if len(list(m1.children())) > 0:
                self.our_rainbow_sampling(m1, m2)
            if isinstance(m1, nn.Conv2d):
                if isinstance(m2, FactConv2d):
                    new_module = FactConv2d(
                        in_channels=m2.in_channels,
                        out_channels=m2.out_channels,
                        kernel_size=m2.kernel_size,
                        stride=m2.stride, padding=m2.padding, 
                        bias=True if m2.bias is not None else
                        False).to(self.device)
                else:
                    new_module = nn.Conv2d(
                        in_channels=int(m2.in_channels),
                        out_channels=int(m2.out_channels),
                        kernel_size=m2.kernel_size,
                        stride=m2.stride, padding=m2.padding, 
                        groups = m2.groups,
                        bias=True if m2.bias is not None else False).to(self.device)
    
                if self.sampling == 'structured_alignment' and self.wa:
                    # right now this function does not do an explicit specification of the colored covariance
                    new_module = self.weight_Alignment(m1, m2, new_module, in_dim=self.in_wa)
                if self.sampling == 'cc_specification':
                    # for conv only
                    if self.wa:
                        new_module = self.weight_Alignment_With_CC(m1, m2, new_module)
                    else:
                        new_module = self.colored_Covariance_Specification(m1, m2, new_module)
                # this step calculates the activation cross-covariance alignment (ACA)
                if m1.in_channels != 3 and self.aca:
                    new_module = self.conv_ACA(m1, m2, new_module)
                # changes the network module
                setattr(new_model, n1, new_module)
    
            #only computes the ACA
            if isinstance(m1, nn.Linear) and self.aca:
                new_module = self.linear_ACA(m1, m2, new_model)
                setattr(new_model, n1, new_module)
            ##just run stats through
            if isinstance(m1, nn.BatchNorm2d):
                self.batchNorm_stats_recalc(m1, m2)

    def run_forward(self, calc_covar=False):
        if calc_covar:
            covar = None
            total = 0
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            try:
                outputs1 = self.net(inputs)
            except Exception:
                pass
            try:
                outputs2 = self.net_new(inputs)
            except Exception:
                pass
            if calc_covar:
                total+= inputs.shape[0]
                if covar is None:
                    #activation is bwh x c
                    covar = self.activation[0].T @ self.activation[1]
                    assert (covar.isfinite().all())
                else: 
                    #activation is bwh x c
                    covar += self.activation[0].T @ self.activation[1]
                    assert (covar.isfinite().all())
                self.activation = []
        if calc_covar:
            #c x c
            covar /= total
            return covar

    def conv_ACA(self, m1, m2, new_module):
        print("Convolutional Input Activations Alignment")
        self.activation = []
        # this hook grabs the input activations of the conv layer
        # rearanges the vector so that the width by height dim is 
        # additional samples to the covariance
        # bwh x c
        def define_hook(m):
            def store_hook(mod, inputs, outputs):
                #from bonner lab tutorial
                x = inputs[0]
                x = x.permute(0, 2, 3, 1)
                x = x.reshape((-1, x.shape[-1]))
                self.activation.append(x)
                raise Exception("Done")
            return store_hook
        hook_handle_1 = m1.register_forward_hook(define_hook(m1))
        hook_handle_2 = m2.register_forward_hook(define_hook(m2))
        print("Starting Sample Cross-Covariance Calculation")
        covar = self.run_forward(calc_covar=True)
        print("Sample Cross-Covariance Calculation finished")
        hook_handle_1.remove()
        hook_handle_2.remove()
        align, _ = calc_svd(covar, name="Cross-Covariance")
        new_module.register_buffer("input_align", align)
        # this hook takes the input to the conv, aligns, then returns
        # to the conv the aligned inputs
        hook_handle_pre_forward  = new_module.register_forward_pre_hook(return_hook())
        return new_module
    
    def batchNorm_stats_recalc(self, m1, m2):
        print("Calculating Batch Statistics")
        m1.train()
        m2.train()
        m1.reset_running_stats()
        m2.reset_running_stats()
        handle_1 = m1.register_forward_hook(lambda mod, inputs, outputs: Exception("Done"))
        handle_2 = m2.register_forward_hook(lambda mod, inputs, outputs: Exception("Done"))
        m1.to(self.device)
        m2.to(self.device)
        self.run_forward(calc_covar=False)
        handle_1.remove()
        handle_2.remove()
        m1.eval()
        m2.eval()
        print("Batch Statistics Calculation Finished")
    
    def linear_ACA(self, m1, m2, new_model):
        print("Linear Input Activations Alignment")
        new_module = nn.Linear(m1.in_features, m1.out_features, bias=True
                if m1.bias is not None else False).to(self.device)
        ref_sd = m1.state_dict()
        loading_sd = new_module.state_dict()
        if m1.out_features == self.num_classes:
            loading_sd['weight'] = ref_sd['weight']
        if m1.bias is not None:
            loading_sd['bias'] = ref_sd['bias']
        self.activation = []
        hook_handle_1 = m1.register_forward_hook(lambda mod, inputs, outputs:
                self.activation.append(inputs[0]))
        hook_handle_2 = m2.register_forward_hook(lambda mod, inputs, outputs:
                self.activation.append(inputs[0]))
        print("Starting Sample Cross-Covariance Calculation")
        covar = self.run_forward(calc_covar=True)
        hook_handle_1.remove()
        hook_handle_2.remove()
        print("Sample Cross-Covariance Calculation finished")
        align, _ = calc_svd(covar, name="Cross-Covariance")
        new_weight = loading_sd['weight']
        new_weight = torch.moveaxis(new_weight, source=1,
                destination=-1)
        new_weight = new_weight@align
        loading_sd['weight'] = torch.moveaxis(new_weight, source=-1,
                destination=1)
        new_module.load_state_dict(loading_sd)
        return new_module

    # IF FACT: we align the generated factnet with the reference fact net's noise
    # IF CONV: we align the generated convnet with the reference conv net's weight matrix
    def weight_Alignment(self, m1, m2, new_module, in_dim=True):   
        ref_sd, gen_sd, loading_sd = self.load_state_dicts(m1, m2, new_module)
        reference_weight = ref_sd['weight']
        generated_weight = loading_sd['weight']
        
        #reshape to outdim x indim*spatial
        reference_weight = reference_weight.reshape(reference_weight.shape[0], -1)
        generated_weight = generated_weight.reshape(generated_weight.shape[0], -1)
        
        #compute weight cross-covariance indim*spatial x indim*spatial
        #TODO REFACTOR TO HAVE REF FIRST. OUTDIM x OUTDIM 
        if in_dim:
            print("Input Weight Alignment")
            weight_cov = (generated_weight.T@reference_weight)
            alignment, _ = calc_svd(weight_cov, name="Weight alignment")
            
            # outdim x indim*spatial
            final_gen_weight = generated_weight@alignment
        else:
            print("Output Weight Alignment")
            weight_cov = (reference_weight@generated_weight.T)
            alignment, _ = calc_svd(weight_cov, name="Weight alignment")
            
            # outdim x indim*spatial
            final_gen_weight = alignment@generated_weight
    
        loading_sd['weight_align'] = alignment
        new_module.register_buffer("weight_align", alignment)

        loading_sd['weight'] = final_gen_weight.reshape(ref_sd['weight'].shape)
        new_module.load_state_dict(loading_sd)
        return new_module

    def weight_Alignment_With_CC(self, m1, m2, new_module):
        print("Weight alignment with Colored Covariance")
        ref_sd, gen_sd, loading_sd = self.load_state_dicts(m1, m2, new_module)
   
        old_weight = ref_sd['weight']
        A = old_weight.reshape(old_weight.shape[0], -1)

        _, (Un, Sn, Vn) = calc_svd(A)
        white_gaussian = torch.randn_like(Un)
    
        copy_weight = Un 
        copy_weight_gen = white_gaussian
        copy_weight = copy_weight.reshape(copy_weight.shape[0], -1)
        copy_weight_gen = copy_weight_gen.reshape(copy_weight_gen.shape[0], -1).T
        weight_cov = (copy_weight_gen@copy_weight)
    
        alignment, _ = calc_svd(weight_cov, name="Weight")
        new_weight = white_gaussian  
        new_weight = new_weight.reshape(new_weight.shape[0], -1)
        new_weight = new_weight@alignment # C_in_reference to C_in_generated
    
        new_module.register_buffer("weight_align", alignment)
        loading_sd['weight_align'] = alignment
        colored_gaussian = white_gaussian @ (Sn[:,None]* Vn)

        loading_sd['weight'] = colored_gaussian.reshape(old_weight.shape)
        new_module.load_state_dict(loading_sd)
        return new_module
    
    # this function does not do an explicit specification of the colored covariance
    @torch.no_grad()
    def colored_Covariance_Specification(self, m1, m2, new_module):
        print("Colored Covariance")
        ref_sd, gen_sd, loading_sd = self.load_state_dicts(m1, m2, new_module)
   
        old_weight = ref_sd['weight']
        A = old_weight.reshape(old_weight.shape[0], -1)

        _, (Un, Sn, Vn) = calc_svd(A)
        white_gaussian = torch.randn_like(Un)
        colored_gaussian = white_gaussian @ (Sn[:,None]* Vn)

        loading_sd['weight'] = colored_gaussian.reshape(old_weight.shape)
        new_module.load_state_dict(loading_sd)
        return new_module


