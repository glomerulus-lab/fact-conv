import torch
import torch.nn as nn

def rms(x):
    x = x**2
    x = torch.mean(x)
    return torch.sqrt(x).detach()


def forwards_hook(name):
    def full_forwards_hook(module, x, output):
        torch.set_printoptions(sci_mode=True, precision=10)
        print(name, "Tri1_vec RMS:", rms(module.tri1_vec))
        print(name, "Tri2_vec RMS:", rms(module.tri2_vec))
    return full_forwards_hook

def backwards_hook(name):
    def full_backwards_hook(grad):
        torch.set_printoptions(sci_mode=True, precision=10)
        print(name, "RMS:", rms(grad))
    return full_backwards_hook


def wandb_forwards_hook(name, store):
    def full_forwards_hook(module, x, output):
        torch.set_printoptions(sci_mode=True, precision=10)
        key1 = name + " Tri1_vec_RMS"
        store[key1] = rms(module.tri1_vec)#.item()
        key2 = name + " Tri2_vec_RMS"
        store[key2] = rms(module.tri2_vec)#.item()
    return full_forwards_hook

def wandb_backwards_hook(name, store):
    def full_backwards_hook(grad):
        torch.set_printoptions(sci_mode=True, precision=10)
        key = name + " RMS"
        store[key] = rms(grad)#.item()
    return full_backwards_hook


