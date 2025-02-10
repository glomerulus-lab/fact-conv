class LoadEnsemble(nn.Module):
    def __init__(self, net, load_dir):
        super().__init__()
        self.load_dir=load_dir

    def forward(self, x):
        # LOAD MODELS IN THE FORWARD PASS
        for sample in range(0, 10):
            load_dir = self.load_dir + "_ensemble{}".format(sample)
            sd = torch.load(load_dir+"/model.pt")
            net.load_state_dict(sd)
            net.forward(x)
