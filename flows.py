import torch
from torch.utils.data import Dataset, DataLoader
from torch.distributions.multivariate_normal import MultivariateNormal


class ObjectDataset(Dataset):
    def __init__(self, obj, num_of_data):
        self.data = torch.from_numpy(obj.sample(num_of_data)).float()

    def __getitem__(self, index):
        x = self.data[index]
        return x

    def __len__(self):
        return len(self.data)

    
class MLPLayers(torch.nn.Module):
    # A group of fully connected layers with adjustable layers size, with dropout
    def __init__(self, dims, dropout_rate):
        super().__init__()
        self.dims = dims
        self.dropout_layer = torch.nn.Dropout(p=dropout_rate)
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(dims[i], dims[i+1]) for i in range(len(dims)-1)]
        )
      
    def forward(self, X):
        # ReLU activations in hidden layers
        for i in range(len(self.dims)-2):
            X = self.dropout_layer(X)
            X = self.layers[i](X).clamp(min=0)
        # last layer no activation
        X = self.dropout_layer(X)
        outputs = self.layers[-1](X)
        return outputs


class MLPLayers_reparam(torch.nn.Module):
    # A group of fully connected layers with adjustable layers size, with reparameterisation
    def __init__(self, dims):
        super().__init__()
        
        mu_Ws = []
        mu_bs = []
        rho_Ws = []
        rho_bs = []
        
        for i in range(len(dims)-1):
            k1 = torch.tensor(1 / dims[i]).sqrt()
            k2 = -10
            mu_W = torch.nn.Parameter(k1 * (2 * torch.rand((dims[i], dims[i+1])) - 1))
            mu_b = torch.nn.Parameter(k1 * (2 * torch.rand(dims[i+1]) - 1))
            rho_W = torch.nn.Parameter(k2 * torch.rand_like(mu_W))
            rho_b = torch.nn.Parameter(k2 * torch.rand_like(mu_b))
            
            mu_Ws.append(mu_W)
            mu_bs.append(mu_b)
            rho_Ws.append(rho_W)
            rho_bs.append(rho_b)
        
        self.dims = dims
        self.mu_Ws = torch.nn.ParameterList(mu_Ws)
        self.mu_bs = torch.nn.ParameterList(mu_bs)
        self.rho_Ws = torch.nn.ParameterList(rho_Ws)
        self.rho_bs = torch.nn.ParameterList(rho_bs)
        
        self.sample_params()
      
    def forward(self, X):
        # forward propagation using existing parameters
        for i in range(len(self.dims)-2):
            X = torch.mm(X, self.Ws[i]) + self.bs[i]
            X = X.clamp(min=0)
        
        # last layer no activation
        return torch.mm(X, self.Ws[-1]) + self.bs[-1]
    
    def sample_params(self):
        epsilon_Ws = []
        epsilon_bs = []
        sigma_Ws = []
        sigma_bs = []
        Ws = []
        bs = []
        
        for i in range(len(self.dims)-1):
            mu_W = self.mu_Ws[i]
            mu_b = self.mu_bs[i]
            sigma_W = self._sigma(self.rho_Ws[i])
            sigma_b = self._sigma(self.rho_bs[i])
            epsilon_W = torch.randn_like(mu_W)
            epsilon_b = torch.randn_like(mu_b)
            #epsilon_W = torch.zeros_like(mu_W)
            #epsilon_b = torch.zeros_like(mu_b)
            
            # reparameterisation trick
            W =  mu_W + sigma_W * epsilon_W
            b =  mu_b + sigma_b * epsilon_b
            
            epsilon_Ws.append(epsilon_W)
            epsilon_bs.append(epsilon_b)
            sigma_Ws.append(sigma_W)
            sigma_bs.append(sigma_b)
            Ws.append(W)
            bs.append(b)
            
        self.epsilon_Ws = epsilon_Ws
        self.epsilon_bs = epsilon_bs
        self.sigma_Ws = sigma_Ws
        self.sigma_bs = sigma_bs
        self.Ws = Ws
        self.bs = bs
    
    def _sigma(self, rho):
        return torch.log(1 + torch.exp(rho))


class AdditiveCouplingLayers(torch.nn.Module):
    def __init__(self, d, D, layers, dropout_rate, var_inf=False):
        assert D > d
        
        super().__init__()
        
        dims1 = (d, 10, 10, D-d)
        dims2 = (D-d, 10, 10, d)
        
        MLP_list = []
        # Construct coupling layers in an alternating manner
        for layer in range(layers):
            if layer % 2 == 0:
                MLP_list.append(MLPLayers(dims1, dropout_rate))
            else:
                MLP_list.append(MLPLayers(dims2, dropout_rate))
        
        self.D = D
        self.d = d
        self.dims1 = dims1
        self.dims2 = dims2
        self.layers = layers
        self.MLP_list = torch.nn.ModuleList(MLP_list)
        if not var_inf:
            self.s = torch.nn.Parameter(torch.randn(D)) # scaling layer parameters

    def forward(self, Z):
        assert Z.shape[1] == self.D
        
        d = self.d
        layers = self.layers
        MLP_list = self.MLP_list
        s = self.s
        
        # combining coupling layers
        Z = torch.exp(-s) * Z
        
        Z_1d = Z[:, :d]
        Z_dD = Z[:, d:]
        for i in reversed(range(layers)):
            MLP = MLP_list[i]
            if i % 2 == 0:
                Z_dD = Z_dD - MLP(Z_1d)
            else:
                Z_1d = Z_1d - MLP(Z_dD)
            
        X = torch.cat((Z_1d, Z_dD), dim=1)
        
        log_det_dXdZ = -s.sum() * torch.ones(X.shape[0])
        
        return X, log_det_dXdZ
    
    def backward(self, X):
        assert X.shape[1] == self.D
        
        d = self.d
        layers = self.layers
        MLP_list = self.MLP_list
        s = self.s
        
        # combining coupling layers
        X_1d = X[:, :d]
        X_dD = X[:, d:]
        for i in range(layers):
            MLP = MLP_list[i]
            if i % 2 == 0:
                X_dD = X_dD + MLP(X_1d)
            else:
                X_1d = X_1d + MLP(X_dD)
        
        X = torch.cat((X_1d, X_dD), dim=1)
        Z = torch.exp(s) * X # scaling layer
        
        log_det_dZdX = s.sum() * torch.ones(X.shape[0])
        
        return Z, log_det_dZdX

    
class AffineCouplingLayers(torch.nn.Module):
    def __init__(self, masks, dropout_rate):
        super().__init__()
        
        D = masks.shape[1]
        dims = (D, 10, 10, D)
        
        scales = []
        translates = []
        
        # Construct coupling layers
        for i in range(len(masks)):
            scales.append(MLPLayers(dims, dropout_rate))
            translates.append(MLPLayers(dims, dropout_rate))
        
        self.D = D
        self.dims = dims
        self.masks = masks
        self.scales = torch.nn.ModuleList(scales)
        self.translates = torch.nn.ModuleList(translates)
        self.softplus = torch.nn.Softplus()
        
    def forward(self, Z):
        assert Z.shape[1] == self.D
        
        masks = self.masks
        scales = self.scales
        translates = self.translates
        
        # combining coupling layers
        log_det_dXdZ = 0
        for i in range(len(masks)):
            mask = masks[i]
            scale = scales[i]
            translate = translates[i]
            
            Z_masked = mask * Z
            s = scale(Z_masked)
            t = translate(Z_masked)
            softplus = self.softplus(s)
            Z = Z_masked + (~mask) * ((Z - t) / softplus)
            
            # calculate log absolute determinant
            log_det_dXdZ += torch.sum((~mask) * (torch.log(1 / softplus)), dim=1)
        
        return Z, log_det_dXdZ
    
    def backward(self, X):
        assert X.shape[1] == self.D
        
        masks = self.masks
        scales = self.scales
        translates = self.translates
        
        # combining coupling layers
        log_det_dZdX = 0
        for i in reversed(range(len(masks))):
            mask = masks[i]
            scale = scales[i]
            translate = translates[i]
            
            X_masked = mask * X
            s = scale(X_masked)
            t = translate(X_masked)
            softplus = self.softplus(s)
            X = X_masked + (~mask) * (X * softplus + t)
        
            # calculate log absolute determinant
            log_det_dZdX += torch.sum((~mask) * torch.log(softplus), dim=1)
            
        return X, log_det_dZdX


class Flow:
    def __init__(self, d, D, layers, dropout_rate=0, coupling_layers='affine'):
        if coupling_layers == 'affine':
            masks = self._create_masks(d, D, layers)
            self.cl = AffineCouplingLayers(masks, dropout_rate)
        elif coupling_layers == 'additive':
            self.cl = AdditiveCouplingLayers(d, D, layers, dropout_rate)
            
        self.prior_dist = MultivariateNormal(torch.zeros(D), torch.eye(D)) # isotropic unit norm Gaussian
    
    def _create_masks(self, d, D, layers):
        masks = []
        for layer in range(layers):
            if layer % 2 == 0:
                mask = [True for i in range(d)] + [False for i in range(D-d)]
            else:
                mask = [False for i in range(d)] + [True for i in range(D-d)]
            masks.append(mask)
        masks = torch.tensor(masks)
        return masks
    
    def forward(self, Z):
        with torch.no_grad():
            X, _ = self.cl(Z)
        return X
    
    def backward(self, X):
        with torch.no_grad():
            Z, _ = self.cl.backward(X)
        return Z
    
    def sample(self, n):
        Z = self.prior_dist.sample((n,))
        X = self.forward(Z)
        return X
    
    def log_likelihood(self, X, with_grad=False):
        if with_grad:
            Z, log_det_dZdX = self.cl.backward(X)
            log_pZ = self.prior_dist.log_prob(Z)
        else:
            with torch.no_grad():
                Z, log_det_dZdX = self.cl.backward(X)
                
                # set non finite values in Z to have -inf log probability
                idx = torch.isfinite(Z).all(dim=1)
                log_pZ = -torch.ones_like(log_det_dZdX) * float('inf')
                log_pZ[idx] = self.prior_dist.log_prob(Z[idx])
        
        return log_pZ + log_det_dZdX
    
    def log_data_likelihood(self, X, with_grad=False):
        log_pX = self.log_likelihood(X, with_grad)
        return log_pX.sum()
    
    def train(self, obj, sample_size, batch_size, epochs, lr, weight_decay=0, show_progress=False):
        optimizer = torch.optim.Adam(self.cl.parameters(), lr=lr, weight_decay=weight_decay)
        dataset = ObjectDataset(obj, sample_size)
        loader = DataLoader(dataset , batch_size=batch_size , shuffle=True)
        
        for epoch in range(epochs):
            for X_train in loader:
                neg_log_like = -self.log_data_likelihood(X_train, with_grad=True)

                optimizer.zero_grad()
                neg_log_like.backward()
                optimizer.step()

            # print status
            if show_progress:
                if epoch % 10 == 0:
                    # calculate and show log likelihood
                    X = torch.from_numpy(obj.sample(sample_size)).float()
                    log_like = self.log_data_likelihood(X)
                    print('epoch = %d, log_like = %.5f' % (epoch+1, log_like))


class AffineCouplingLayers_VI(AffineCouplingLayers):
    def __init__(self, masks):
        super().__init__(masks, dropout_rate=0)
        
        scales = []
        translates = []
        
        # Construct coupling layers
        for i in range(len(masks)):
            scales.append(MLPLayers_reparam(self.dims))
            translates.append(MLPLayers_reparam(self.dims))
        
        self.scales = torch.nn.ModuleList(scales)
        self.translates = torch.nn.ModuleList(translates)
    
    def sample_params(self):
        for scale, translate in zip(self.scales, self.translates):
            scale.sample_params()
            translate.sample_params()


class AdditiveCouplingLayers_VI(AdditiveCouplingLayers):
    def __init__(self, d, D, layers):
        super().__init__(d, D, layers, dropout_rate=0, var_inf=True)
        
        MLP_list = []
        # Construct coupling layers in an alternating manner
        for layer in range(layers):
            if layer % 2 == 0:
                MLP_list.append(MLPLayers_reparam(self.dims1))
            else:
                MLP_list.append(MLPLayers_reparam(self.dims2))
        
        mu_s = torch.nn.Parameter(torch.randn(D))
        rho_s = torch.nn.Parameter(torch.randn(D))
        sigma_s = self._sigma(rho_s)
        epsilon_s = torch.randn(D)
        s = mu_s + sigma_s * epsilon_s
        
        self.MLP_list = torch.nn.ModuleList(MLP_list)
        self.mu_s = mu_s
        self.rho_s = rho_s
        self.sigma_s = sigma_s
        self.s = s
    
    def sample_params(self):
        for MLP in self.MLP_list:
            MLP.sample_params()
        
        self.epsilon_s = torch.randn(self.D)
        self.sigma_s = self._sigma(self.rho_s)
        self.s = self.mu_s + self.sigma_s * self.epsilon_s
    
    def _sigma(self, rho):
        return torch.log(1 + torch.exp(rho))
        
        


class Flow_VI(Flow):
    def __init__(self, d, D, layers, coupling_layers='affine'):
        super().__init__(d, D, layers, dropout_rate=0, coupling_layers=coupling_layers)
        
        if coupling_layers == 'affine':
            masks = self._create_masks(d, D, layers)
            self.cl = AffineCouplingLayers_VI(masks)
        elif coupling_layers == 'additive':
            self.cl = AdditiveCouplingLayers_VI(d, D, layers)
    
    def train(self):
        pass
    
    def sample_params(self):
        self.cl.sample_params()
        
        
        
# class BatchNorm(torch.nn.Module):
#     def __init__(self, dim, eps=1e-05, momentum=0.1):
#         super().__init__()
#         self.momentum = momentum
#         self.mean = torch.zeros(dim)
#         self.var = torch.ones(dim)
#         self.eps = eps
        
#     def forward(self, X):
#         if self.training:
#             mom = self.momentum
#             self.mean = (1 - mom) * self.mean.detach() + mom * X.mean(dim=0)
#             self.var = (1 - mom) * self.var.detach() + mom * X.var(dim=0)
        
#         return (X - self.mean)/torch.sqrt(self.var + self.eps)
    
#     def backward(self, Z):
#         return Z * torch.sqrt(self.var + self.eps) + self.meanz