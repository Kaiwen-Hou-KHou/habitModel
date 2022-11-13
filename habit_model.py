import torch
import torch.nn as nn

class CCHabitModel(nn.Module):

    def __init__(self, s_lower, order, S_bar=.057, phi=.87, g=1.89/100, beta=.89, gamma=2):
        super(CCHabitModel, self).__init__()
        self.s_lower = s_lower
        # s_lower not set to be too negative to prevent overflows, and to prevent overweighing the tail errors
        self.order = order
        self.S_bar = S_bar
        self.phi = phi
        self.g = g
        self.beta = beta
        self.gamma = gamma
        self.linear = nn.Linear(order+1, order+1, bias=False)
        self.mask = torch.eye(order+1, dtype=bool)
        
    def forward(self, nu, save=False):
        
        batch_size, N = nu.size()
        self.linear.weight.data *= self.mask
        
        s_bar = torch.log(torch.tensor(self.S_bar))
        s_upper = s_bar + (1 - self.S_bar**2) / 2
        S = torch.linspace(torch.exp(torch.tensor(self.s_lower)), torch.exp(s_upper), N)
        s = torch.log(S)

        x = 2 * (s - self.s_lower)/(s_upper - self.s_lower) - 1
        cheb = torch.special.chebyshev_polynomial_t(x.unsqueeze(-1), torch.arange(self.order+1))
        output = self.linear(cheb)
        assert output.size() == (N, self.order+1)
        v = output.sum(axis=-1) # N
        if save:
            torch.save(v, 'PC.pt')
            torch.save(S, 'S.pt')

        lam = 1 / self.S_bar * torch.sqrt(1 - 2*(s-s_bar)) - 1
        s_prime = (1 - self.phi) * s_bar + self.phi * s + lam * nu # batch_size * N
        s_prime = torch.clamp(s_prime, self.s_lower, s_upper)
        x = 2 * (s_prime - self.s_lower)/(s_upper - self.s_lower) - 1
        cheb = torch.special.chebyshev_polynomial_t(x.unsqueeze(-1), torch.arange(self.order+1)) 
        output = self.linear(cheb) 
        assert output.size() == (batch_size, N, self.order+1)
        v_prime = output.sum(axis=-1) # batch_size * N
        
        M = self.beta * torch.exp(-self.gamma * (s_prime - s + self.g + nu)) # batch_size * N
        assert (M == torch.inf).sum() == 0 # check no overflows
        diff = M * torch.exp(self.g + nu) * (v_prime + 1) - v # batch_size * N
        
        return diff.mean(axis=0) # N