import torch
import torch.nn as nn
from torch.utils.interpolate import spline_basis



# built with OPUS
class KANLayer(nn.Module):
    def __init__(self, in_features, out_features, num_knots, spline_degree):
        super(KANLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_knots = num_knots
        self.spline_degree = spline_degree
        
        self.spline_coeffs = nn.Parameter(torch.randn(in_features, out_features, num_knots))
        self.spline_knots = nn.Parameter(torch.linspace(0, 1, num_knots, requires_grad=False))
        
    def forward(self, x):
        batch_size = x.shape[0]
        x = x.unsqueeze(1).unsqueeze(2)  # Add dimensions for out_features and num_knots
        
        # Evaluate B-spline basis functions
        basis = spline_basis(self.spline_knots, self.spline_degree, x, extrapolate=False)
        
        # Multiply basis functions with spline coefficients
        output = (basis * self.spline_coeffs).sum(dim=-1)
        
        return output.sum(dim=1)  # Sum over in_features dimension

class KAN(nn.Module):
    def __init__(self, layer_shapes, num_knots, spline_degree):
        super(KAN, self).__init__()
        self.layers = nn.ModuleList()
        
        for i in range(len(layer_shapes) - 1):
            in_features = layer_shapes[i]
            out_features = layer_shapes[i + 1]
            self.layers.append(KANLayer(in_features, out_features, num_knots, spline_degree))
            
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x