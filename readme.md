
```python
# Define the KAN model
class KANNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, knot_count=5, order=3, extra_knots=2):
        super().__init__()
        self.input_layer = SplineActivation(input_size, hidden_size, knot_count, order, extra_knots)
        self.hidden_layer = SplineActivation(hidden_size, output_size, knot_count, order, extra_knots)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layer(x)
        return x
```
