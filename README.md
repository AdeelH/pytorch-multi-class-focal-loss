[![DOI](https://zenodo.org/badge/292520399.svg)](https://zenodo.org/badge/latestdoi/292520399)

# Multi-class Focal Loss

An implementation of Focal Loss, as described in the RetinaNet paper, https://arxiv.org/abs/1708.02002, generalized to the multi-class case.

It is essentially an enhancement to cross-entropy loss and is useful for classification tasks when there is a large class imbalance. It has the effect of underweighting easy examples.

# Usage
- `FocalLoss` is an `nn.Module` and behaves very much like `nn.CrossEntropyLoss()` i.e.
    - supports the `reduction` and `ignore_index` params, and
    - is able to work with 2D inputs of shape `(N, C)` as well as K-dimensional inputs of shape `(N, C, d1, d2, ..., dK)`.

- Example usage
    ```python3
    focal_loss = FocalLoss(alpha, gamma)
	...
	inp, targets = batch
    out = model(inp)
	loss = focal_loss(out, targets)
    ```

# Loading through torch.hub
This repo supports importing modules through `torch.hub`. `FocalLoss` can be easily imported into your code via, for example:
```python3
focal_loss = torch.hub.load(
	'adeelh/pytorch-multi-class-focal-loss',
	model='FocalLoss',
	alpha=torch.tensor([.75, .25]),
	gamma=2,
	reduction='mean',
	force_reload=False
)
x, y = torch.randn(10, 2), (torch.rand(10) > .5).long()
loss = focal_loss(x, y)
```
Or:
```python3
focal_loss = torch.hub.load(
	'adeelh/pytorch-multi-class-focal-loss',
	model='focal_loss',
	alpha=[.75, .25],
	gamma=2,
	reduction='mean',
	device='cpu',
	dtype=torch.float32,
	force_reload=False
)
x, y = torch.randn(10, 2), (torch.rand(10) > .5).long()
loss = focal_loss(x, y)
```
