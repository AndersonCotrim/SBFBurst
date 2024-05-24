def put_requires_grad(layer, req=False):
    for name, param in layer.named_parameters():
        param.requires_grad = req
