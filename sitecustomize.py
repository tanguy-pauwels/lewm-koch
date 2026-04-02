from __future__ import annotations

# Compatibility shim for older torchvision builds that don't expose
# torchvision.transforms.v2.RGB, while stable-pretraining expects it.

try:
    import inspect
    import torch
    import torch.utils._pytree as pytree
    from torchvision.transforms import v2
    from torchvision.transforms.v2 import Transform
    from torchvision.transforms.v2 import functional as F
except Exception:
    v2 = None
else:
    register_sig = inspect.signature(pytree.register_pytree_node)
    if 'flatten_with_keys_fn' not in register_sig.parameters:
        _orig_register_pytree_node = pytree.register_pytree_node

        def _compat_register_pytree_node(
            cls,
            flatten_fn,
            unflatten_fn,
            *args,
            flatten_with_keys_fn=None,
            **kwargs,
        ):
            return _orig_register_pytree_node(
                cls,
                flatten_fn,
                unflatten_fn,
                *args,
                **kwargs,
            )

        pytree.register_pytree_node = _compat_register_pytree_node

    for name in dir(v2):
        obj = getattr(v2, name, None)
        if inspect.isclass(obj) and hasattr(obj, '_transform') and not hasattr(obj, 'transform'):
            obj.transform = obj._transform

    if not hasattr(v2, 'RGB'):
        class RGB(Transform):
            def __init__(self) -> None:
                super().__init__()

            def _transform(self, inpt, params):
                if hasattr(F, 'grayscale_to_rgb'):
                    return self._call_kernel(F.grayscale_to_rgb, inpt)

                if torch.is_tensor(inpt):
                    if inpt.shape[-3] == 1:
                        return inpt.expand(*inpt.shape[:-3], 3, *inpt.shape[-2:])
                    return inpt

                if hasattr(inpt, 'convert'):
                    return inpt.convert('RGB')
                return inpt

        v2.RGB = RGB
