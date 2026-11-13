"""Run-to-run determinism for the LLaVA-3D eval WITHOUT editing LLaVA-3D's files (2026-10-15).

Active only when this directory is on PYTHONPATH and DETERMINISTIC=1, e.g.
    DETERMINISTIC=1 PYTHONPATH=/workspace/3dzip/determinism bash scripts/eval/eval_fig4.sh ...
(the eval scripts append their own PYTHONPATH after it).

Sources of run-to-run variation it removes (found 2026-10-15, results/sparsevlm_determinism/):
 1. llava_arch.py caps a scene at 2560 voxel tokens with an unseeded torch.randperm (subset AND order change every
    run) and SparseVLM's density tie-break uses an unseeded torch.rand. -> every LlavaLlamaForCausalLM.generate call
    is preceded by torch.manual_seed(DET_SEED + crc32(input_ids)), so a question always draws the same subset /
    order whatever ran before it. The cap itself (random 2560 of the voxels) is LLaVA-3D's behaviour and is kept.
 2. torch_scatter.scatter_mean / scatter_add / scatter_sum use CUDA atomics (summation order varies -> bf16 voxel
    features and xyz differ by a few ulp -> pruning decisions flip). -> replaced by a sort + float64 cumulative-sum
    segment reduction (deterministic; result equal to the exact sum up to the final cast to the input dtype).
 3. CUBLAS_WORKSPACE_CONFIG=:4096:8, torch.use_deterministic_algorithms(True, warn_only=True), no TF32 /
    cudnn benchmark (residual LLM-kernel variation seen at SparseVLM's last, 2-3-token stages).
Nothing else changes; with DETERMINISTIC unset this module does nothing.
"""
import os
import sys

if os.environ.get('DETERMINISTIC', '0') == '1':
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import zlib

    import torch
    import torch_scatter

    torch.use_deterministic_algorithms(True, warn_only=True)   # deterministic kernels where PyTorch has them
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    _orig = {n: getattr(torch_scatter, n) for n in ('scatter_mean', 'scatter_add', 'scatter_sum')}

    def _segment_sum(src, index, dim, dim_size):
        d = dim % src.dim()
        x = src.movedim(d, 0)
        idx = index if index.dim() == 1 else index.movedim(d, 0).reshape(index.shape[d], -1)[:, 0]
        n = int(idx.max()) + 1 if dim_size is None and idx.numel() else (dim_size or 0)
        order = torch.argsort(idx, stable=True)
        xs = x[order].double().reshape(x.shape[0], -1)
        cs = torch.cat([xs.new_zeros(1, xs.shape[1]), xs.cumsum(0)], 0)
        cnt = torch.bincount(idx, minlength=n)[:n]
        ends = cnt.cumsum(0)
        sums = cs[ends] - cs[ends - cnt]
        return sums.reshape((n,) + x.shape[1:]), cnt, d

    def _ok(src, index, dim, out):
        if out is not None or not torch.is_floating_point(src):
            return False
        d = dim % src.dim()
        if index.dim() == 1:
            return index.shape[0] == src.shape[d]
        # broadcast-style index (same value along the non-scatter dims)
        return index.shape == src.shape

    def _make(name):
        def f(src, index, dim=-1, out=None, dim_size=None, **kw):
            if kw or not _ok(src, index, dim, out):
                return _orig[name](src, index, dim=dim, out=out, dim_size=dim_size, **kw)
            sums, cnt, d = _segment_sum(src, index, dim, dim_size)
            if name == 'scatter_mean':
                c = cnt.clamp(min=1).double().reshape((-1,) + (1,) * (sums.dim() - 1))
                sums = sums / c
            return sums.to(src.dtype).movedim(0, d)
        f.__name__ = name
        return f

    for _n in _orig:
        setattr(torch_scatter, _n, _make(_n))

    # per-question seeding of LlavaLlamaForCausalLM.generate, applied when that module is first imported
    import importlib.abc
    import importlib.machinery

    _TARGET = 'llava.model.language_model.llava_llama'
    _BASE = int(os.environ.get('DET_SEED', '0'))

    def _wrap(mod):
        # every class of the module that defines its own generate (LlavaLlamaForCausalLM, and e.g. AdaToken-3D's
        # LlavaLlamaForCausalLM_PDrop subclass, which overrides generate)
        for cls in [c for c in vars(mod).values() if isinstance(c, type) and 'generate' in vars(c)]:
            gen = vars(cls)['generate']

            def generate(self, inputs=None, *a, __gen=gen, **kw):
                key = inputs.detach().cpu().numpy().tobytes() if torch.is_tensor(inputs) else b''
                torch.manual_seed(_BASE + zlib.crc32(key))
                return __gen(self, inputs, *a, **kw)
            generate.__wrapped__ = gen
            cls.generate = generate

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname != _TARGET:
                return None
            sys.meta_path.remove(self)
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            sys.meta_path.insert(0, self)
            if spec is None:
                return None
            loader = spec.loader
            exec_module = loader.exec_module

            def exec_and_wrap(module):
                exec_module(module)
                _wrap(module)
                sys.meta_path[:] = [m for m in sys.meta_path if not isinstance(m, _Finder)]
            loader.exec_module = exec_and_wrap
            return spec

    sys.meta_path.insert(0, _Finder())
    print('[determinism] torch_scatter scatter_{mean,add,sum} -> deterministic; per-question seed on generate', file=sys.stderr)
