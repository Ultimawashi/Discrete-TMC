import numpy as np
import json
from hmm import HMC_ctod, HTMC_ctod, MHTMC_ctod, GHTMC_ctod


def calc_err(ref_im, seg_im):
    terr = np.sum(seg_im != ref_im) / np.prod(ref_im.shape)
    return (terr <= 0.5) * terr + (terr > 0.5) * (1 - terr)


nbc_x = 2
nbc_u = 3

N = 1000
steps=1

res = {}
models = [
    {'name': 'hmc', 'model': HMC_ctod(nbc_x), 'params': {'p':[0.5,0.5], 't':[[0.9,0.1],
              [0.1,0.9]], 'mu': [
    [20, 100],
    [10, 50]
], 'sigma':[
    [[2,3],
     [3,10]],
    [[2,3],
     [3,10]]
]}},
    {'name': 'htmc', 'model': HTMC_ctod(nbc_x, nbc_u), 'params': {'p':[0.3,0.1,0.2,0.1,0.1,0.2], 't':[[0.5,0.1,0.1,0.1,0.1,0.1],
                   [0.1,0.5,0.1,0.1,0.1,0.1],
                   [0.1,0.1,0.5,0.1,0.1,0.1],
                   [0.1,0.1,0.1,0.5,0.1,0.1],
                   [0.1,0.1,0.1,0.1,0.5,0.1],
                   [0.1,0.1,0.1,0.1,0.1,0.5]
                   ], 'mu': [
    [20, 100],
    [10, 50]
], 'sigma':[
    [[2,3],
     [3,10]],
    [[2,3],
     [3,10]]
]}},
    {'name': 'mhtmc', 'model': MHTMC_ctod(nbc_x, nbc_u), 'params': {'p':[0.3,0.1,0.2,0.1,0.1,0.2], 't':[[0.5,0.1,0.1,0.1,0.1,0.1],
                   [0.1,0.5,0.1,0.1,0.1,0.1],
                   [0.1,0.1,0.5,0.1,0.1,0.1],
                   [0.1,0.1,0.1,0.5,0.1,0.1],
                   [0.1,0.1,0.1,0.1,0.5,0.1],
                   [0.1,0.1,0.1,0.1,0.1,0.5]
                   ], 'mu': [
    [20, 100],
    [10, 50]
], 'sigma':[
    [[2,3],
     [3,10]],
    [[2,3],
     [3,10]]
]}},
    {'name': 'ghtmc', 'model': GHTMC_ctod(nbc_x, nbc_u), 'params': {'p':[0.3,0.1,0.2,0.1,0.1,0.2], 't':[[0.5,0.1,0.1,0.1,0.1,0.1],
                   [0.1,0.5,0.1,0.1,0.1,0.1],
                   [0.1,0.1,0.5,0.1,0.1,0.1],
                   [0.1,0.1,0.1,0.5,0.1,0.1],
                   [0.1,0.1,0.1,0.1,0.5,0.1],
                   [0.1,0.1,0.1,0.1,0.1,0.5]
                   ], 'mu': [
    [20, 100],
    [10, 50]
], 'sigma':[
    [[2,3],
     [3,10]],
    [[2,3],
     [3,10]]
]}},
    ]

for model in models:
    model['model'].give_param(model['params'])
    x, y = model['model'].generate_sample(N)
    x_real = x[1:-1]
    for m in models:
        if m['name'] == 'hmc':
            m['model'].estim_param_sup(x,y)
        else:
            m['model'].estim_param_EM_semisup_x(x, y, 100)

        seg_x_sup = np.zeros(x_real.shape, dtype=int)
        for i in range(1, N - 1, steps):
            seg_x_sup[i:(i + steps)] = m['model'].predict_mpm(y[:i], horizon=steps)[1:]
        res['gen_'+model['name']+'_seg_'+m['name'] + '_sup'] = calc_err(x_real, seg_x_sup)

        m['model'].estim_param_EM(y, 100, V=True)

        seg_x_nsup = np.zeros(x_real.shape, dtype=int)
        for i in range(1, N - 1, steps):
            seg_x_nsup[i:(i + steps)] = m['model'].predict_mpm(y[:i], horizon=steps)[1:]
        res['gen_' + model['name'] + '_seg_' + m['name'] + '_nsup'] = calc_err(x_real, seg_x_nsup)

print(res)
