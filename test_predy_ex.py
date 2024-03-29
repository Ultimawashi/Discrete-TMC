import numpy as np
import json
from hmm import HMC_ctod, HTMC_ctod, MHTMC_ctod, GHTMC_ctod


def calc_mean_square_error(x, y, normalised=True):
    """Calcul de l'erreur quadratique moyenne"""
    coeff = 1
    if normalised:
        coeff = 1 / x.shape[0]
    return (coeff) * ((x - y) ** 2).sum()


nbc_x = 2
nbc_u = 3

N = 1000
steps=1
cmpnt = 0

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
    y_real = y[1:-1, cmpnt]
    for m in models:
        if m['name'] == 'hmc':
            m['model'].estim_param_sup(x,y)
        else:
            m['model'].estim_param_EM_semisup_x(x, y, 100)

        seg_y_sup = np.zeros(y_real.shape)
        for i in range(1, N - 1, steps):
            pred, forward_curr = m['model'].predict_mpm_y(y[:i], horizon=steps)
            seg_y_sup[i:(i + steps)] = pred[1:, cmpnt]
        res['gen_'+model['name']+'_seg_'+m['name'] + '_sup'] = calc_mean_square_error(y_real, seg_y_sup)

        m['model'].estim_param_EM(y, 100, V=True)

        seg_y_nsup = np.zeros(y_real.shape)
        for i in range(1, N - 1, steps):
            pred, forward_curr = m['model'].predict_mpm_y(y[:i], horizon=steps)
            seg_y_nsup[i:(i + steps)] = pred[1:, cmpnt]
        res['gen_' + model['name'] + '_seg_' + m['name'] + '_nsup'] = calc_mean_square_error(y_real, seg_y_nsup)

print(res)
