import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal
import json


def np_multivariate_normal_pdf(x, mu, cov):
    # if mu.shape[0] != x.shape[0]:
    #     broadc = (len(mu.shape) - len(x.shape) + 1)
    #     x = x.reshape((x.shape[0],) + (1,) * broadc + x.shape[1:])
    # else:
    #     broadc = (len(mu.shape) - len(x.shape))
    #     x = x.reshape((x.shape[0],) + (1,) * broadc + x.shape[1:])
    broadc = (len(mu.shape) - 1)
    x = x.reshape(x.shape[:-1] + (1,) * broadc + (x.shape[-1],))
    part1 = 1 / (((2 * np.pi) ** (mu.shape[-1] / 2)) * (np.linalg.det(cov) ** (1 / 2)))
    part2 = (-1 / 2) * np.einsum('...j,...j',np.einsum('...j,...ji',(x - mu),np.linalg.inv(cov)),(x - mu))
    return part1 * np.exp(part2)


def np_multivariate_normal_pdf_marginal(x, mu, cov, j, i=0):
    # if mu.shape[0] != x.shape[0]:
    #     broadc = (len(mu.shape) - len(x.shape) + 1)
    #     x = x.reshape((x.shape[0],) + (1,) * broadc + x.shape[1:])
    # else:
    #     broadc = (len(mu.shape) - len(x.shape))
    #     x = x.reshape((x.shape[0],) + (1,) * broadc + x.shape[1:])
    broadc = (len(mu.shape) - len(x.shape) + 1)
    x = x.reshape((x.shape[0],) + (1,) * broadc + x.shape[1:])
    part1 = 1 / (((2 * np.pi) ** (mu[...,i:j+1].shape[-1] / 2)) * (np.linalg.det(cov[...,i:j+1,i:j+1]) ** (1 / 2)))
    part2 = (-1 / 2) * np.einsum('...j,...j',np.einsum('...j,...ji',(x - mu[...,i:j+1]),np.linalg.inv(cov[...,i:j+1,i:j+1])),(x - mu[...,i:j+1]))
    return part1 * np.exp(part2)


def convertcls_vect(cls, rand_vect_param):
    aux = cls
    res=np.zeros((len(rand_vect_param)))
    for i in reversed(range(len(rand_vect_param))):
        res[len(rand_vect_param) - i - 1] = aux % rand_vect_param[i]
        aux = aux // rand_vect_param[len(rand_vect_param) - i - 1]
    return res


def convert_multcls_vectors(data, rand_vect_param):
    classes = range(np.max(data).astype(int) + 1)
    assert (len(classes) <= np.prod(rand_vect_param)), 'Les paramètres du vecteur aléatoire ne correspondent pas'
    res = np.zeros(data.shape + (len(rand_vect_param),))
    aux = [convertcls_vect(cls, rand_vect_param) for cls in classes]
    for c in classes:
        res[data==c] = aux[c]
    return res.astype('int')


class HMC_ctod:

    __slots__ = ('p', 't', 'mu', 'sigma', 'nbc_x', 'reg')

    def __init__(self, nbc_x, p=None, t=None, mu=None, sigma=None, reg=10**-10):
        self.p = p
        self.t = t
        self.mu = mu
        self.sigma = sigma
        self.nbc_x = nbc_x
        self.reg = reg

    def save_param_to_json(self, filepath):
        param_s = {'p': self.p.tolist(), 't': self.t.tolist(), 'mu': self.mu.tolist(), 'sigma': self.mu.tolist()}
        with open(filepath,'w') as f:
            json.dump(param_s, f, ensure_ascii=False)

    def load_param_from_json(self, filepath):
        with open(filepath, 'r') as f:
            params = json.load(f)
        self.give_param(params)

    def give_param(self, params):
        self.p = np.array(params['p'])
        self.t = np.array(params['t'])
        self.mu = np.array(params['mu'])
        self.sigma = np.array(params['sigma'])

    def generate_sample(self, length):
        x = np.zeros(length, dtype=int)
        y = np.zeros((length, self.mu.shape[-1]))
        T = self.t
        test = np.random.multinomial(1, self.p)
        x[0] = np.argmax(test)
        y[0] = multivariate_normal.rvs(self.mu[x[0]], self.sigma[x[0]])
        for i in range(1, length):
            test = np.random.multinomial(1, T[x[i - 1], :])
            x[i] = np.argmax(test)
            y[i] = multivariate_normal.rvs(self.mu[x[i]], self.sigma[x[i]])
        return x, y

    def get_gaussians(self, y):
        mask_data = np.isnan(y)
        y[mask_data] = 0
        gausses = np_multivariate_normal_pdf(y, self.mu, self.sigma)
        gausses[np.any(mask_data, axis=1)] = 1
        return gausses

    def calc_proba_apost(self, y):
        gaussians = self.get_gaussians(y)
        forward = self.get_forward(gaussians)
        backward = self.get_backward(gaussians)
        p_apost = forward * backward
        p_apost = p_apost / (p_apost.sum(axis=1)[..., np.newaxis])
        return p_apost

    def calc_forward_nobs(self,y, horizon):
        T = self.t
        gaussians = self.get_gaussians(y)
        forward_curr = np.zeros((horizon + 1, T.shape[0]))
        forward_curr[0] = self.get_forward(gaussians)[-1]
        for l in range(1, horizon + 1):
            forward_curr[l] = (forward_curr[l - 1] @ T)
            forward_curr[l] = forward_curr[l] / forward_curr[l].sum()
        return forward_curr

    def seg_mpm(self, y):
        p_apost = self.calc_proba_apost(y)
        return np.argmax(p_apost, axis=1)

    def get_forward(self, gaussians):
        P = self.p
        T = self.t
        forward = np.zeros((len(gaussians), P.shape[0]))
        forward[0] = P * gaussians[0]
        for l in range(1, len(gaussians)):
            forward[l] = (gaussians[l]) * (forward[l - 1] @ T)
            forward[l] = forward[l] / (forward[l].sum() + self.reg)
        return forward

    def get_backward(self, gaussians):
        T = self.t
        backward = np.zeros((len(gaussians), T.shape[0]))
        backward[len(gaussians) - 1] = np.ones(T.shape[0])
        for l in reversed(range(0, len(gaussians)-1)):
            backward[l] = ((gaussians[l + 1]) * backward[l + 1]) @ T.T
            backward[l] = backward[l] / (backward[l].sum() + self.reg)
        return backward

    def predict_mpm(self, y, horizon=1):
        forward_curr = self.calc_forward_nobs(y, horizon)
        return np.argmax(forward_curr, axis=1)

    def predict_mpm_y(self, y, horizon=1):
        forward_curr = self.calc_forward_nobs(y, horizon)
        pred = (forward_curr[...,np.newaxis]*self.mu[np.newaxis,...]).sum(axis=1)
        return pred, forward_curr

    def estim_p_t_sup(self, x):
        xprime = np.stack((x[:-1], x[1:]), axis=-1)
        aux = np.moveaxis(np.indices((self.nbc_x, self.nbc_x)), 0, -1)
        c = (1 / (len(xprime) - 1)) * (
            np.all(xprime[:, np.newaxis, np.newaxis, ...] == aux[np.newaxis, ...], axis=-1).sum(
                axis=0))
        p = (1 / (len(x))) * (x[..., np.newaxis] == np.indices((self.nbc_x,))).sum(axis=0) + self.reg
        p = p / p.sum()
        t = (c.T / p).T
        return p, t

    def estim_moy_var_sup(self, x, y):
        mu = (((x[..., np.newaxis] == np.indices((self.nbc_x,)))[..., np.newaxis] * y[:, np.newaxis,
                                                                                    ...]).sum(axis=0) /
              (
                      x[..., np.newaxis] == np.indices((self.nbc_x,))).sum(axis=0)[
                  ..., np.newaxis])
        sigma = (((x[..., np.newaxis] == np.indices((self.nbc_x,))).reshape(
            (x.shape[0], self.nbc_x))[..., np.newaxis, np.newaxis] * np.einsum('...i,...j',
                                                                               (y[:, np.newaxis, ...] -
                                                                                mu[np.newaxis, ...]),
                                                                               (y[:, np.newaxis, ...] -
                                                                                mu[
                                                                                    np.newaxis, ...]))).sum(
            axis=0)
                 / ((x[..., np.newaxis] == np.indices((self.nbc_x,))).sum(
                    axis=0)).reshape((self.nbc_x,))[..., np.newaxis, np.newaxis])
        return mu, sigma

    def calc_param_sup(self, x, y):
        p, t = self.estim_p_t_sup(x)
        mu, sigma = self.estim_moy_var_sup(x, y)
        return p, t, mu, sigma

    def estim_param_sup(self, x, y):
        self.p, self.t, self.mu, self.sigma = self.calc_param_sup(x, y)
        print({'p':self.p, 't':self.t, 'mu':self.mu, 'sigma':self.sigma})

    def estim_param_sup_db(self, x, y):
        test = [k for i in range(x.shape[0]) for k in self.calc_param_sup(x[i], y[i])]
        self.p = (1/len(test[::4]))*np.array(sum(test[::4]))
        self.t = (1/len(test[1::4]))*np.array(sum(test[1::4]))
        self.mu = (1 / len(test[2::4])) * np.array(sum(test[2::4]))
        self.sigma = (1 / len(test[3::4])) * np.array(sum(test[3::4]))
        print({'p': self.p, 't': self.t, 'mu':self.mu, 'sigma':self.sigma})

    def init_param(self, y):
        kmeans = KMeans(n_clusters=self.nbc_x).fit(y)
        x = kmeans.labels_
        p, t, mu, sigma = self.calc_param_sup(x, y)
        return p, t, mu, sigma

    def calc_psi_gamma(self, y):
        gaussians = self.get_gaussians(y)
        forward = self.get_forward(gaussians)
        backward = self.get_backward(gaussians)
        T = self.t
        gamma = (
                forward[:-1, :, np.newaxis]
                * (gaussians[1:, np.newaxis, :]
                   * backward[1:, np.newaxis, :]
                   * T[np.newaxis, :, :])
        )
        gamma = gamma / (gamma.sum(axis=(1, 2))[..., np.newaxis, np.newaxis])
        psi = forward * backward
        psi = psi / (psi.sum(axis=1)[..., np.newaxis])
        return gamma, psi

    def estim_moy_var_EM(self, psi, y):
        mu = (((psi[..., np.newaxis] * y[:, np.newaxis, ...]).sum(axis=0)) / (
            psi.sum(axis=0)[..., np.newaxis])).reshape(self.mu.shape)

        sigma = (psi.reshape((psi.shape[0], self.mu.shape[0]))[..., np.newaxis, np.newaxis] * np.einsum(
            '...i,...j',
            (y[:, np.newaxis, ...] - mu[np.newaxis, ...]),
            (y[:, np.newaxis, ...] -
             mu[np.newaxis, ...])
        )).sum(
            axis=0) / (psi.sum(axis=0)[..., np.newaxis, np.newaxis])
        return mu, sigma

    def estim_p_t_EM(self, psi, gamma):
        p = (psi.sum(axis=0)) / psi.shape[0] + self.reg
        c = gamma.sum(axis=0) / gamma.shape[0]
        p = p / p.sum()
        t = (c.T / p).T
        return p, t

    def calc_param_EM(self, y):
        gamma, psi = self.calc_psi_gamma(y)
        p, t = self.estim_p_t_EM(psi, gamma)
        mu, sigma = self.estim_moy_var_EM(psi, y)
        return p, t, mu, sigma

    def estim_param_EM(self, y, iter=100, V=False):
        self.p, self.t, self.mu, self.sigma = self.init_param(y)
        if V:
            print({'iter': 0, 'p': self.p, 't': self.t, 'mu':self.mu, 'sigma':self.sigma})
        for q in range(iter):
            self.p, self.t, self.mu, self.sigma = self.calc_param_EM(y)
            if V:
                print({'iter': q + 1, 'p': self.p, 't': self.t, 'mu':self.mu, 'sigma':self.sigma})

    def estim_param_EM_db(self, y_db, iter=100, V=False):
        test = [k for i in range(y_db.shape[0]) for k in self.init_param(y_db[i])]
        self.p = (1 / len(test[::4])) * np.array(sum(test[::4]))
        self.t = (1 / len(test[1::4])) * np.array(sum(test[1::4]))
        self.mu = (1 / len(test[2::4])) * np.array(sum(test[2::4]))
        self.sigma = (1 / len(test[3::4])) * np.array(sum(test[3::4]))
        if V:
            print({'iter': 0, 'p': self.p, 't': self.t, 'mu':self.mu, 'sigma':self.sigma})
        for q in range(iter):
            test = [k for i in range(y_db.shape[0]) for k in self.calc_param_EM(y_db[i])]
            self.p = (1 / len(test[::4])) * np.array(sum(test[::4]))
            self.t = (1 / len(test[1::4])) * np.array(sum(test[1::4]))
            self.mu = (1 / len(test[2::4])) * np.array(sum(test[2::4]))
            self.sigma = (1 / len(test[3::4])) * np.array(sum(test[3::4]))
            if V:
                print({'iter': q+1, 'p': self.p, 't': self.t, 'mu':self.mu, 'sigma':self.sigma})


class HTMC_ctod(HMC_ctod):

    __slots__ = ('p', 't', 'mu', 'sigma', 'nbc_x', 'nbc_u', 'reg')

    def __init__(self, nbc_x, nbc_u, p=None, t=None, mu=None, sigma=None, reg=10**-10):
        HMC_ctod.__init__(self, nbc_x, p, t, mu, sigma, reg)
        self.nbc_u = nbc_u

    def get_param_form(self, cu, px, tx):
        c = (cu[np.newaxis, :, np.newaxis, :] * (
                px[..., np.newaxis, np.newaxis] * px[np.newaxis, np.newaxis, ...])).reshape(self.nbc_x * self.nbc_u,
                                                                                            self.nbc_x * self.nbc_u)
        p = np.sum(c, axis=1) + self.reg
        p = p / p.sum()
        t = (c.T / p).T
        return p, t

    def generate_sample(self, length, x_only=True):
        xu = np.zeros(length, dtype=int)
        y = np.zeros((length, self.mu.shape[-1]))
        T = self.t
        mu = np.repeat(self.mu, self.nbc_u, axis=0)
        sigma = np.repeat(self.sigma, self.nbc_u, axis=0)
        test = np.random.multinomial(1, self.p)
        xu[0] = np.argmax(test)
        y[0] = multivariate_normal.rvs(mu[xu[0]], sigma[xu[0]])
        for i in range(1, length):
            test = np.random.multinomial(1, T[xu[i - 1], :])
            xu[i] = np.argmax(test)
            y[i] = multivariate_normal.rvs(mu[xu[i]], sigma[xu[i]])
        if x_only:
            xu = convert_multcls_vectors(xu, (self.nbc_u, self.nbc_x))[:, 1]
        return xu, y

    def get_gaussians(self, y):
        mu = np.repeat(self.mu, self.nbc_u, axis=0)
        sigma = np.repeat(self.sigma, self.nbc_u, axis=0)
        mask_data = np.isnan(y)
        y[mask_data] = 0
        gausses = np_multivariate_normal_pdf(y, mu, sigma)
        gausses[np.any(mask_data, axis=1)] = 1
        return gausses

    def seg_mpm_x(self, y):
        p_apost = self.calc_proba_apost(y)
        p_apost_x = (p_apost.reshape((p_apost.shape[0], self.nbc_x, self.nbc_u))).sum(axis=2)
        return np.argmax(p_apost_x, axis=1)

    def seg_mpm_u(self, y):
        p_apost = self.calc_proba_apost(y)
        p_apost_u = (p_apost.reshape((p_apost.shape[0], self.nbc_x, self.nbc_u))).sum(axis=1)
        return np.argmax(p_apost_u, axis=1)

    def get_sup_x_forward(self, x):
        P = self.p.reshape(self.nbc_x, self.nbc_u)
        T = self.t.reshape(self.nbc_x, self.nbc_u, self.nbc_x, self.nbc_u)
        forward = np.zeros((len(x), self.nbc_u))
        forward[0] = P[x[0], :]
        for l in range(1, len(x)):
            forward[l] = (forward[l - 1] @ T[x[l-1], :, x[l], :])
            forward[l] = forward[l] / (forward[l].sum() + self.reg)
        return forward

    def get_sup_x_backward(self, x):
        T = self.t.reshape(self.nbc_x, self.nbc_u, self.nbc_x, self.nbc_u)
        backward = np.zeros((len(x), self.nbc_u))
        backward[len(x) - 1] = np.ones(self.nbc_u)
        for l in reversed(range(0, len(x)-1)):
            backward[l] = (backward[l + 1]) @ T[x[l], :, x[l+1], :].T
            backward[l] = backward[l] / (backward[l].sum() + self.reg)
        return backward

    def get_sup_u_forward(self, u, gaussians):
        P = self.p.reshape(self.nbc_x, self.nbc_u)
        T = self.t.reshape(self.nbc_x, self.nbc_u, self.nbc_x, self.nbc_u)
        forward = np.zeros((len(gaussians), P.shape[0]))
        forward[0] = P[:, u[0]] * gaussians[0]
        for l in range(1, len(gaussians)):
            forward[l] = (gaussians[l]) * (forward[l - 1] @ T[:, u[l-1], :, u[l]])
            forward[l] = forward[l] / (forward[l].sum() + self.reg)
        return forward

    def get_sup_u_backward(self, u, gaussians):
        T = self.t.reshape(self.nbc_x, self.nbc_u, self.nbc_x, self.nbc_u)
        backward = np.zeros((len(gaussians), self.nbc_x))
        backward[len(gaussians) - 1] = np.ones(self.nbc_x)
        for l in reversed(range(0, len(gaussians) - 1)):
            backward[l] = (backward[l + 1]) @ T[:, u[l], :, u[l + 1]].T
            backward[l] = backward[l] / (backward[l].sum() + self.reg)

    def predict_mpm_x(self, y, horizon=1):
        forward_curr = self.calc_forward_nobs(y, horizon)
        forward_curr_x = (forward_curr.reshape((forward_curr.shape[0], self.nbc_x, self.nbc_u))).sum(axis=2)
        return np.argmax(forward_curr_x, axis=1)

    def predict_mpm_u(self, y, horizon=1):
        forward_curr = self.calc_forward_nobs(y, horizon)
        forward_curr_u = (forward_curr.reshape((forward_curr.shape[0], self.nbc_x, self.nbc_u))).sum(axis=1)
        return np.argmax(forward_curr_u, axis=1)

    def predict_mpm_y(self, y, horizon=1):
        mu = np.repeat(self.mu, self.nbc_u, axis=0)
        sigma = np.repeat(self.sigma, self.nbc_u, axis=0)
        forward_curr = self.calc_forward_nobs(y, horizon)
        pred = (forward_curr[...,np.newaxis]*mu[np.newaxis,...]).sum(axis=1)
        return pred, forward_curr

    def init_param_semisup_x(self, x, y):
        p, t, mu, sigma = self.calc_param_sup(x, y)
        c=(t.T*p).T
        cf = np.block([[c * (1 / (self.nbc_u ** 2)) for i in range(self.nbc_u)] for j in range(self.nbc_u)])
        p = cf.sum(axis=1) + self.reg
        p = p / p.sum()
        t = (cf.T / p).T
        return p, t, mu, sigma

    def init_param_semisup_u(self, u, y):
        p, t, mu, sigma = HMC_ctod.init_param(self, y)
        pprime, tprime = self.estim_p_t_sup(u)
        cprime = (tprime.T*pprime).T
        c=(t.T*p).T
        cf = np.block([[c * cprime[i,j] for i in range(self.nbc_u)] for j in range(self.nbc_u)])
        p = cf.sum(axis=1) + self.reg
        p = p / p.sum()
        t = (cf.T / p).T
        return p, t, mu, sigma

    def init_param(self, y):
        p, t, mu, sigma = HMC_ctod.init_param(self, y)
        c = (t.T * p).T
        cf = np.block([[c * (1 / (self.nbc_u ** 2)) for i in range(self.nbc_u)] for j in range(self.nbc_u)])
        p = cf.sum(axis=1) + self.reg
        p = p / p.sum()
        t = (cf.T / p).T
        return p, t, mu, sigma

    def estim_moy_var_EM(self, psi, y):
        psi = (psi.reshape((psi.shape[0], self.nbc_x, self.nbc_u))).sum(axis=2)
        mu, sigma = HMC_ctod.estim_moy_var_EM(self, psi, y)
        return mu, sigma

    def estim_p_t_EM(self, psi, gamma):
        gamma = (gamma.sum(axis=0) / gamma.shape[0]).reshape(self.nbc_x, self.nbc_u, self.nbc_x, self.nbc_u)
        gammaprime = gamma.sum(axis=(0, 2))
        p = ((psi.sum(axis=0)) / psi.shape[0]).reshape(self.nbc_x, self.nbc_u) + self.reg
        px = p / p.sum(axis=0)[np.newaxis, :]
        p, t = self.get_param_form(gammaprime, px, None)
        return p, t

    def calc_psi_gamma_semisup_x(self, x):
        T = self.t.reshape(self.nbc_x, self.nbc_u, self.nbc_x, self.nbc_u)
        forward = self.get_sup_x_forward(x)
        backward = self.get_sup_x_backward(x)
        gamma = (
                        forward[:-1, :, np.newaxis]
                        * (backward[1:, np.newaxis, :]
                           * T[x[:-1], :, x[1:], :])
                ) + self.reg
        gamma = gamma / (gamma.sum(axis=(1, 2))[..., np.newaxis, np.newaxis] + self.reg)
        psi = forward * backward
        psi = psi / (psi.sum(axis=1)[..., np.newaxis])
        return gamma, psi

    def estim_p_t_EM_semisup_x(self, x, psi, gamma):
        gamma = (gamma.sum(axis=0)) / gamma.shape[0]
        px = (((psi[:, np.newaxis, :] * (x[:, np.newaxis] == np.indices((self.nbc_x,)))[..., np.newaxis]).sum(
            axis=0)) / (
                      psi.sum(axis=0)[np.newaxis, ...] + self.reg))
        p, t = self.get_param_form(gamma, px, None)
        return p, t

    def calc_param_EM_semisup_x(self, x):
        gamma, psi = self.calc_psi_gamma_semisup_x(x)
        p, t = self.estim_p_t_EM_semisup_x(x, psi, gamma)
        return p, t

    def estim_param_EM_semisup_x(self, x, y, iter=100, V=False):
        self.p, self.t, self.mu, self.sigma = self.init_param_semisup_x(x, y)
        if V:
            print({'iter': 0, 'p': self.p, 't': self.t, 'mu':self.mu, 'sigma':self.sigma})
        for q in range(iter):
            self.p, self.t = self.calc_param_EM_semisup_x(x)
            if V:
                print({'iter': q+1, 'p': self.p, 't': self.t})

    def estim_param_EM_semisup_x_db(self, x_db, y_db, iter=100, V=False):
        test = [k for i in range(y_db.shape[0]) for k in  self.init_param_semisup_x(x_db[i], y_db[i])]
        self.p = (1 / len(test[::4])) * np.array(sum(test[::4]))
        self.t = (1 / len(test[1::4])) * np.array(sum(test[1::4]))
        self.mu = (1 / len(test[2::4])) * np.array(sum(test[2::4]))
        self.sigma = (1 / len(test[3::4])) * np.array(sum(test[3::4]))
        if V:
            print({'iter': 0, 'p': self.p, 't': self.t, 'mu':self.mu, 'sigma':self.sigma})
        for q in range(iter):
            test = [k for i in range(y_db.shape[0]) for k in self.calc_param_EM_semisup_x(x_db[i])]
            self.p = (1 / len(test[::2])) * np.array(sum(test[::2]))
            self.t = (1 / len(test[1::2])) * np.array(sum(test[1::2]))
            if V:
                print({'iter': q+1, 'p': self.p, 't': self.t})

    def calc_psi_gamma_semisup_u(self, u, y):
        gaussians = self.get_gaussians(y)
        T = self.t.reshape(self.nbc_x, self.nbc_u, self.nbc_x, self.nbc_u)
        forward = self.get_sup_u_forward(u, gaussians)
        backward = self.get_sup_u_backward(u, gaussians)
        gamma = (
                        forward[:-1, :, np.newaxis]
                        * (gaussians[1:, np.newaxis, :]*
                           backward[1:, np.newaxis, :]
                           * T[:, u[:-1], :, u[1:]])
                ) + self.reg
        gamma = gamma / (gamma.sum(axis=(1, 2))[..., np.newaxis, np.newaxis] + self.reg)
        psi = forward * backward
        psi = psi / (psi.sum(axis=1)[..., np.newaxis])
        return gamma, psi

    def estim_p_t_EM_semisup_u(self, u, psi, gamma):
        pprime, tprime = self.estim_p_t_sup(u)
        cprime = (tprime.T * pprime).T
        px = (psi.sum(axis=0)) / psi.shape[0] + self.reg
        p, t = self.get_param_form(cprime, px, None)
        return p, t

    def calc_param_EM_semisup_u(self, u, y):
        gamma, psi = self.calc_psi_gamma_semisup_u(u,y)
        p, t = self.estim_p_t_EM_semisup_u(u, psi, gamma)
        mu, sigma = HMC_ctod.estim_moy_var_EM(self, psi, y)
        return p, t, mu, sigma

    def estim_param_EM_semisup_u(self, u, y, iter=100, V=False):
        self.p, self.t, self.mu, self.sigma = self.init_param_semisup_u(u, y)
        if V:
            print({'iter': 0, 'p': self.p, 't': self.t, 'mu':self.mu, 'sigma':self.sigma})
        for q in range(iter):
            self.p, self.t, self.mu, self.sigma = self.calc_param_EM_semisup_u(u, y)
            if V:
                print({'iter': q+1, 'p': self.p, 't': self.t, 'mu':self.mu, 'sigma':self.sigma})

    def estim_param_EM_semisup_u_db(self, u_db, y_db, iter=100, V=False):
        test = [k for i in range(y_db.shape[0]) for k in self.init_param_semisup_u(u_db[i], y_db[i])]
        self.p = (1 / len(test[::4])) * np.array(sum(test[::4]))
        self.t = (1 / len(test[1::4])) * np.array(sum(test[1::4]))
        self.mu = (1 / len(test[2::4])) * np.array(sum(test[2::4]))
        self.sigma = (1 / len(test[3::4])) * np.array(sum(test[3::4]))
        if V:
            print({'iter': 0, 'p': self.p, 't': self.t, 'mu':self.mu, 'sigma':self.sigma})
        for q in range(iter):
            test = [k for i in range(y_db.shape[0]) for k in self.calc_param_EM_semisup_u(u_db[i], y_db[i])]
            self.p = (1 / len(test[::4])) * np.array(sum(test[::4]))
            self.t = (1 / len(test[1::4])) * np.array(sum(test[1::4]))
            self.mu = (1 / len(test[2::4])) * np.array(sum(test[2::4]))
            self.sigma = (1 / len(test[3::4])) * np.array(sum(test[3::4]))
            if V:
                print({'iter': q+1, 'p': self.p, 't': self.t, 'mu':self.mu, 'sigma':self.sigma})


class MHTMC_ctod(HTMC_ctod):

    def get_param_form(self, cu, px, tx):
        c = (cu[np.newaxis,:,np.newaxis,:]*tx[:,np.newaxis,:, :]*(px[..., np.newaxis, np.newaxis])).reshape(self.nbc_x*self.nbc_u, self.nbc_x*self.nbc_u)
        p = np.sum(c, axis=1) + self.reg
        p = p / p.sum()
        t = (c.T / p).T
        return p, t

    def estim_p_t_EM(self, psi, gamma):
        gamma = (gamma.sum(axis=0) / gamma.shape[0]).reshape(self.nbc_x, self.nbc_u, self.nbc_x, self.nbc_u)
        gammaprime = gamma.sum(axis=(0, 2))
        tx = gamma.sum(axis=1)
        pxprime = tx.sum(axis=1) + self.reg
        pxprime = pxprime / pxprime.sum(axis=0)[np.newaxis, ...]
        tx = tx / pxprime[:, np.newaxis, :]
        p = ((psi.sum(axis=0)) / psi.shape[0]).reshape(self.nbc_x, self.nbc_u)
        px = p / p.sum(axis=0)[np.newaxis, :]
        p, t = self.get_param_form(gammaprime, px, tx)
        return p, t

    def estim_p_t_EM_semisup_x(self, x, psi, gamma):
        gamma = (gamma.sum(axis=0) / gamma.shape[0])
        px = (((psi[:, np.newaxis, :] * (x[:, np.newaxis] == np.indices((self.nbc_x,)))[..., np.newaxis]).sum(
            axis=0)) / (
                      psi.sum(axis=0)[np.newaxis, ...] + self.reg))
        xprime = np.stack((x[:-1], x[1:]), axis=-1)
        aux = np.moveaxis(np.indices((self.nbc_x, self.nbc_x)), 0, -1)
        tx = (((psi[1:, np.newaxis, np.newaxis, :] * (np.all(
            xprime[:, np.newaxis, np.newaxis, ...] == aux[np.newaxis, ...], axis=-1))[:, :, :, np.newaxis]).sum(
            axis=0)) / (
                      psi[1:].sum(axis=0)[np.newaxis, np.newaxis, :] + self.reg))
        pxprime = tx.sum(axis=1) + self.reg
        pxprime = pxprime / pxprime.sum(axis=0)[np.newaxis, ...]
        tx = tx / pxprime[:, np.newaxis, :]
        p, t = self.get_param_form(gamma, px, tx)
        return p, t

    def estim_p_t_EM_semisup_u(self, u, psi, gamma):
        gamma = (gamma.sum(axis=0) / gamma.shape[0])
        pprime, tprime = self.estim_p_t_sup(u)
        cprime = (tprime.T * pprime).T
        px = (psi.sum(axis=0)) / psi.shape[0] + self.reg
        tx = gamma.sum(axis=1)
        tx = tx / px[:, np.newaxis, :]
        p, t = self.get_param_form(cprime, px, tx)
        return p, t


class GHTMC_ctod(HTMC_ctod):

    def get_param_form(self, cu, px, tx):
        c = (cu[np.newaxis,:,np.newaxis,:]*px).reshape(self.nbc_x*self.nbc_u, self.nbc_x*self.nbc_u)
        p = np.sum(c, axis=1) + self.reg
        p = p / p.sum()
        t = (c.T / p).T
        return p, t

    def estim_p_t_EM(self, psi, gamma):
        p, t = HMC_ctod.estim_p_t_EM(self, psi, gamma)
        return p, t

    def estim_p_t_EM_semisup_x(self, x, psi, gamma):
        gammaprime = (gamma.sum(axis=0)) / gamma.shape[0]
        xprime = np.stack((x[:-1], x[1:]), axis=-1)
        aux = np.moveaxis(np.indices((self.nbc_x, self.nbc_x)), 0, -1)
        px = (((gamma[:, np.newaxis, :, np.newaxis, ...] * (np.all(
            xprime[:, np.newaxis, np.newaxis, ...] == aux[np.newaxis, ...], axis=-1))[:, :, np.newaxis, :,
                                                           np.newaxis]).sum(axis=0)) / (
                      gamma.sum(axis=0)[np.newaxis, :, np.newaxis, ...] + self.reg))
        p, t = self.get_param_form(gammaprime, px, None)
        return p, t

    def estim_p_t_EM_semisup_u(self, u, psi, gamma):
        pprime, tprime = self.estim_p_t_sup(u)
        cprime = (tprime.T * pprime).T
        px = gamma.sum(axis=0) / gamma.shape[0]
        p, t = self.get_param_form(cprime, px, None)
        return p, t
