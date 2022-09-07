import torch
import matplotlib.pyplot as plt

# Ji, Jun. "Robust inversion using biweight norm and its application to seismic inversion." Exploration Geophysics 43.2 (2012): 70-76.

if __name__ == "__main__":
    n_data = 100
    n_random = 20
    X = torch.cat([torch.rand(n_data,1)*10.-5., torch.ones(n_data,1)], dim=1)
    m = torch.tensor([[2.], [-4.]]) #the model we want to identify
    m_other = torch.tensor([[-10.], [-4.]]) #the model we want to identify
    Y = X.mm(m)+torch.randn(n_data,1)
    Y[:n_random] = X[:n_random].mm(m_other)+torch.randn(n_random,1)*2.
    print(X.size(), Y.size())
    #Y = torch.cat([X[:n_data-n_random]*2-4.+torch.randn(n_data-n_random, 1), torch.rand(n_random,1)*20-10])

    m_hat = torch.inverse(X.t().mm(X)).mm(X.t()).mm(Y)

    m_hat_abs = m_hat
    for i in range(10):
        r = X.mm(m_hat_abs) - Y
        w_abs = 1./torch.abs(r)
        W = torch.eye(n_data)*w_abs
        m_hat_abs = torch.inverse(X.t().mm(W).mm(X)).mm(X.t()).mm(W).mm(Y)

    m_hat_tukey = m_hat
    for i in range(10):
        r = X.mm(m_hat_tukey) - Y
        r_abs = torch.norm(r, dim=1)
        MAD = torch.median(r_abs)
        print(r_abs)
        print(MAD)
        epsilon = 4.685 * MAD / 0.6745
        w_tukey = torch.zeros(n_data,1)
        w_tukey[r_abs<=epsilon,0] = ((1.-(r_abs/epsilon)**2.)**2.)[r_abs<=epsilon]
        #w_abs = 1./torch.abs(r)
        W = torch.eye(n_data)*w_tukey
        #m_hat_tukey = torch.inverse(X.t().mm(W).mm(X)).mm(X.t()).mm(W).mm(Y)
        m_hat_tukey = torch.inverse(X.t().mm(w_tukey * X)).mm(X.t()).mm(w_tukey * Y)


    X_lim = torch.tensor([[-5.,1.],
                          [5. ,1.]])
    plt.plot(X_lim[:,0], X_lim.mm(m_hat))
    plt.plot(X_lim[:, 0], X_lim.mm(m_hat_abs), c='r')
    plt.plot(X_lim[:, 0], X_lim.mm(m_hat_tukey), c='g')
    plt.scatter(X[:,0],Y)#, s = w_tukey)
    plt.show()
