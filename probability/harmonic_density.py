import matplotlib.pyplot as plt
import torch


class HarmonicPDF():
    def __init__(self, n_levels, n_bases):
        '''

        :param n_levels:
        :param n_bases:
        '''
        self.n_levels = n_levels
        self.n_bases = n_bases



    def activation(self, samples):
        n_samples = samples.numel()
        offsets = torch.linspace(0., 2.*torch.pi * (self.n_bases-1) / self.n_bases, self.n_bases)
        outer_samples_offsets = samples.unsqueeze(0)-offsets.unsqueeze(1)
        a_ = torch.cos(outer_samples_offsets)*.5+.5
        a = a_ / torch.sum(a_, dim=0)
        a = torch.einsum('bs->b',a)
        a = a / n_samples / self.n_bases
        return a

    def pdf(self, a, xs):
        offsets = torch.linspace(0., 2.*torch.pi * (self.n_bases-1) / self.n_bases, self.n_bases)
        outer_samples_offsets = xs.unsqueeze(0)-offsets.unsqueeze(1)
        ys = torch.cos(outer_samples_offsets)*.5+.5
        ys = torch.einsum("bx,b->x", ys, a) / torch.pi *2
        return ys

    def plot(self, ax, a, n_steps = 10000, c='k'):
        xs = torch.linspace(0, torch.pi * 2. * (n_steps-1)/n_steps, n_steps)
        ys = self.pdf(a, xs)
        ax.plot(xs, ys,c=c)

        #print integral as sanity check
        print('Integral:', torch.sum(ys/n_steps*2.*torch.pi))
        print('Integral cos:', torch.sum(torch.cos(xs)*.5+.5/n_steps*2.*torch.pi))


if __name__ == "__main__":
    fig, axarr = plt.subplots(2)

    n_samples = 1000
    mu, Sigma = torch.tensor([2.]*n_samples), torch.tensor([1.0]*n_samples)
    samples = torch.normal(mu, Sigma)
    samples = torch.fmod(samples+2.0*torch.pi, 2.*torch.pi)

    axarr[0].hist(samples, density=True)

    pdf = HarmonicPDF(1,3)
    a = pdf.activation(samples)
    print('a', a)
    pdf.plot(axarr[0], a, c='r')
    pdf.plot(axarr[0], torch.tensor([1., 0., 0.])/ torch.pi *2, c='r')
    pdf.plot(axarr[0], torch.tensor([0., 1., 0.])/ torch.pi *2, c='g')
    pdf.plot(axarr[0], torch.tensor([0., 0., 1.])/ torch.pi *2, c='b')


    plt.show()