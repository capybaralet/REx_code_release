
import matplotlib.pyplot as plt
import numpy as np
import numpy.lib as npl
import sys
from load import YEARS


EXP_LIST = [
    [2014, 2015, 2016],
    [2014, 2015, 2017],
    [2014, 2015, 2018],
    [2014, 2016, 2017],
    [2014, 2016, 2018],
    [2014, 2017, 2018],
    [2015, 2016, 2017],
    [2015, 2016, 2018],
    [2015, 2017, 2018],
    [2016, 2017, 2018],
]


def plot_data(means, stds, descr, plot_every=8, title='', ylabel=''):
    fig, ax = plt.subplots(1, figsize=(4,3))
    for m, s, info in zip(means, stds, descr):
        num_el = len(m)
        x = np.arange(num_el)
        x = x[:-(num_el%plot_every)].reshape(-1,plot_every).mean(1)
        y = m[:-(num_el%plot_every)].reshape(-1,plot_every).mean(1)
        dy = s[:-(num_el%plot_every)].reshape(-1,plot_every).mean(1)
        ax.fill_between(x, y-dy, y+dy, alpha=0.25)
        ax.plot(x, y, label=info)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Iteration")

    fig.tight_layout()
    plt.legend()


if __name__ == "__main__":

    if sys.argv[-1] == 'all':
        # combine
        mat_erm, mat_irm, mat_rex = [np.loadtxt('results/%s_matrix.dat' % k) for k in ['erm', 'irm', 'rex']]
        mat_train = mat_erm[:, :len(YEARS)] / mat_erm[:, :len(YEARS)]
        mat_erm = mat_erm[:, len(YEARS):]
        mat_irm = mat_irm[:, len(YEARS):]
        mat_rex = mat_rex[:, len(YEARS):]

        irm_rel = 100 * (mat_irm - mat_erm) / mat_erm
        rex_rel = 100 * (mat_rex - mat_erm) / mat_erm

        notnan = irm_rel > -999
        vmax = np.abs([irm_rel[notnan].min(), irm_rel[notnan].max(), rex_rel[notnan].min(), rex_rel[notnan].max()]).max()

        e = mat_erm[notnan]
        i = mat_irm[notnan]
        r = mat_rex[notnan]
        ir = irm_rel[notnan]
        rr = rex_rel[notnan]

        print('cases irm better', (ir > 0).mean())
        print('cases rex better', (rr > 0).mean())
        print('irm rel performance (mean, std, min, max)', ir.mean(), ir.std(), ir.min(), ir.max())
        print('rex rel performance (mean, std, min, max)', rr.mean(), rr.std(), rr.min(), rr.max())
        print('erm performance (mean, std, min, max)', e.mean(), e.std(), e.min(), e.max())
        print('irm performance (mean, std, min, max)', i.mean(), i.std(), i.min(), i.max())
        print('rex performance (mean, std, min, max)', r.mean(), r.std(), r.min(), r.max())

        x, y = np.meshgrid(range(5), range(10))
        x = x[mat_train == 1]
        y = y[mat_train == 1]

        ax = plt.subplot(131)
        plt.grid(zorder=-99)
        plt.imshow(mat_train, cmap='binary')
        plt.scatter(x, y, c='k', zorder=10)
        plt.xticks(range(len(YEARS)), YEARS)
        plt.yticks(range(10), ['']*10)
        plt.title('Training envs.')
        plt.ylabel('Task')
        plt.xticks(rotation=45)
        plt.xlim(-0.5, 4.5)
        plt.ylim(9.5, -0.5)

        ax = plt.subplot(132)
        plt.imshow(irm_rel, cmap='PiYG', vmin=-vmax, vmax=vmax, zorder=1)
        plt.scatter(x, y, c='k', zorder=10)
        plt.xticks(range(len(YEARS)), YEARS)
        plt.yticks(range(10), ['']*10)
        plt.title('Test IRM')
        plt.xticks(rotation=45)

        ax = plt.subplot(133)
        im = plt.imshow(rex_rel, cmap='PiYG', vmin=-vmax, vmax=vmax, zorder=1)
        plt.scatter(x, y, c='k', zorder=10)
        plt.xticks(range(len(YEARS)), YEARS)
        plt.yticks(range(10), ['']*10)
        plt.title('Test REx')
        plt.xticks(rotation=45)


        cbar = plt.colorbar(im)
        cbar.ax.set_ylabel('Performance relative to ERM (%)', rotation=90)

        plt.savefig('grid.pdf')
        plt.show()
        exit()

    # plot single experiment
    train_envs = [np.array(e, dtype=int) for e in EXP_LIST]
    test_envs = [npl.setxor1d(YEARS, e) for e in train_envs]

    fnames = ['results/%s_%s_%s.npz' % (sys.argv[-1],
        ','.join([str(e) for e in train_env]),
        ','.join([str(e) for e in test_env])) \
            for train_env, test_env in zip(train_envs, test_envs)]
    print(fnames)

    files = [np.load(fname) for fname in fnames]
    SMOOTH = 50

    train_means = [f['train_acc'][:,-SMOOTH:].mean() for f in files]
    train_stds = [f['train_acc'][:,-SMOOTH:].std() for f in files]

    max_pts = [f['test_acc'].mean(0)[:-1].reshape((-1, SMOOTH, 2)).mean(1).argmax(0).clip(1, np.inf) * SMOOTH for f in files]
    max_pts = [p.astype(int)[::-1] for p in max_pts]

    print('stopping points:', max_pts)

    test_means = [
        np.stack([
        f['test_acc'][:, pts[i]-SMOOTH//2:pts[i]+SMOOTH//2, i].mean() \
            for i in [0,1]]) \
                for f,pts in zip(files, max_pts)]

    out_img = np.zeros((len(train_envs), 2*len(YEARS)))

    for i, (e_tr, m_tr, e_te, m_te) in enumerate(zip(train_envs, train_means, test_envs, test_means)):
        out_img[i, e_tr-YEARS[0]] = m_tr
        out_img[i, len(YEARS)-YEARS[0]+e_te] = m_te

    np.savetxt('results/%s_matrix.dat' % sys.argv[-1], out_img)
    #plt.imshow(out_img)
    #plt.show()

