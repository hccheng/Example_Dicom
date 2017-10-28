import numpy as np


# @njit
def _conf_mat_loop(a, b, n_class):
    """ Calculate the confusion matrix
    Args:
        a: a flat array
        b: a flat array

    Returns: the confusion matrix

    """
    assert len(a) == len(b), 'incompatible array length'
    # assert a.dtype == np.int32, 'incompatible type %s' % a.dtype
    # assert b.dtype == np.int32, 'incompatible type %s' % b.dtype
    cmat = np.zeros((n_class, n_class), dtype=np.int64)

    for i in range(len(a)):
        cmat[a[i], b[i]] += 1

    return cmat


def _conf_mat_np(a, b, n_class):
    assert len(a) == len(b), 'incompatible array length'
    # assert a.dtype == np.int32, 'incompatible type %s' % a.dtype
    # assert b.dtype == np.int32, 'incompatible type %s' % b.dtype
    cmat = np.zeros((n_class, n_class), dtype=np.int64)

    # this is much faster than naiive loop (without numba)
    for i in range(n_class):
        for j in range(n_class):
            cmat[i, j] = np.count_nonzero((a == i) * (b == j))

    return cmat

# (optional) speed-up using numba
try:
    from numba import double, jit
    _conf_mat = jit(_conf_mat_loop)
except ImportError as e:
    double = None
    jit = None
    # if we can't use numba, use this function instead
    _conf_mat = _conf_mat_np


def confusion_matrix(a, b, n_class):
    if not isinstance(a, np.ndarray):
        a = np.asarray(a, dtype=np.int32)
    if not isinstance(b, np.ndarray):
        b = np.asarray(b, dtype=np.int32)
    assert a.shape == b.shape, 'incompatible label shapes %s != %s' % (str(a.shape), str(b.shape))

    # a = a.astype(np.int32)
    # b = b.astype(np.int32)

    return _conf_mat(a.ravel(), b.ravel(), n_class)


def calc_measure(*args, **kwargs):
    """ Calculate performance measures
    Args:
        y: ground-truth labels
        yhat: prediction labels

    Returns: a dictionary of reported measures

    """
    cmat = None
    if len(args) == 1:
        cmat = args[0]
    elif len(args) == 2:
        y = args[0]
        yhat = args[1]
        # exclude those that should be omitted
        m = (y >= 0)
        try:
            y = y[m]
            yhat = yhat[m]
        except ValueError as err:
            print(err)
            exit()

        # calculate confusion matrix
        cmat = confusion_matrix(y, yhat, kwargs['n_class'])
    else:
        print('calc_measure takes either a confusion matrix or two arrays of label')
        exit()

    assert cmat is not None

    try:
        cmat = cmat.astype(np.float)

        # number of predictions and support
        n_pred = np.sum(cmat, axis=0)
        support = np.sum(cmat, axis=1)

        # precision
        precisions = cmat.diagonal() / (n_pred + (n_pred == 0))
        # recall
        recalls = cmat.diagonal() / (support + (support == 0))
        # f1-score (aka dice)
        f1 = 2.0*(precisions*recalls) / (precisions+recalls + ((precisions+recalls) == 0))
        # calculate per-class VOC score = TP/(TP+FP+FN); c.f. Everingham et al. 2009
        # FYI this is also known as Jaccard index and IoU
        voc = [cmat[c, c] / (np.sum(cmat[c, :]) + np.sum(cmat[:, c]) - cmat[c, c]) for c in range(cmat.shape[0])]

        # TODO it *seems* that taking the "vanila mean of vocs" is the right thing to do
        # c.f. the matlab code in VOCevalseg.m from the following link
        # http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar

        # higher is better, in [0, 1]
        # return both raw and adjusted index
        def rand_index(cm):
            assert cm.dtype == np.float
            n = np.sum(cm)
            nis = np.sum(np.sum(cm, axis=1)*np.sum(cm, axis=1))
            njs = np.sum(np.sum(cm, axis=0)*np.sum(cm, axis=0))
            nc = (n*(n**2+1) - (n+1)*nis - (n+1)*njs + 2.0*(nis*njs)/n)/(2.0*(n-1))
            t1 = n*(n-1)/2
            t2 = np.sum(cm * cm)
            t3 = 0.5*(nis+njs)
            return (t1+t2-t3)/t1, (t1+t2-t3-nc)/(t1-nc)

        # Martin et al., ICCV 2001
        # lower is better
        def gce(cm):
            assert cm.dtype == np.float
            n = np.sum(cm)
            m1 = np.sum(cm, axis=1)
            m2 = np.sum(cm, axis=0)
            e1 = 1.0 - np.sum(np.divide(np.sum(cm * cm, axis=1),  m1 + (m1 == 0))) / n
            e2 = 1.0 - np.sum(np.divide(np.sum(cm * cm, axis=0),  m2 + (m2 == 0))) / n
            return min(e1, e2)

        result = dict()
        result['class'] = list(range(cmat.shape[0]))
        result['support'] = support
        result['precision'] = precisions
        result['recall'] = recalls
        result['f1'] = f1
        result['voc'] = voc
        result['ri'], result['ari'] = rand_index(cmat)
        result['gce'] = gce(cmat)

        return result
    except ZeroDivisionError:
        return None


def print_report(result):
    """ Print the result
    Args:
        result: performance measures calculated

    Returns:

    """
    support = result['support']
    precisions = result['precision']
    recalls = result['recall']
    f1 = result['f1']
    voc = result['voc']

    def arr_round(arr):
        return ['{0:.3f}'.format(round(a, 3)) for a in arr]

    def avg(arr, weighted_by_support=True):
        # FIXME at least for VOC this should NOT be a support-weighted average
        # I might just leave it like this because it is trivial to calculate anyway
        if weighted_by_support:
            return '{0:.3f}'.format(round(np.sum(arr * support) / np.sum(support), 3))
        else:
            return '{0:.3f}'.format(round(np.sum(arr) / len(arr), 3))

    from prettytable import PrettyTable
    table = PrettyTable()
    table.float_format = '0.4'
    table.add_column('class', result['class'] + ['', 'avg / all'])
    table.add_column('precision', arr_round(precisions) + ['', avg(precisions)], align='r')
    table.add_column('recall', arr_round(recalls) + ['', avg(recalls)], align='r')
    table.add_column('f1-score', arr_round(f1) + ['', avg(f1)], align='r')
    table.add_column('voc', arr_round(voc) + ['', avg(voc)], align='r')
    table.add_column('support', support.astype(np.int).tolist() + ['', np.sum(support).astype(np.int)], align='r')
    print(table)

    print('rand index = %0.4f' % result['ri'])
    print('adjusted rand index = %0.4f' % result['ari'])
    print('global consistancy error = %0.4f' % result['gce'])

if __name__ == '__main__':
    # n_elem = 1024*768*165
    n = 5
    n_elem = (300, 300, 300)
    d1 = np.random.randint(0, n, n_elem, dtype=np.int32)
    d2 = np.random.randint(0, n, n_elem, dtype=np.int64)

    print(n_elem)

    import timeit
    begin = timeit.default_timer()
    mat = confusion_matrix(d1, d2, n_class=n)
    print(timeit.default_timer() - begin)

    print_report(calc_measure(d1, d2, n_class=n))
    print_report(calc_measure(mat + mat))
