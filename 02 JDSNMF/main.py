import os
import pandas as pd
import tensorflow as tf
import numpy as np
import math
import time
from math import sqrt
import matplotlib.pyplot as plt
SEED = 1
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

tf.compat.v1.disable_eager_execution()


def set_seeds(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.compat.v1.random.set_random_seed(seed)
    np.random.seed(seed)


def set_global_determinism(seed=SEED):
    set_seeds(seed=seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    tf.compat.v1.config.threading.set_inter_op_parallelism_threads(1)
    tf.compat.v1.config.threading.set_intra_op_parallelism_threads(1)


set_global_determinism(seed=SEED)


def multipl(a, b):
    sumofab = 0.0
    for i in range(len(a)):
        temp = a[i] * b[i]
        sumofab += temp
    return sumofab


def corrcoef(x, y):
    n = len(x)
    # 求和
    sum1 = sum(x)
    sum2 = sum(y)
    # 求乘积之和
    sumofxy = multipl(x, y)
    # 求平方和
    sumofx2 = sum([pow(i, 2) for i in x])
    sumofy2 = sum([pow(j, 2) for j in y])
    num = sumofxy - (float(sum1) * float(sum2) / n)
    # 计算皮尔逊相关系数
    den = sqrt((sumofx2 - float(sum1 ** 2) / n) * (sumofy2 - float(sum2 ** 2) / n))
    return num / den


class EarlyStopping:
    def __init__(self, patience=0, verbose=0):
        self._step = 0
        self._loss = float('inf')
        self.patience = patience
        self.verbose = verbose

    def validate(self, loss):
        if self._loss < loss:
            self._step += 1
            if self._step > self.patience:
                if self.verbose:
                    print('Training process is stopped early')
                return True
        else:
            self._step = 0
            self._loss = loss

        return False


def frob(z):
    vec = tf.reshape(z, [-1])
    return tf.reduce_sum(tf.multiply(vec, vec))


def minmax_sca(data):
    """
    离差标准化
    param data:传入的数据
    return:标准化之后的数据
    """
    new_data = (data - data.min()) / (data.max() - data.min())
    return new_data


def get_connectivity(data, alpha):
    X = np.power(np.corrcoef(data.T), alpha)
    diag_X = np.diag(np.diag(X))
    X = X - diag_X
    L = np.diag(sum(X, 2))
    Penalty = L - X
    Penalty = np.array(Penalty, dtype='float32')
    return Penalty


def main():
    np.random.seed(1)
    gene_df = pd.read_csv('train_x_gene_noname.csv', header=None)
    # gene_df = minmax_sca(gene_df)
    meth_df = pd.read_csv('train_x_meth_noname.csv', header=None)
    # meth_df = minmax_sca(meth_df)
    gene = gene_df.values
    meth = meth_df.values

    L1 = get_connectivity(gene, 2)
    L1 = tf.Variable(L1, tf.float32)

    L2 = get_connectivity(meth, 2)
    L2 = tf.Variable(L2, tf.float32)
    # Hyperparameters
    max_steps = 15000
    early_stopping = EarlyStopping(patience=200, verbose=1)
    first_reduced_dimension_1 = 151  # K值
    second_reduced_dimension = 113
    third_reduced_dimension = 76
    fourth_reduced_dimension = 38

    lambda_ = 0.001

    n, dm = meth.shape
    _, dg = gene.shape

    tf.compat.v1.set_random_seed(1)
    sess = tf.compat.v1.InteractiveSession()

    DM = tf.compat.v1.placeholder(tf.float32, shape=(None, dm))
    GE = tf.compat.v1.placeholder(tf.float32, shape=(None, dg))

    # Initialization using SVD
    DM_svd_u_1, _, DM_svd_vh_1 = np.linalg.svd(meth, full_matrices=False)
    DM_svd_u_2, _, DM_svd_vh_2 = np.linalg.svd(DM_svd_u_1, full_matrices=False)
    DM_svd_u_3, _, DM_svd_vh_3 = np.linalg.svd(DM_svd_u_2, full_matrices=False)
    DM_svd_u_4, _, DM_svd_vh_4 = np.linalg.svd(DM_svd_u_3, full_matrices=False)

    GE_svd_u_1, _, GE_svd_vh_1 = np.linalg.svd(gene, full_matrices=False)
    GE_svd_u_2, _, GE_svd_vh_2 = np.linalg.svd(GE_svd_u_1, full_matrices=False)
    GE_svd_u_3, _, GE_svd_vh_3 = np.linalg.svd(GE_svd_u_2, full_matrices=False)
    GE_svd_u_4, _, GE_svd_vh_4 = np.linalg.svd(GE_svd_u_3, full_matrices=False)

    U = tf.Variable(tf.cast(DM_svd_u_4[:, 0:first_reduced_dimension_1], tf.float32))

    Z21 = tf.Variable(
        tf.cast(DM_svd_u_2[0:first_reduced_dimension_1, 0:second_reduced_dimension], tf.float32))
    Z11 = tf.Variable(
        tf.cast(GE_svd_u_2[0:first_reduced_dimension_1, 0:second_reduced_dimension], tf.float32))

    Z22 = tf.Variable(
        tf.cast(DM_svd_u_3[0:second_reduced_dimension, 0:third_reduced_dimension], tf.float32))
    Z12 = tf.Variable(
        tf.cast(GE_svd_u_3[0:second_reduced_dimension, 0:third_reduced_dimension], tf.float32))

    Z23 = tf.Variable(
        tf.cast(DM_svd_u_4[0:third_reduced_dimension, 0:fourth_reduced_dimension], tf.float32))
    Z13 = tf.Variable(
        tf.cast(GE_svd_u_4[0:third_reduced_dimension, 0:fourth_reduced_dimension], tf.float32))

    H23 = tf.Variable(tf.cast(DM_svd_vh_1[0:fourth_reduced_dimension, :], tf.float32))
    H13 = tf.Variable(tf.cast(GE_svd_vh_1[0:fourth_reduced_dimension, :], tf.float32))

    H10 = tf.sigmoid(tf.matmul(Z11, tf.sigmoid(tf.matmul(Z12, tf.sigmoid(tf.matmul(Z13, H13))))))
    H20 = tf.sigmoid(tf.matmul(Z21, tf.sigmoid(tf.matmul(Z22, tf.sigmoid(tf.matmul(Z23, H23))))))

    H11 = tf.sigmoid(tf.matmul(Z12, tf.sigmoid(tf.matmul(Z13, H13))))
    H21 = tf.sigmoid(tf.matmul(Z22, tf.sigmoid(tf.matmul(Z23, H23))))

    H12 = tf.sigmoid(tf.matmul(Z13, H13))
    H22 = tf.sigmoid(tf.matmul(Z23, H23))

    # loss function
    loss = frob(
        GE - tf.matmul(U, tf.sigmoid(
            tf.matmul(Z11, tf.sigmoid(tf.matmul(Z12, tf.sigmoid(tf.matmul(Z13, H13)))))))) + \
           frob(
               DM - tf.matmul(U, tf.sigmoid(
                   tf.matmul(Z21, tf.sigmoid(tf.matmul(Z22, tf.sigmoid(tf.matmul(Z23, H23)))))))) + \
           lambda_ * (frob(U) + frob(Z11) + frob(Z12) + frob(Z13) + frob(H13) + frob(Z21) + frob(
        Z22) + frob(
        Z23) + frob(
        H23)
                      )

    diff_DM = frob(
        DM - tf.matmul(U, tf.sigmoid(
            tf.matmul(Z21, tf.sigmoid(tf.matmul(Z22, tf.sigmoid(tf.matmul(Z23, H23))))))))
    diff_GE = frob(
        GE - tf.matmul(U, tf.sigmoid(
            tf.matmul(Z11, tf.sigmoid(tf.matmul(Z12, tf.sigmoid(tf.matmul(Z13, H13))))))))

    MF = frob(
        GE - tf.matmul(U, tf.sigmoid(
            tf.matmul(Z11, tf.sigmoid(tf.matmul(Z12, tf.sigmoid(tf.matmul(Z13, H13)))))))) + \
         frob(
             DM - tf.matmul(U, tf.sigmoid(
                 tf.matmul(Z21, tf.sigmoid(tf.matmul(Z22, tf.sigmoid(tf.matmul(Z23, H23))))))))
    F = lambda_ * (
            frob(U) + frob(Z11) + frob(Z12) + frob(Z13) + frob(H13) + frob(Z21) + frob(Z22) + frob(
        Z23) + frob(H23))

    train_step = tf.compat.v1.train.AdamOptimizer(1e-3).minimize(loss)  # 优化器

    tf.compat.v1.global_variables_initializer().run()

    funval = []
    _, loss_iter = sess.run([train_step, loss], feed_dict={DM: meth, GE: gene})
    funval.append(loss_iter)

    indicate_dict = {}
    loss_list = []
    diff_me_list = []
    diff_ge_list = []
    diff_linc_list = []
    MF_list = []
    L_list = []
    indicate_corr = {}

    for i in range(max_steps):
        _, loss_iter = sess.run([train_step, loss], feed_dict={DM: meth, GE: gene})
        funval.append(loss_iter)
        loss_1 = sess.run(loss, feed_dict={DM: meth, GE: gene})
        diff_me = sess.run(diff_DM, feed_dict={DM: meth, GE: gene})
        diff_ge = sess.run(diff_GE, feed_dict={DM: meth, GE: gene})
        MF_1 = sess.run(MF, feed_dict={DM: meth, GE: gene}), sess.run(F, feed_dict={DM: meth,
                                                                                    GE: gene})
        loss_list.append(loss_1)
        diff_me_list.append(diff_me)
        diff_ge_list.append(diff_ge)
        MF_list.append(MF_1)
        print(i, " Loss : %f" % loss_1)
        print(" Average diff_me : %f" % diff_me)
        print(" Average diff_ge : %f" % diff_ge)
        print(" MF: %f  l2: %f" % MF_1)
        if early_stopping.validate(loss_iter):
            break
        if math.isnan(loss_iter):
            break

    print("==============================================================================")
    print(" Loss : %f" % loss_list[-1])
    print(" Average diff_me : %f" % diff_me_list[-1])
    print(" Average diff_ge : %f" % diff_ge_list[-1])
    print(" MF: %f  l2: %f" % MF_list[-1])
    H20 = sess.run(tf.compat.v1.sigmoid(
        tf.matmul(Z21, tf.sigmoid(tf.matmul(Z22, tf.sigmoid(tf.matmul(Z23, H23)))))),
        feed_dict={DM: meth, GE: gene})
    H10 = sess.run(
        tf.sigmoid(tf.matmul(Z11, tf.sigmoid(tf.matmul(Z12, tf.sigmoid(tf.matmul(Z13, H13)))))),
        feed_dict={DM: meth, GE: gene})
    U_ = sess.run(U, feed_dict={DM: meth, GE: gene})

    H20_df = pd.DataFrame(H20)
    H20_df.to_csv('H201_150.csv', header=None, index=False)
    H10_df = pd.DataFrame(H10)
    H10_df.to_csv('H101_150.csv', header=None, index=False)
    U_df = pd.DataFrame(U_)
    U_df.to_csv('U1_150.csv', header=None, index=False)
    indicate_dict['Loss'] = loss_list
    indicate_dict['Average diff_me'] = diff_me_list
    indicate_dict['Average diff_ge'] = diff_ge_list
    indicate_dict['MF'] = MF_list
    indicate_df = pd.DataFrame(indicate_dict)
    # indicate_df.to_csv('indicated_ori_150.csv', index=False)
    tf.compat.v1.reset_default_graph()
    sess.close()
    XX10 = np.dot(U_df.values, H10_df.values)
    XX10_df = pd.DataFrame(XX10)
    XX20 = np.dot(U_df.values, H20_df.values)
    XX20_df = pd.DataFrame(XX20)
    # XX10_df.to_csv('WH1_150.csv', header=None, index=False)
    # XX20_df.to_csv('WH2_150.csv', header=None, index=False)
    pd.DataFrame(loss_list).to_csv('loss.csv', header=None, index=False)
    sss1 = len(gene_df.index.values)
    sss3 = len(gene_df.columns.values)
    sss4 = len(meth_df.columns.values)
    XX11 = gene_df.values.reshape((1, sss1 * sss3))
    VV10 = XX10.reshape((1, sss1 * sss3))
    plt.scatter(XX11, VV10, color="skyblue", s=100)  # 创建散点图
    plt.show()
    XX22 = meth_df.values.reshape((1, sss1 * sss4))
    VV20 = XX20.reshape((1, sss1 * sss4))
    plt.scatter(XX22, VV20, color="skyblue", s=100)  # 创建散点图
    plt.show()
    corr10 = np.corrcoef(XX11, VV10)[0, 1]
    corr20 = np.corrcoef(XX22, VV20)[0, 1]
    print(corr10, corr20)


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print(end - start)

# 热图绘制
# sss1 = len(gene_df.index.values)
# sss3 = len(gene_df.columns.values)
# X12score = np.zeros(shape=(sss1, sss3))
# for i_x1 in range(sss3):
#     for i_x2 in range(sss3):
#         tmp_x11 = gene_df.values[:, i_x1]
#         tmp_x21 = XX1[:, i_x2]
#         # print(np.corrcoef(tmp_x11, tmp_x21))
#         X12score[i_x1, i_x2] = np.corrcoef(tmp_x11, tmp_x21)[0, 1]
# print(X12score)
