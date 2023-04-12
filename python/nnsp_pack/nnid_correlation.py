"""
Correlation function for speaker verification
"""
import numpy as np
import tensorflow as tf

def get_corr(
        feats,
        num_group_ppls,
        num_prons,
        eps = 0):
    """
    Correlation function
    """
    mat_corr = np.zeros((num_group_ppls*num_prons,num_group_ppls), dtype=np.float32)
    for i in range(num_group_ppls):
        center = np.sum(feats[i*num_prons:num_prons * (i+1),:], axis=0)
        print(center)
        for j in range(num_group_ppls):
            for k in range(num_prons):
                feat = feats[k + j * num_prons,:]
                if i == j:
                    center0 = (center - feat) / (num_prons - 1)
                else:
                    center0 = center / num_prons
                norm_feat = np.sum(feat**2)
                norm_center0 = np.sum(center0**2)

                corr = np.matmul(feat, center0)
                corr = corr / (np.sqrt(norm_feat * norm_center0) + eps)
                mat_corr[k + j * num_prons, i] = corr
    return mat_corr

def get_corr_fast(
        feats,
        diag,
        num_group_ppls,
        num_prons,
        eps = 0):
    """
    Fast Correlation function
    """
    # print(f"feats: {feats}")
    ones_1xnum_group_ppls = tf.ones((1,num_group_ppls), dtype=tf.float32)
    ones_sentsx1 = tf.ones((num_group_ppls * num_prons,1), dtype=tf.float32)
    centers = tf.matmul(tf.transpose(feats),diag)
    # print(f"centers {centers}")
    mat_corr = tf.matmul(feats, centers)
    eps = tf.reduce_sum(feats**2, axis=-1)
    eps = tf.reshape(eps, (num_group_ppls * num_prons,-1))
    eps = tf.matmul(eps, ones_1xnum_group_ppls) * diag
    mat_corr = mat_corr - eps
    # print(f"eps: {eps}")
    # print(f"mat_cor: {mat_corr}")

    scale = tf.ones(
        (num_group_ppls * num_prons,num_group_ppls),
        dtype=tf.float32) * num_prons - diag
    scale = 1/scale
    mat_corr = mat_corr * scale
    # print(scale)
    norm_centers = tf.reduce_sum(centers**2, axis=0)
    norm_centers = tf.reshape(norm_centers, (-1,num_group_ppls))

    norm_centers = tf.matmul(
        ones_sentsx1,
        norm_centers) * (1 - diag)
    # print(f"norm_centers: {norm_centers}")

    norm_centers_diag = tf.transpose(tf.matmul(centers, tf.transpose(diag)))
    # print(norm_centers_diag.shape)
    norm_centers_diag = tf.reduce_sum((norm_centers_diag - feats)**2, axis=-1)
    norm_centers_diag = tf.reshape(norm_centers_diag, (num_group_ppls * num_prons,1))
    norm_centers_diag = tf.matmul(
        norm_centers_diag,
        ones_1xnum_group_ppls) * diag
    # print(norm_centers_diag)
    norm_centers = (norm_centers + norm_centers_diag) * (scale**2)
    # print(norm_centers)
    norm_feats = tf.reduce_sum(feats**2, axis=-1)
    norm_feats = tf.reshape(norm_feats, (num_group_ppls * num_prons,1))
    norm_feats = tf.matmul(norm_feats, ones_1xnum_group_ppls)
    # print(norm_feats)
    mat_corr = mat_corr / (tf.sqrt(norm_centers * norm_feats) + eps)
    # print(mat_corr.numpy())
    return mat_corr

def gen_nnid_diag(
        num_group_ppls,
        num_prons):
    """
    Generate diag for nnid
    """
    diag = tf.constant(np.kron(np.eye(num_group_ppls), np.ones((num_prons,1))), dtype=np.float32)
    return diag

if __name__=="__main__":
    NUM_GROUP_PPLS = 3
    NUM_PRONS = 4
    DIM_FEAT = 6
    eps = 0
    FEATS = tf.constant(np.random.randn(NUM_GROUP_PPLS * NUM_PRONS, DIM_FEAT), dtype=np.float32)
    DIAG = gen_nnid_diag(NUM_GROUP_PPLS, NUM_PRONS)

    mat= get_corr_fast(
        tf.identity(FEATS),
        DIAG,
        NUM_GROUP_PPLS,
        NUM_PRONS,
        eps=eps).numpy()
    mat_ref = get_corr(
        FEATS.numpy(),
        NUM_GROUP_PPLS,
        NUM_PRONS,
        eps=eps)
    # print("corr:")
    # print(mat)
    
    # print("corr ref:")
    # print(mat_ref)

    print("err")
    print(mat-mat_ref)
    