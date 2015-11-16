from __future__ import print_function
from collections import defaultdict
import numpy as np
import scipy as sp
import cPickle as pickle
import time
from collections import defaultdict

num_users = 35736
num_items = 37801

def get_mse(ratings, ratings_predict):
    return np.mean((np.array(ratings).astype('float') -
                    np.array(ratings_predict).astype('float')) ** 2.)

def pack(theta, K, alpha, beta_users, beta_items, gamma_users, gamma_items):
    """ pack to theta, do not allocate new memory, just copy value"""
    theta[0] = alpha
    curr_ind = 1
    theta[curr_ind : curr_ind + num_users] = beta_users
    curr_ind += num_users
    theta[curr_ind : curr_ind + num_items] = beta_items
    curr_ind += num_items
    theta[curr_ind : curr_ind + num_users * K] = gamma_users.reshape((-1, ))
    curr_ind += num_users * K
    theta[curr_ind :] = gamma_items.reshape((-1, ))
    return theta

def unpack(theta, K):
    alpha = theta[0]
    curr_ind = 1
    beta_users = theta[curr_ind : curr_ind + num_users]
    curr_ind += num_users
    beta_items = theta[curr_ind : curr_ind + num_items]
    curr_ind += num_items
    gamma_users = theta[curr_ind : curr_ind + num_users * K].reshape((num_users, K))
    curr_ind += num_users * K
    gamma_items = theta[curr_ind :].reshape((num_items, K))
    return (alpha, beta_users, beta_items, gamma_users, gamma_items)

# # sanity check of pack / unpack
# # init theta
# theta = init_theta(K, num_users, num_items) # all parameters

# # check pack and unpack function
# a, bu, bi, gu, gi = unpack(theta, K)
# theta_new = pack(theta, K, a, bu, bi, gu, gi)
# assert np.array_equal(theta, theta_new)

# # check value and id sustained
# theta[0] = 1234.
# (a, bu, bi, gu, gi) = unpack(theta, K)
# new_theta = pack(theta, K, a, bu, bi, gu, gi)
# assert a == 1234.
# assert id(new_theta) == id(theta)
# assert new_theta[0] == 1234.

def objective(theta, grad_buffer, rating_array, lam, K):
    alpha, beta_users, beta_items, gamma_users, gamma_items = unpack(theta, K)
    cost = 0.0
    for datum in rating_array:
        user_index = datum[0]
        item_index = datum[1]
        cost += (float(alpha)
                 + beta_users[user_index]
                 + beta_items[item_index]
                 + np.dot(gamma_users[user_index], gamma_items[item_index])
                 - float(datum[2])
                ) ** 2.0
    cost += lam * (np.linalg.norm(theta) ** 2.0)
    return 0.5 * cost #/ rating_array.shape[0]

def gradient_user(theta, grad_buffer, rating_array, lam, K):
    """ keep gamma_items_grad to zero """
    # unpack theta
    alpha, beta_users, beta_items, gamma_users, gamma_items = unpack(theta, K)
    # reset and unpack grad_buffer
    grad_buffer.fill(0.)
    alpha_grad, beta_users_grad, beta_items_grad, gamma_users_grad, gamma_items_grad = unpack(grad_buffer, K)

    # cost term: accumulate gradients
    for datum in rating_array:
        # make prediction
        user_index = datum[0]
        item_index = datum[1]
        prediction = (float(alpha)
                      + beta_users[user_index]
                      + beta_items[item_index]
                      + np.dot(gamma_users[user_index], gamma_items[item_index]))
        common_offset = (prediction - float(datum[2])) # offset error
        # alpha
        alpha_grad += common_offset
        # beta_user
        beta_users_grad[user_index] += common_offset
        # beta_item
        beta_items_grad[item_index] += common_offset
        # gamma_user
        gamma_users_grad[user_index] += common_offset * gamma_items[item_index]
        # gamm_item
        gamma_items_grad[item_index].fill(0.)
    # regularization term
    beta_users_grad = beta_users_grad + lam * beta_users
    # pack
    grad_buffer = pack(grad_buffer, K, alpha_grad,
                       beta_users_grad, beta_items_grad,
                       gamma_users_grad, gamma_items_grad)
    grad_buffer = grad_buffer #/ rating_array.shape[0]
    return grad_buffer

def gradient_item(theta, grad_buffer, rating_array, lam, K):
    """ keep gamma_users_grad to zero """
    # unpack theta
    alpha, beta_users, beta_items, gamma_users, gamma_items = unpack(theta, K)
    # reset and unpack grad_buffer
    grad_buffer.fill(0.)
    alpha_grad, beta_users_grad, beta_items_grad, gamma_users_grad, gamma_items_grad = unpack(grad_buffer, K)

    # cost term: accumulate gradients
    for datum in rating_array:
        # make prediction
        user_index = datum[0]
        item_index = datum[1]
        prediction = (float(alpha)
                      + beta_users[user_index]
                      + beta_items[item_index]
                      + np.dot(gamma_users[user_index], gamma_items[item_index]))
        common_offset = (prediction - float(datum[2])) # offset error
        # alpha
        alpha_grad += common_offset
        # beta_user
        beta_users_grad[user_index] += common_offset
        # beta_item
        beta_items_grad[item_index] += common_offset
        # gamma_user
        gamma_users_grad[user_index].fill(0.)
        # gamm_item
        gamma_items_grad[item_index] += common_offset * gamma_users[user_index]
    # regularization term
    beta_items_grad = beta_items_grad + lam * beta_items

    # pack
    grad_buffer = pack(grad_buffer, K, alpha_grad,
                       beta_users_grad, beta_items_grad,
                       gamma_users_grad, gamma_items_grad)
    grad_buffer = grad_buffer #/ rating_array.shape[0]
    return grad_buffer

def gradient(theta, grad_buffer, rating_array, lam, K):
    """ keep gamma_items_grad to zero """
    # unpack theta
    alpha, beta_users, beta_items, gamma_users, gamma_items = unpack(theta, K)
    # reset and unpack grad_buffer
    grad_buffer.fill(0.)
    alpha_grad, beta_users_grad, beta_items_grad, gamma_users_grad, gamma_items_grad = unpack(grad_buffer, K)

    # cost term: accumulate gradients
    for datum in rating_array:
        # make prediction
        user_index = datum[0]
        item_index = datum[1]
        prediction = (float(alpha)
                      + beta_users[user_index]
                      + beta_items[item_index]
                      + np.dot(gamma_users[user_index], gamma_items[item_index]))
        common_offset = (prediction - float(datum[2])) # offset error
        # alpha
        alpha_grad += common_offset
        # beta_user
        beta_users_grad[user_index] += common_offset
        # beta_item
        beta_items_grad[item_index] += common_offset
        # gamma_user
        gamma_users_grad[user_index] += common_offset * gamma_items[item_index]
        # gamm_item
        gamma_items_grad[item_index] += common_offset * gamma_users[user_index]
    # regularization term
    beta_users_grad = beta_users_grad + lam * beta_users
    beta_items_grad = beta_items_grad + lam * beta_items
    # pack
    grad_buffer = pack(grad_buffer, K, alpha_grad,
                       beta_users_grad, beta_items_grad,
                       gamma_users_grad, gamma_items_grad)
    grad_buffer = grad_buffer #/ rating_array.shape[0]
    return grad_buffer

def predict_one_rating(user_index, item_index, theta, K):
    user_index = int(user_index)
    item_index = int(item_index)
    alpha, beta_users, beta_items, gamma_users, gamma_items = unpack(theta, K)

    # user
    beta_user = beta_users[user_index]
    gamma_user = gamma_users[user_index]

    # item
    beta_item = beta_items[item_index]
    gamma_item = gamma_items[item_index]

    return alpha + beta_user + beta_item + np.dot(gamma_user, gamma_item)