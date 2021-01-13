# -*- coding: utf-8 -*-

'''
@Time    : 2020/12/30 下午6:08
@Author  : liou
@FileName: clustering.py
@Software: PyCharm
 
'''

import os
import torch
import json
import numpy as np
from sklearn.cluster import SpectralClustering

def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data

def get_train_label(anet_label):
    '''
    heck all videos and build two dick, notice all labels are from dataset's ground truth
    Parameters
    ----------
    anet_label

    Returns
    -------

    '''
    training_index = []
    action_2_video = {}
    video_2_action = {}

    for tv in anet_label:
        if anet_label[tv]["subset"] != "training":
            continue
        training_index.append(tv)
        tc = anet_label[tv]["annotations"][0]["label"]
        if action_2_video.has_key(tc):
            action_2_video[tc].append(tv)
        else:
            action_2_video[tc] = []
            action_2_video[tc].append(tv)

        if video_2_action.has_key(tv):
            video_2_action[tv].append(tc)
        else:
            video_2_action[tv] = []
            video_2_action[tv].append(tc)

    for tk in action_2_video:
        action_2_video[tk] = list(set(action_2_video[tk]))
    for tk in video_2_action:
        video_2_action[tk] = list(set(video_2_action[tk]))
    return training_index, action_2_video, video_2_action

def get_feature(training_index, flow_path, rgb_path, atten_weight_path=None):
    if atten_weight_path is None:
        print("Avg pooling feature")
    else:
        print("Attention pooling feature")
    all_atten_fea = []
    all_atten_rgb = []
    all_atten_flow = []
    late_fusion_flag = False
    global_score_flag = False
    cnt = 0
    for tv_name in training_index:
        cnt += 1
        if cnt % 500 == 0:
            print("%d loaded" % cnt)
        tf = "v_" + tv_name + "-flow.npz"
        sig_flow_path = os.path.join(flow_path, tf)
        flow_data = np.load(sig_flow_path)
        flow_fea = flow_data['feature']

        tr = "v_" + tv_name + "-rgb.npz"
        sig_rgb_path = os.path.join(rgb_path, tr)
        rgb_data = np.load(sig_rgb_path)
        rgb_fea = rgb_data['feature']

        # avg fea
        if atten_weight_path is None:
            att_weight_flow_fea = np.mean(flow_fea[0, :, :], axis=0)
            att_weight_rgb_fea = np.mean(rgb_fea[0, :, :], axis=0)

        elif late_fusion_flag:
            tcas = tv_name + ".npz"
            cas = np.load(os.path.join(atten_weight_path, tcas))
            att_weight = cas['weight']

            # downsample
            tmp_flow_fea = flow_fea[0, :, :]
            tmp_rgb_fea = rgb_fea[0, :, :]

            atten_flow_fea = tmp_flow_fea * att_weight
            att_weight_flow_fea = np.sum(atten_flow_fea, axis=0)
            atten_rgb_fea = tmp_rgb_fea * att_weight
            att_weight_rgb_fea = np.sum(atten_rgb_fea, axis=0)

            if global_score_flag is True:
                global_score = cas['global_score']
                global_score = global_score / np.linalg.norm(global_score)

        else:
            # print("use diff weight for rgb and flow")
            tcas = tv_name + ".npz"
            rgb_weight_path = atten_weight_path.replace("both", "rgb")
            flow_weight_path = atten_weight_path.replace("both", "flow")

            rgb_cas = np.load(os.path.join(rgb_weight_path, tcas))
            flow_cas = np.load(os.path.join(flow_weight_path, tcas))

            rgb_att_weight = rgb_cas['weight']
            flow_att_weight = flow_cas['weight']

            # downsample
            tmp_flow_fea = flow_fea[0, :, :]
            tmp_rgb_fea = rgb_fea[0, :, :]

            atten_flow_fea = tmp_flow_fea * flow_att_weight
            att_weight_flow_fea = np.sum(atten_flow_fea, axis=0)
            atten_rgb_fea = tmp_rgb_fea * rgb_att_weight
            att_weight_rgb_fea = np.sum(atten_rgb_fea, axis=0)

            if global_score_flag is True:
                rgb_global = rgb_cas['global_score']
                flow_global = flow_cas['global_score']
                global_score = (rgb_global + flow_global) / 2.0
                global_score = global_score / np.linalg.norm(global_score)

        # normalize flow and rgb seperately
        att_weight_flow_fea = att_weight_flow_fea / np.linalg.norm(att_weight_flow_fea)
        att_weight_rgb_fea = att_weight_rgb_fea / np.linalg.norm(att_weight_rgb_fea)

        if (atten_weight_path is None) or (global_score_flag is False):
            fuse_fea = np.concatenate((att_weight_rgb_fea, att_weight_flow_fea), axis=0)
        elif global_score_flag:
            fuse_fea = np.concatenate((att_weight_rgb_fea, att_weight_flow_fea, global_score), axis=0)

        all_atten_fea.append(fuse_fea)
        all_atten_flow.append(att_weight_flow_fea)
        all_atten_rgb.append(att_weight_rgb_fea)

    return all_atten_fea, all_atten_rgb, all_atten_flow

def get_subset(num_subset_class, action_2_video, training_index, all_atten, all_atten_rgb, all_atten_flow):
    # num_subset_cluster = num_subset_class
    subset_class = action_2_video.keys()[0:num_subset_class]
    subset_index = []
    subset_atten = []
    subset_atten_rgb = []
    subset_atten_flow = []
    for tc in subset_class:
        tc_sub = action_2_video[tc]
        for tv in tc_sub:
            subset_index.append(tv)
            tv_fea_index = training_index.index(tv)
            subset_atten.appen (all_atten[tv_fea_index])
            subset_atten_rgb.append(all_atten_rgb[tv_fea_index])
            subset_atten_flow.append(all_atten_flow[tv_fea_index])

    subset_atten = np.stack(subset_atten, axis=0)
    subset_atten_rgb = np.stack(subset_atten_rgb, axis=0)
    subset_atten_flow = np.stack(subset_atten_flow, axis=0)

    return subset_class, subset_index, subset_atten, subset_atten_rgb, subset_atten_flow

def get_affinity(video_index, video_feature, action_2_video):
    sorted_video_index = []
    sorted_video_fea = []
    cnt = 0
    for tmp_act in action_2_video:
        for tmp_vid in action_2_video[tmp_act]:
            if tmp_vid in sorted_video_index:
                continue
            sorted_video_index.append(tmp_vid)
            tmp_vid_index = video_index.index(tmp_vid)
            sorted_video_fea.append(video_feature[tmp_vid_index])

    num_video = len(sorted_video_index)
    weight = np.zeros((num_video, num_video))
    beta = 0
    cnt = 0

    for i in range(num_video):
        for j in range(i, num_video):
            # calculate gamma
            cnt += 1
            beta += np.linalg.norm(sorted_video_fea[i] - sorted_video_fea[j])

            dis = np.square(np.linalg.norm(sorted_video_fea[i] - sorted_video_fea[j]))
            weight[i][j] = dis
            weight[j][i] = dis

    beta = beta / cnt
    gamma = - 1.0 / (2 * beta * beta)
    print("gamma is %f " % gamma)

    weight = np.exp(gamma * weight)

    return weight, sorted_video_index, sorted_video_fea


if __name__ == '__main__':
    num_subset_class = 100

    anet_label_path = "activity_net.v1-2.min-missing-removed.json"
    flow_path = "ANET_I3D_FEATURE//flow-resize-step16"
    rgb_path = "ANET_I3D_FEATURE//rgb-resize-step16"

    atten_weight_path = "your_attention_weight_path"

    anet_label = load_json(anet_label_path)
    anet_label = anet_label['database']

    all_flow_file = os.listdir(flow_path)
    all_rgb_file = os.listdir(rgb_path)

    training_index, action_2_video, video_2_action = get_train_label(anet_label)

    att_fea, att_rgb, att_flow = get_feature(
        training_index,
        flow_path,
        rgb_path,
        atten_weight_path=atten_weight_path
    )

    subset_class, subset_index, subset_atten_all, subset_atten_rgb, subset_atten_flow = get_subset(
        num_subset_class,
        action_2_video,
        training_index,
        att_fea,
        att_rgb,
        att_flow
    )

    affinity_matrix, sorted_video_index, sorted_video_fea= get_affinity(subset_index, subset_atten, action_2_video)
    subset_atten_fea = sorted_video_fea
    subset_index = sorted_video_index

    affinity_matrix_rgb, sorted_video_index_rgb, sorted_video_fea_rgb = get_affinity(subset_index, subset_atten_rgb, action_2_video)
    subset_atten_fea_rgb = sorted_video_fea_rgb
    subset_index_rgb = sorted_video_index_rgb

    affinity_matrix_flow, sorted_video_index_flow, sorted_video_fea_flow = get_affinity(subset_index, subset_atten_flow, action_2_video)
    subset_atten_fea_flow = sorted_video_fea_flow
    subset_index_flow = sorted_video_index_flow

    num_of_cluster = num_subset_class
    estimator = SpectralClustering(n_clusters=num_of_cluster, random_state=0, affinity='precomputed')
    estimator.fit_predict(affinity_matrix)
    label_pred_both = estimator.labels_

    estimator_rgb = SpectralClustering(n_clusters=num_of_cluster, random_state=0, affinity='precomputed')
    estimator_rgb.fit_predict(affinity_matrix_rgb)
    label_pred_rgb = estimator_rgb.labels_

    estimator_flow = SpectralClustering(n_clusters=num_of_cluster, random_state=0, affinity='precomputed')
    estimator_flow.fit_predict(affinity_matrix_flow)
    label_pred_flow = estimator_flow.labels_