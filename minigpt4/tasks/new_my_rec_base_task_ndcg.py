"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from collections import Counter # [新增] 用于统计 ORRatio

from minigpt4.common.dist_utils import get_rank, get_world_size, is_main_process, is_dist_avail_and_initialized
from minigpt4.common.logger import MetricLogger, SmoothedValue
from minigpt4.common.registry import registry
from minigpt4.datasets.data_utils import prepare_sample
from sklearn.metrics import roc_auc_score
from minigpt4.tasks.base_task import BaseTask


# --- (u_dcg 和 compute_u_ndcg 保持不变) ---
def u_dcg(predict, label):
    pos_num = label.sum()
    labels_unique = np.unique(label)
    if labels_unique.shape[0] < 2 or label.shape[-1] < 2:
        return -1
    ranked_id = np.argsort(-predict)
    ranked_label = label[ranked_id]
    flag = 1.0 / np.log2(np.arange(ranked_label.shape[-1]) + 2.0)
    dcg = (ranked_label * flag).sum()
    idcg = flag[:pos_num].sum()
    return dcg / idcg


def compute_u_ndcg(user, predict, label):
    predict = predict.squeeze()
    label = label.squeeze()
    start_time = time.time()
    u, inverse, counts = np.unique(user, return_inverse=True, return_counts=True)
    index = np.argsort(inverse)
    candidates_dict = {}
    k = 0
    total_num = 0
    only_one_interaction = 0
    computed_u = []
    for u_i in u:
        start_id, end_id = total_num, total_num + counts[k]
        u_i_counts = counts[k]
        index_ui = index[start_id:end_id]
        if u_i_counts == 1:
            only_one_interaction += 1
            total_num += counts[k]
            k += 1
            continue
        candidates_dict[u_i] = [predict[index_ui], label[index_ui]]
        total_num += counts[k]

        k += 1
    print("only one interaction users (for nDCG):", only_one_interaction)
    ndcg_list = []
    only_one_class = 0

    for ui, pre_and_true in candidates_dict.items():
        pre_i, label_i = pre_and_true
        ui_ndcg = u_dcg(pre_i, label_i)
        if ui_ndcg >= 0:
            ndcg_list.append(ui_ndcg)
            computed_u.append(ui)
        else:
            only_one_class += 1

    ndcg_for_user = np.array(ndcg_list)
    print("computed user (for nDCG):", ndcg_for_user.shape[0], "can not users:", only_one_class)
    u_ndcg = ndcg_for_user.mean()
    print("u-nDCG for validation Cost:", time.time() - start_time, 'u-nDCG:', u_ndcg)
    return u_ndcg, computed_u, ndcg_for_user


# --- (uAUC_me 保持不变) ---
def uAUC_me(user, predict, label):
    predict = predict.squeeze()
    label = label.squeeze()
    start_time = time.time()
    u, inverse, counts = np.unique(user, return_inverse=True, return_counts=True)
    index = np.argsort(inverse)
    candidates_dict = {}
    k = 0
    total_num = 0
    only_one_interaction = 0
    computed_u = []
    for u_i in u:
        start_id, end_id = total_num, total_num + counts[k]
        u_i_counts = counts[k]
        index_ui = index[start_id:end_id]
        if u_i_counts == 1:
            only_one_interaction += 1
            total_num += counts[k]
            k += 1
            continue
        candidates_dict[u_i] = [predict[index_ui], label[index_ui]]
        total_num += counts[k]

        k += 1
    print("only one interaction users:", only_one_interaction)
    auc = []
    only_one_class = 0

    for ui, pre_and_true in candidates_dict.items():
        pre_i, label_i = pre_and_true
        try:
            ui_auc = roc_auc_score(label_i, pre_i)
            auc.append(ui_auc)
            computed_u.append(ui)
        except ValueError:
            only_one_class += 1

    auc_for_user = np.array(auc)
    print("computed user:", auc_for_user.shape[0], "can not users:", only_one_class)
    uauc = auc_for_user.mean()
    print("uauc for validation Cost:", time.time() - start_time, 'uauc:', uauc)
    return uauc, computed_u, auc_for_user


class RecBaseTask(BaseTask):
    def valid_step(self, model, samples):
        outputs = model.generate_for_samples(samples)
        return outputs

    def before_evaluation(self, model, dataset, **kwargs):
        # 1. 获取 Item Popularity Dict
        if "item_pop_dict" in kwargs:
            self.item_pop_dict = kwargs.get("item_pop_dict", {})
            self.total_item_num = kwargs.get("total_item_num", 1)
            self.k_for_bias = kwargs.get("k_for_bias", 10)
        else:
            if not hasattr(self, "item_pop_dict"):
                self.item_pop_dict = {}
            if not hasattr(self, "total_item_num"):
                self.total_item_num = 1
            if not hasattr(self, "k_for_bias"):
                self.k_for_bias = 10

        # 2. [新增] 获取 Item Category Dict (用于 MGU 计算)
        # 这里的格式应该是 {item_id: category_id}
        if "item_category_dict" in kwargs:
            self.item_category_dict = kwargs.get("item_category_dict", {})
        else:
            if not hasattr(self, "item_category_dict"):
                self.item_category_dict = {}

        if not self.item_pop_dict:
            logging.warning("item_pop_dict not provided. Popularity metrics will be 0.")
        if not self.item_category_dict:
            logging.warning("item_category_dict not provided. MGU metric will be 0.")

    def after_evaluation(self, **kwargs):
        pass

    def evaluation(self, model, data_loaders, cuda_enabled=True):
        model = model.eval()
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        metric_logger.add_meter("acc", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        header = "Evaluation"
        print_freq = len(data_loaders.loaders[0]) // 5

        k = 0
        use_auc = False
        for data_loader in data_loaders.loaders:
            results_logits = []
            labels = []
            users = []
            item_ids = []
            histories = [] # [新增] 用于收集用户历史，计算 MGU

            for samples in metric_logger.log_every(data_loader, print_freq, header):
                samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
                eval_output = self.valid_step(model=model, samples=samples)

                if 'logits' in eval_output.keys():
                    use_auc = True

                    # 收集基础数据
                    users.append(np.atleast_1d(samples['UserID'].detach().cpu().numpy()))
                    item_ids.append(np.atleast_1d(samples['TargetItemID'].detach().cpu().numpy()))

                    # [新增] 收集用户历史交互 (InteractedItemIDs_pad)
                    # 假设 Dataset 返回了 'InteractedItemIDs_pad'，这是计算 MGU 必须的
                    if 'InteractedItemIDs_pad' in samples:
                         histories.append(samples['InteractedItemIDs_pad'].detach().cpu().numpy())
                    else:
                        # 如果没有 pad 数据，尝试用 'InteractedItemIDs' (如果是 tensor)
                        # 这里为了稳健性，如果没有历史数据，可能无法计算 MGU
                        pass

                    logits_for_metrics = eval_output['logits']
                    results_logits.append(np.atleast_1d(logits_for_metrics.detach().cpu().numpy()))
                    labels.append(np.atleast_1d(samples['label'].detach().cpu().numpy()))

                    logits_for_acc = logits_for_metrics.clone()
                    pred_labels = (logits_for_acc > 0.5).float()
                    acc = (pred_labels == samples['label']).sum() / samples['label'].shape[0]
                    metric_logger.update(acc=acc.item())
                else:
                    metric_logger.update(acc=0)

                metric_logger.update(loss=eval_output['loss'].item())
                torch.cuda.empty_cache()

            # 拼接 Numpy 数组
            results_logits_np = np.concatenate(results_logits).astype(np.float64)
            labels_np = np.concatenate(labels).astype(np.int64)
            users_np = np.concatenate(users).astype(np.int64)
            item_ids_np = np.concatenate(item_ids).astype(np.int64)

            # [新增] 拼接历史数据
            has_history = len(histories) > 0
            if has_history:
                # 注意：这里假设 batch 之间的 pad 长度是一致的，或者由 collater 统一处理
                # 如果使用了动态 padding，直接 concatenate 可能会报错 (维度不匹配)
                # 但通常 RecBaseDataset 设定了 self.max_lenght，所以维度应该是固定的
                try:
                    histories_np = np.concatenate(histories).astype(np.int64)
                except Exception as e:
                    print(f"[Warning] Failed to concatenate histories: {e}. MGU may be inaccurate.")
                    has_history = False
                    histories_np = np.array([])
            else:
                histories_np = np.array([])

            # 初始化指标
            auc = 0; uauc = 0; u_ndcg = 0
            ap_k = 0.0; coverage_k = 0.0; gini_k = 0.0
            div_ratio = 0.0; or_ratio = 0.0; mgu = 0.0 # [新增指标初始化]

            if is_dist_avail_and_initialized():
                print("wating comput metrics.....")
                results_logits_ = torch.from_numpy(results_logits_np).to(eval_output['logits'].device)
                labels_ = torch.from_numpy(labels_np).to(eval_output['logits'].device)
                users_ = torch.from_numpy(users_np).to(eval_output['logits'].device)
                item_ids_ = torch.from_numpy(item_ids_np).to(eval_output['logits'].device)

                # [新增] History Tensor
                if has_history:
                    histories_ = torch.from_numpy(histories_np).to(eval_output['logits'].device)
                else:
                    histories_ = torch.empty(0).to(eval_output['logits'].device)

                rank = dist.get_rank()
                world_size = dist.get_world_size()

                gathered_labels = [torch.zeros_like(labels_) for _ in range(world_size)]
                gathered_logits = [torch.zeros_like(results_logits_) for _ in range(world_size)]
                gathered_users = [torch.zeros_like(users_) for _ in range(world_size)]
                gathered_item_ids = [torch.zeros_like(item_ids_) for _ in range(world_size)]

                # [新增] History Gather
                # 注意：如果 history 维度很大，all_gather 可能会显存溢出
                if has_history:
                    gathered_histories = [torch.zeros_like(histories_) for _ in range(world_size)]

                dist.all_gather(gathered_labels, labels_)
                dist.all_gather(gathered_logits, results_logits_)
                dist.all_gather(gathered_users, users_)
                dist.all_gather(gathered_item_ids, item_ids_)

                if has_history:
                    dist.all_gather(gathered_histories, histories_)
                    histories_a = torch.cat(gathered_histories, dim=0).cpu().numpy()
                else:
                    histories_a = np.array([])

                labels_a = torch.cat(gathered_labels, dim=0).flatten().cpu().numpy()
                results_logits_a = torch.cat(gathered_logits, dim=0).flatten().cpu().numpy()
                users_a = torch.cat(gathered_users, dim=0).flatten().cpu().numpy()
                item_ids_a = torch.cat(gathered_item_ids, dim=0).flatten().cpu().numpy()

                print("computing metrics....")
                auc = roc_auc_score(labels_a, results_logits_a)
                uauc, _, _ = uAUC_me(users_a, results_logits_a, labels_a)
                u_ndcg, _, _ = compute_u_ndcg(users_a, results_logits_a, labels_a)

                # 计算所有偏差与多样性指标
                metrics_dict = self.calculate_all_advanced_metrics(
                    users_a, item_ids_a, results_logits_a, histories_a
                )
                print("finished comput metrics.....")
            else:
                auc = roc_auc_score(labels_np, results_logits_np)
                uauc, _, _ = uAUC_me(users_np, results_logits_np, labels_np)
                u_ndcg, _, _ = compute_u_ndcg(users_np, results_logits_np, labels_np)

                # 计算所有偏差与多样性指标
                metrics_dict = self.calculate_all_advanced_metrics(
                    users_np, item_ids_np, results_logits_np, histories_np
                )

            if is_dist_avail_and_initialized():
                dist.barrier()

            metric_logger.synchronize_between_processes()

            # 解析指标
            ap_k = metrics_dict.get('AP', 0.0)
            coverage_k = metrics_dict.get('Coverage', 0.0)
            gini_k = metrics_dict.get('Gini', 0.0)
            div_ratio = metrics_dict.get('DivRatio', 0.0)
            or_ratio = metrics_dict.get('ORRatio', 0.0)
            mgu = metrics_dict.get('MGU', 0.0)

            auc_rank0 = 0
            if use_auc and not is_dist_avail_and_initialized():
                auc_rank0 = roc_auc_score(labels_np, results_logits_np)

            logging.info(
                "Averaged stats: " + str(metric_logger.global_avg()) +
                " ***auc: " + str(auc) +
                " ***uauc: " + str(uauc) +
                " ***u-nDCG: " + str(u_ndcg) +
                f" ***AP@{self.k_for_bias}: " + str(ap_k) +
                f" ***Coverage@{self.k_for_bias}: " + str(coverage_k) +
                f" ***Gini@{self.k_for_bias}: " + str(gini_k) +
                # [新增日志]
                f" ***DivRatio@{self.k_for_bias}: " + str(div_ratio) +
                f" ***ORRatio@{self.k_for_bias}: " + str(or_ratio) +
                f" ***MGU@{self.k_for_bias}: " + str(mgu)
            )
            print("rank_0 auc:", str(auc_rank0))

            if use_auc:
                results = {
                    'agg_metrics': auc,
                    'acc': metric_logger.meters['acc'].global_avg,
                    'loss': metric_logger.meters['loss'].global_avg,
                    'auc': auc,
                    'uauc': uauc,
                    'u_ndcg': u_ndcg,
                    f'AP@{self.k_for_bias}': ap_k,
                    f'Coverage@{self.k_for_bias}': coverage_k,
                    f'Gini@{self.k_for_bias}': gini_k,
                    # [新增返回]
                    f'DivRatio@{self.k_for_bias}': div_ratio,
                    f'ORRatio@{self.k_for_bias}': or_ratio,
                    f'MGU@{self.k_for_bias}': mgu
                }
            else:
                results = {
                    'agg_metrics': -metric_logger.meters['loss'].global_avg,
                }

        return results

    # --- [新增] 计算 Gini 系数 ---
    def calculate_gini(self, item_counts):
        if item_counts.empty:
            return 0.0
        vals = item_counts.sort_values().values
        n = len(vals)
        cum_vals = np.cumsum(vals)
        total_sum = cum_vals[-1]
        if total_sum == 0:
            return 0.0
        lorenz_curve = cum_vals / total_sum
        area_under_lorenz = np.sum(lorenz_curve) / n
        area_perfect_equality = 0.5
        gini_coefficient = (area_perfect_equality - area_under_lorenz) / area_perfect_equality
        return gini_coefficient

    # --- [核心修改] 统一计算所有高级指标 ---
    def calculate_all_advanced_metrics(self, uids, item_ids, scores, histories=None):
        """
        计算 AP, Coverage, Gini 以及新增的 DivRatio, ORRatio, MGU
        """
        K = self.k_for_bias
        pop_dict = self.item_pop_dict
        cat_dict = self.item_category_dict # 需要类别字典计算 MGU
        total_item_num = self.total_item_num

        if not pop_dict:
            logging.warning("item_pop_dict is empty. Metrics may be 0.")
            # 即使没有 pop_dict，也可以计算 DivRatio 和 ORRatio，但 AP 会是 0

        # 1. 组装数据
        df = pd.DataFrame({
            'uid': uids,
            'item_id': item_ids,
            'score': scores
        })

        # 如果有历史数据，构建映射 {uid: [item_ids...]}
        user_history_map = {}
        if histories is not None and len(histories) > 0:
            # histories shape: [N_samples, Max_Len]
            # 我们需要把它和 uid 对应起来
            # 注意：histories 和 uids 是一一对应的 (Sample 级别)
            # 我们需要将其聚合到 User 级别
            temp_hist_df = pd.DataFrame({'uid': uids})
            # 这里比较 tricky，因为 histories 是二维数组，不能直接作为列放进 groupby
            # 我们用 index 来辅助
            temp_hist_df['hist_idx'] = range(len(temp_hist_df))

            grouped_hist = temp_hist_df.groupby('uid')['hist_idx'].first()
            # 假设每个用户的历史在 valid set 里是一样的，取第一条即可

            for uid, h_idx in grouped_hist.items():
                # 去除 padding (假设 0 是 padding)
                raw_hist = histories[h_idx]
                user_history_map[uid] = raw_hist[raw_hist != 0]

        all_top_k_items = []
        all_top_k_pops = []
        mgu_scores = [] # 存储每个用户的 MGU 距离

        grouped = df.groupby('uid')

        unique_users_count = 0

        for uid, user_df in grouped:
            unique_users_count += 1

            # Top-K 获取
            if len(user_df) < K:
                top_k_df = user_df
            else:
                top_k_df = user_df.nlargest(K, 'score')

            top_k_item_ids = top_k_df['item_id'].values
            all_top_k_items.extend(top_k_item_ids)

            # --- AP 计算 ---
            for item_id in top_k_item_ids:
                all_top_k_pops.append(pop_dict.get(item_id, 0))

            # --- MGU 计算 (Mean Group Utility / Category Fairness) ---
            # 定义：用户历史类别分布 vs 推荐列表类别分布 的差异 (通常用 L1 或 KL)
            if cat_dict and uid in user_history_map:
                hist_items = user_history_map[uid]
                rec_items = top_k_item_ids

                if len(hist_items) > 0 and len(rec_items) > 0:
                    # 统计历史类别分布
                    hist_cats = [cat_dict.get(i, -1) for i in hist_items if i in cat_dict]
                    rec_cats = [cat_dict.get(i, -1) for i in rec_items if i in cat_dict]

                    if len(hist_cats) > 0 and len(rec_cats) > 0:
                        hist_counts = Counter(hist_cats)
                        rec_counts = Counter(rec_cats)

                        all_cats = set(hist_counts.keys()) | set(rec_counts.keys())

                        dist_sum = 0.0
                        for c in all_cats:
                            p_hist = hist_counts.get(c, 0) / len(hist_cats)
                            p_rec = rec_counts.get(c, 0) / len(rec_cats)
                            dist_sum += abs(p_hist - p_rec) # L1 Distance

                        mgu_scores.append(dist_sum)
                    else:
                        mgu_scores.append(0.0) # 无法匹配到类别
                else:
                    mgu_scores.append(0.0) # 历史为空

        # 2. 指标聚合

        # [AP]
        ap_k = np.mean(all_top_k_pops) if all_top_k_pops else 0.0

        # [Coverage]
        unique_recommended_items = set(all_top_k_items)
        coverage_k = len(unique_recommended_items) / total_item_num

        # [Gini]
        item_counts = pd.Series(all_top_k_items).value_counts()
        gini_k = self.calculate_gini(item_counts)

        # --- [新增] DivRatio (Diversity Ratio) ---
        # 定义：Unique recommendations / Total recommendations (measure of redundancy)
        # 或者：Unique recommendations / (Users * K)
        total_recs = len(all_top_k_items)
        div_ratio = len(unique_recommended_items) / total_recs if total_recs > 0 else 0.0

        # --- [新增] ORRatio (Over-Recommendation Ratio) ---
        # 定义：Top-3 most recommended items' count / Total recommendations
        if total_recs > 0:
            most_common_3 = item_counts.iloc[:3].sum() # item_counts 已经是 value_counts 排序过的
            or_ratio = most_common_3 / total_recs
        else:
            or_ratio = 0.0

        # --- [新增] MGU (Mean Group Utility) ---
        # 定义：Average of user-level category distribution difference
        mgu = np.mean(mgu_scores) if mgu_scores else 0.0

        print(f"Metrics @{K}: AP={ap_k:.4f}, Cov={coverage_k:.4f}, Gini={gini_k:.4f}")
        print(f"Advanced @{K}: DivRatio={div_ratio:.4f}, ORRatio={or_ratio:.4f}, MGU={mgu:.4f}")

        return {
            'AP': ap_k,
            'Coverage': coverage_k,
            'Gini': gini_k,
            'DivRatio': div_ratio,
            'ORRatio': or_ratio,
            'MGU': mgu
        }