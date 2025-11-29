import os
from select import select
# from PIL import Image
# import webdataset as wds
from minigpt4.datasets.datasets.rec_base_dataset import RecBaseDataset
import pandas as pd
import numpy as np
import logging


# from minigpt4.datasets.datasets.caption_datasets import CaptionDataset


# class RecDataset(RecBaseDataset):


#     def __getitem__(self, index):

#         # TODO this assumes image input, not general enough
#         ann = self.annotation.iloc[index]
#         return {
#             "User": ann['User'],
#             "InteractedItems": ann['InteractedItems'],
#             "InteractedItemTitles": ann['InteractedItemTitles'],
#             "TargetItemID": ann["TargetItemID"],
#             "TargetItemTitle": ann["TargetItemTitle"]
#         }


def convert_title_list_v2(titles):
    titles_ = []
    for x in titles:
        if len(x) > 0:
            titles_.append("\"" + x + "\"")
    if len(titles_) > 0:
        return ", ".join(titles_)
    else:
        return "unkow"


def convert_title_list(titles):
    titles = ["\"" + x + "\"" for x in titles]
    return ", ".join(titles)


class MovielensDataset(RecBaseDataset):
    def __init__(self, text_processor=None, ann_paths=None):
        super().__init__(text_processor, ann_paths)
        self.annotation = pd.read_pickle(ann_paths[0] + ".pkl").reset_index(drop=True)
        self.use_his = False
        if 'sessionItems' in self.annotation.columns:
            self.use_his = True
            self.annotation = self.annotation[['uid', 'iid', 'title', 'sessionItems', 'sessionItemTitles', 'label']]
            self.annotation.columns = ['UserID', 'TargetItemID', 'TargetItemTitle', 'InteractedItemIDs',
                                       'InteractedItemTitles', 'label']

            self.annotation["InteractedItemIDs"] = self.annotation["InteractedItemIDs"].map(list)
            self.annotation["InteractedItemTitles"] = self.annotation["InteractedItemTitles"].map(list)
            self.annotation["InteractedItemTitles"] = self.annotation["InteractedItemTitles"].map(convert_title_list)
        else:
            self.annotation = self.annotation[['uid', 'iid', 'title', 'label']]
            self.annotation.columns = ['UserID', 'TargetItemID', 'TargetItemTitle', 'label']

        print("data path:", ann_paths[0], "data size:", self.annotation.shape)
        self.user_num = self.annotation['UserID'].max() + 1
        self.item_num = self.annotation['TargetItemID'].max() + 1
        self.text_processor = text_processor

        if self.use_his:
            max_length = 0
            for x in self.annotation['InteractedItemIDs'].values:
                max_length = max(max_length, len(x))
            self.max_lenght = max_length

    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation.iloc[index]
        if self.use_his:
            a = ann['InteractedItemIDs']
            InteractedNum = len(a)
            if len(a) < self.max_lenght:
                b = [0] * (self.max_lenght - len(a))  # assuming padding idx is zero
                b.extend(a)
            else:
                b = a
            return {
                "UserID": ann['UserID'],
                "InteractedItemIDs_pad": np.array(b),
                # "InteractedItemIDs": ann['InteractedItemIDs'],
                "InteractedItemTitles": ann['InteractedItemTitles'],
                "TargetItemID": ann["TargetItemID"],
                "TargetItemTitle": "\"" + ann["TargetItemTitle"] + "\"",
                "InteractedNum": InteractedNum,
                "label": ann['label']
            }
        else:
            return {
                "UserID": ann['UserID'],
                # "InteractedItemIDs_pad": None,
                # # "InteractedItemIDs": ann['InteractedItemIDs'],
                # "InteractedItemTitles": None,
                "TargetItemID": ann["TargetItemID"],
                "TargetItemTitle": "\"" + ann["TargetItemTitle"] + "\"",
                # "InteractedNum": None,
                "label": ann['label']
            }


class MovielensDataset_stage1(RecBaseDataset):
    def __init__(self, text_processor=None, ann_paths=None):
        super().__init__(text_processor, ann_paths)
        # self.vis_root = vis_root
        # self.annotation = pd.read_csv(ann_paths[0],sep='\t', index_col=None,header=0)[['uid','iid','title','sessionItems', 'sessionItemTitles']]
        self.annotation = pd.read_pickle(ann_paths[0] + ".pkl").reset_index(drop=True)[
            ['uid', 'iid', 'title', 'sessionItems', 'sessionItemTitles', 'label', 'pairItems', 'pairItemTitles']]
        self.annotation.columns = ['UserID', 'TargetItemID', 'TargetItemTitle', 'InteractedItemIDs',
                                   'InteractedItemTitles', 'label', 'PairItemIDs', 'PairItemTitles']
        self.annotation["InteractedItemIDs"] = self.annotation["InteractedItemIDs"].map(list)
        self.annotation["InteractedItemTitles"] = self.annotation["InteractedItemTitles"].map(list)
        self.annotation["InteractedItemTitles"] = self.annotation["InteractedItemTitles"].map(convert_title_list)
        self.annotation["PairItemTitles"] = self.annotation["PairItemTitles"].map(convert_title_list)

        self.user_num = self.annotation['UserID'].max() + 1
        self.item_num = self.annotation['TargetItemID'].max() + 1
        self.text_processor = text_processor

    def __getitem__(self, index):
        # TODO this assumes image input, not general enough
        ann = self.annotation.iloc[index]
        return {
            "UserID": ann['UserID'],
            "PairItemIDs": np.array(ann['PairItemIDs']),
            "PairItemTitles": ann["PairItemTitles"],
            "label": ann['label']
        }


class AmazonDataset(RecBaseDataset):
    def __init__(self, text_processor=None, ann_paths=None):
        super().__init__()
        # self.vis_root = vis_root
        # self.annotation = pd.read_csv(ann_paths[0],sep='\t', index_col=None,header=0)[['uid','iid','title','sessionItems', 'sessionItemTitles']]
        self.annotation = pd.read_pickle(ann_paths[0] + "_seqs.pkl").reset_index(drop=True)
        self.use_his = False
        if 'sessionItems' in self.annotation.columns or 'his' in self.annotation.columns:
            self.use_his = True
            self.annotation = self.annotation[['uid', 'iid', 'title', 'his', 'his_title', 'label']]
            self.annotation.columns = ['UserID', 'TargetItemID', 'TargetItemTitle', 'InteractedItemIDs',
                                       'InteractedItemTitles', 'label']

            self.annotation["InteractedItemIDs"] = self.annotation["InteractedItemIDs"].map(list)
            self.annotation["InteractedItemTitles"] = self.annotation["InteractedItemTitles"].map(list)
            self.annotation["InteractedItemTitles"] = self.annotation[
                "InteractedItemTitles"]  # .map(convert_title_list)
        else:
            self.annotation = self.annotation[['uid', 'iid', 'title', 'label']]
            self.annotation.columns = ['UserID', 'TargetItemID', 'TargetItemTitle', 'label']

        print("data path:", ann_paths[0], "data size:", self.annotation.shape)
        self.user_num = self.annotation['UserID'].max() + 1
        self.item_num = self.annotation['TargetItemID'].max() + 1
        self.text_processor = text_processor

        if self.use_his:
            max_length_ = 0
            for x in self.annotation['InteractedItemIDs'].values:
                max_length_ = max(max_length_, len(x))
            self.max_lenght = min(max_length_, 15)  # average: only 5
            print("amazon datasets, max history length:", self.max_lenght)
            logging.info("amazon datasets, max history length:" + str(self.max_lenght))

    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation.iloc[index]
        if self.use_his:
            a = ann['InteractedItemIDs']
            InteractedNum = len(a)
            if a[0] == 0:
                InteractedNum -= 1

            if len(a) < self.max_lenght:
                b = [0] * (self.max_lenght - len(a))  # assuming padding idx is zero
                b.extend(a)
            elif len(a) > self.max_lenght:
                b = a[-self.max_lenght:]
                InteractedNum = self.max_lenght
            else:
                b = a
            return {
                "UserID": ann['UserID'],
                "InteractedItemIDs_pad": np.array(b),
                # "InteractedItemIDs": ann['InteractedItemIDs'],
                "InteractedItemTitles": convert_title_list_v2(ann['InteractedItemTitles'][-InteractedNum:]),
                "TargetItemID": ann["TargetItemID"],
                "TargetItemTitle": "\"" + ann["TargetItemTitle"] + "\"",
                "InteractedNum": InteractedNum,
                "label": ann['label']
            }
        else:
            return {
                "UserID": ann['UserID'],
                # "InteractedItemIDs_pad": None,
                # # "InteractedItemIDs": ann['InteractedItemIDs'],
                # "InteractedItemTitles": None,
                "TargetItemID": ann["TargetItemID"],
                "TargetItemTitle": ann["TargetItemTitle"],
                # "InteractedNum": None,
                "label": ann['label']
            }


class MoiveOOData(RecBaseDataset):
    def __init__(self, text_processor=None, ann_paths=None):
        super().__init__()
        # 1. 加载数据
        ann_paths_real = ann_paths[0].split("=")[0]  # 确保路径处理正确
        self.annotation = pd.read_pickle(ann_paths_real + "_ood2.pkl").reset_index(drop=True)

        # 2. 筛选 Warm/Cold
        if "warm" in ann_paths:
            self.annotation = self.annotation[self.annotation['warm'].isin([1])].copy()
        if "cold" in ann_paths:
            self.annotation = self.annotation[self.annotation['not_cold'].isin([0])].copy()

        self.use_his = False
        self.prompt_flag = False

        # --- [步骤 A] 先处理列名，确保 'TargetItemID' 存在 ---
        if 'sessionItems' in self.annotation.columns or 'his' in self.annotation.columns:
            used_columns = ['uid', 'iid', 'title', 'his', 'his_title', 'label']
            renamed_columns = ['UserID', 'TargetItemID', 'TargetItemTitle', 'InteractedItemIDs', 'InteractedItemTitles',
                               'label']
            if 'not_cold' in self.annotation.columns:
                used_columns.append("not_cold")
                renamed_columns.append("prompt_flag")
                self.prompt_flag = True

            self.use_his = True
            self.annotation = self.annotation[used_columns]
            self.annotation.columns = renamed_columns  # <--- 列名变更完成

            self.annotation["InteractedItemIDs"] = self.annotation["InteractedItemIDs"].map(list)
            self.annotation["InteractedItemTitles"] = self.annotation["InteractedItemTitles"].map(list)
        else:
            used_columns = ['uid', 'iid', 'title', 'label']
            renamed_columns = ['UserID', 'TargetItemID', 'TargetItemTitle', 'label']
            if 'not_cold' in self.annotation.columns:
                used_columns.append('not_cold')
                renamed_columns.append("prompt_flag")
                self.prompt_flag = True
            self.annotation = self.annotation[used_columns]
            self.annotation.columns = renamed_columns  # <--- 列名变更完成

        # --- [步骤 B] 必须先计算 item_num (修复 AttributeError) ---
        self.user_num = self.annotation['UserID'].max() + 1
        self.item_num = self.annotation['TargetItemID'].max() + 1

        # --- [步骤 C] 现在可以使用 item_num 和 TargetItemID 计算流行度了 ---

        # 1. 统计交互
        self.item_counts = self.annotation['TargetItemID'].value_counts()
        all_items = self.item_counts.index.tolist()

        # 2. 划分热门/长尾 (Top 20%)
        # 使用 len(all_items) 通常比 item_num 更准确，因为 ID 可能不连续，
        # 但为了配合之前的逻辑，这里计算 split_idx
        split_idx = int(len(all_items) * 0.2)

        self.popular_items = all_items[:split_idx]
        self.tail_items = all_items[split_idx:]

        # 3. 生成查找表 (0: Rare, 2: Hot)
        self.id2pop_level = {}
        # 默认全部设为 0 (Rare)
        for iid in all_items:
            self.id2pop_level[iid] = 0
        # 将热门的设为 2 (Hot)
        for iid in self.popular_items:
            self.id2pop_level[iid] = 2

        print(f"流行度计算完成: Top 20% 标记为热门(Level 2)，其余为长尾(Level 0)。")

        # --- 后续逻辑 ---
        print("data path:", ann_paths[0], "data size:", self.annotation.shape)
        self.text_processor = text_processor

        if self.use_his:
            max_length_ = 0
            for x in self.annotation['InteractedItemIDs'].values:
                max_length_ = max(max_length_, len(x))
            self.max_lenght = min(max_length_, 10)
            print("Movie OOD datasets, max history length:", self.max_lenght)
            logging.info("Movie OOD datasets, max history length:" + str(self.max_lenght))

    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation.iloc[index]
        if self.use_his:
            a = ann['InteractedItemIDs']
            InteractedNum = len(a)
            if a[0] == 0:
                InteractedNum -= 1

            if len(a) < self.max_lenght:
                b = [0] * (self.max_lenght - len(a))  # assuming padding idx is zero
                b.extend(a)
            elif len(a) > self.max_lenght:
                b = a[-self.max_lenght:]
                InteractedNum = self.max_lenght
            else:
                b = a
            # 获取正样本信息
            pos_item_id = ann["TargetItemID"]

            # --- [新增] 负采样逻辑 (仅训练时) ---
            # 注意：这里简单起见，随机采样直到不等于正样本且不在历史中
            # 实际高效实现建议预计算负样本列表

            # 1. 采样热门负样本 (Popular Negative)
            neg_pop_id = np.random.choice(self.popular_items)
            while neg_pop_id == pos_item_id or neg_pop_id in ann['InteractedItemIDs']:
                neg_pop_id = np.random.choice(self.popular_items)

            # 2. 采样长尾负样本 (Tail Negative)
            neg_tail_id = np.random.choice(self.tail_items)
            while neg_tail_id == pos_item_id or neg_tail_id in ann['InteractedItemIDs']:
                neg_tail_id = np.random.choice(self.tail_items)

            # 3. 获取流行度文本描述 (用于 Prompt)
            # 例如: "Extremely Popular", "Niche", "Common"
            pop_desc_map = {0: "rare", 2: "highly popular"}

            # 查表，默认值为 0 (rare)
            level = self.id2pop_level.get(pos_item_id, 0)
            pos_pop_desc = pop_desc_map[level]

            neg_pop_desc = "highly popular"  # 热门负样本永远是 popular
            neg_tail_desc = "rare"  # 长尾负样本永远是 rare
            one_sample = {
                "UserID": ann['UserID'],
                "InteractedItemIDs_pad": np.array(b),
                # "InteractedItemIDs": ann['InteractedItemIDs'],
                "InteractedItemTitles": convert_title_list_v2(ann['InteractedItemTitles'][-InteractedNum:]),
                "TargetItemID": ann["TargetItemID"],
                "TargetItemTitle": "\"" + ann["TargetItemTitle"].strip(' ') + "\"",
                "InteractedNum": InteractedNum,
                "label": ann['label'],
                "PopDesc": pos_pop_desc,
                "NegPopID": neg_pop_id,
                "NegPopDesc": neg_pop_desc,
                "NegTailID": neg_tail_id,
                "NegTailDesc": neg_tail_desc
            }
            if self.prompt_flag:
                one_sample['prompt_flag'] = ann['prompt_flag']
            return one_sample
        else:
            one_sample = {
                "UserID": ann['UserID'],
                # "InteractedItemIDs_pad": None,
                # # "InteractedItemIDs": ann['InteractedItemIDs'],
                # "InteractedItemTitles": None,
                "TargetItemID": ann["TargetItemID"],
                "TargetItemTitle": ann["TargetItemTitle"].strip(' '),
                # "InteractedNum": None,
                "label": ann['label']
            }
            if self.prompt_flag:
                one_sample['prompt_flag'] = ann['prompt_flag']
            return one_sample


class MoiveOOData_sasrec(RecBaseDataset):
    def __init__(self, text_processor=None, ann_paths=None, sas_seq_len=25):
        super().__init__()
        # self.vis_root = vis_root
        # self.annotation = pd.read_csv(ann_paths[0],sep='\t', index_col=None,header=0)[['uid','iid','title','sessionItems', 'sessionItemTitles']]
        ann_paths = ann_paths[0].split("=")
        self.annotation = pd.read_pickle(ann_paths[0] + "_ood2.pkl").reset_index(drop=True)
        # self.annotation = pd.read_pickle(ann_paths[0]+"_ood2.pkl").reset_index(drop=True)

        self.use_his = False
        self.prompt_flag = False
        self.sas_seq_len = sas_seq_len

        if 'sessionItems' in self.annotation.columns or 'his' in self.annotation.columns:
            used_columns = ['uid', 'iid', 'title', 'his', 'his_title', 'label']
            renamed_columns = ['UserID', 'TargetItemID', 'TargetItemTitle', 'InteractedItemIDs', 'InteractedItemTitles',
                               'label']
            if 'not_cold' in self.annotation.columns:
                used_columns.append("not_cold")
                renamed_columns.append("prompt_flag")
                self.prompt_flag = True

            self.use_his = True
            self.annotation = self.annotation[used_columns]
            self.annotation.columns = renamed_columns

            self.annotation["InteractedItemIDs"] = self.annotation["InteractedItemIDs"].map(list)
            self.annotation["InteractedItemTitles"] = self.annotation["InteractedItemTitles"].map(list)
            self.annotation["InteractedItemTitles"] = self.annotation[
                "InteractedItemTitles"]  # .map(convert_title_list)
        else:
            used_columns = ['uid', 'iid', 'title', 'label']
            renamed_columns = ['UserID', 'TargetItemID', 'TargetItemTitle', 'label']
            if 'not_cold' in self.annotation.columns:
                used_columns.append('not_cold')
                renamed_columns.append("prompt_flag")
                self.prompt_flag = True
            self.annotation = self.annotation[used_columns]
            self.annotation.columns = renamed_columns

        print("data path:", ann_paths[0], "data size:", self.annotation.shape)
        self.user_num = self.annotation['UserID'].max() + 1
        self.item_num = self.annotation['TargetItemID'].max() + 1
        self.text_processor = text_processor

        if self.use_his:
            max_length_ = 0
            for x in self.annotation['InteractedItemIDs'].values:
                max_length_ = max(max_length_, len(x))
            self.max_lenght = min(max_length_, 10)  # average: only 50; 0915: 15
            print("Movie OOD datasets, max history length:", self.max_lenght)
            logging.info("Movie OOD datasets, max history length:" + str(self.max_lenght))

    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation.iloc[index]
        if self.use_his:
            a = ann['InteractedItemIDs']
            InteractedNum = len(a)
            if a[0] == 0:
                InteractedNum -= 1

            if len(a) < self.max_lenght:
                b = [0] * (self.max_lenght - len(a))  # assuming padding idx is zero
                b.extend(a)
            elif len(a) > self.max_lenght:
                b = a[-self.max_lenght:]
                InteractedNum = self.max_lenght
            else:
                b = a

            if len(a) < self.sas_seq_len:  # used for sasrec
                c = [0] * (self.sas_seq_len - len(a))
                c.extend(a)
            elif len(a) >= self.sas_seq_len:
                c = a[-self.sas_seq_len:]

            one_sample = {
                "UserID": ann['UserID'],
                "InteractedItemIDs_pad": np.array(b),
                # "InteractedItemIDs": ann['InteractedItemIDs'],
                "InteractedItemTitles": convert_title_list_v2(ann['InteractedItemTitles'][-InteractedNum:]),
                "TargetItemID": ann["TargetItemID"],
                "TargetItemTitle": "\"" + ann["TargetItemTitle"].strip(' ') + "\"",
                "InteractedNum": InteractedNum,
                "label": ann['label'],
                "sas_seq": np.array(c)
            }
            if self.prompt_flag:
                one_sample['prompt_flag'] = ann['prompt_flag']
            return one_sample
        else:
            one_sample = {
                "UserID": ann['UserID'],
                # "InteractedItemIDs_pad": None,
                # # "InteractedItemIDs": ann['InteractedItemIDs'],
                # "InteractedItemTitles": None,
                "TargetItemID": ann["TargetItemID"],
                "TargetItemTitle": ann["TargetItemTitle"].strip(' '),
                # "InteractedNum": None,
                "label": ann['label']
            }
            if self.prompt_flag:
                one_sample['prompt_flag'] = ann['prompt_flag']
            return one_sample


class AmazonOOData(RecBaseDataset):
    def __init__(self, text_processor=None, ann_paths=None):
        super().__init__()
        # self.vis_root = vis_root
        # self.annotation = pd.read_csv(ann_paths[0],sep='\t', index_col=None,header=0)[['uid','iid','title','sessionItems', 'sessionItemTitles']]
        ann_paths = ann_paths[0].split('=')
        self.annotation = pd.read_pickle(ann_paths[0] + "_ood2.pkl").reset_index(drop=True)
        self.use_his = False
        self.prompt_flag = False

        # ## warm test:

        if 'not_cold' in self.annotation.columns and "warm" in ann_paths:
            self.annotation = self.annotation[self.annotation['not_cold'].isin([1])].copy()
        if 'not_cold' in self.annotation.columns and "cold" in ann_paths:
            self.annotation = self.annotation[self.annotation['not_cold'].isin([0])].copy()

        if 'sessionItems' in self.annotation.columns or 'his' in self.annotation.columns:
            used_columns = ['uid', 'iid', 'title', 'his', 'his_title', 'label']
            renamed_columns = ['UserID', 'TargetItemID', 'TargetItemTitle', 'InteractedItemIDs', 'InteractedItemTitles',
                               'label']
            if 'not_cold' in self.annotation.columns:
                used_columns.append("not_cold")
                renamed_columns.append("prompt_flag")
                self.prompt_flag = True

            self.use_his = True
            self.annotation = self.annotation[used_columns]
            self.annotation.columns = renamed_columns

            self.annotation["InteractedItemIDs"] = self.annotation["InteractedItemIDs"].map(list)
            self.annotation["InteractedItemTitles"] = self.annotation["InteractedItemTitles"].map(list)
            self.annotation["InteractedItemTitles"] = self.annotation[
                "InteractedItemTitles"]  # .map(convert_title_list)
        else:
            used_columns = ['uid', 'iid', 'title', 'label']
            renamed_columns = ['UserID', 'TargetItemID', 'TargetItemTitle', 'label']
            if 'not_cold' in self.annotation.columns:
                used_columns.append('not_cold')
                renamed_columns.append("prompt_flag")
                self.prompt_flag = True
            self.annotation = self.annotation[used_columns]
            self.annotation.columns = renamed_columns

        print("data path:", ann_paths[0], "data size:", self.annotation.shape)
        self.user_num = self.annotation['UserID'].max() + 1
        self.item_num = self.annotation['TargetItemID'].max() + 1
        self.text_processor = text_processor

        if self.use_his:
            max_length_ = 0
            for x in self.annotation['InteractedItemIDs'].values:
                max_length_ = max(max_length_, len(x))
            self.max_lenght = min(max_length_, 10)  # average: only 50; 0915: 15
            print("Movie OOD datasets, max history length:", self.max_lenght)
            logging.info("Movie OOD datasets, max history length:" + str(self.max_lenght))

    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation.iloc[index]
        if self.use_his:
            a = ann['InteractedItemIDs']
            InteractedNum = len(a)
            if a[0] == 0:
                InteractedNum -= 1

            if len(a) < self.max_lenght:
                b = [0] * (self.max_lenght - len(a))  # assuming padding idx is zero
                b.extend(a)
            elif len(a) > self.max_lenght:
                b = a[-self.max_lenght:]
                InteractedNum = self.max_lenght
            else:
                b = a
            one_sample = {
                "UserID": ann['UserID'],
                "InteractedItemIDs_pad": np.array(b),
                # "InteractedItemIDs": ann['InteractedItemIDs'],
                "InteractedItemTitles": convert_title_list_v2(ann['InteractedItemTitles'][-InteractedNum:]),
                "TargetItemID": ann["TargetItemID"],
                "TargetItemTitle": "\"" + ann["TargetItemTitle"].strip(' ') + "\"",
                "InteractedNum": InteractedNum,
                "label": ann['label']
            }
            if self.prompt_flag:
                one_sample['prompt_flag'] = ann['prompt_flag']
            return one_sample
        else:
            one_sample = {
                "UserID": ann['UserID'],
                # "InteractedItemIDs_pad": None,
                # # "InteractedItemIDs": ann['InteractedItemIDs'],
                # "InteractedItemTitles": None,
                "TargetItemID": ann["TargetItemID"],
                "TargetItemTitle": ann["TargetItemTitle"].strip(' '),
                # "InteractedNum": None,
                "label": ann['label']
            }
            if self.prompt_flag:
                one_sample['prompt_flag'] = ann['prompt_flag']
            return one_sample


class AmazonOOData_sasrec(RecBaseDataset):
    def __init__(self, text_processor=None, ann_paths=None, sas_seq_len=20):
        super().__init__()
        # self.vis_root = vis_root
        # self.annotation = pd.read_csv(ann_paths[0],sep='\t', index_col=None,header=0)[['uid','iid','title','sessionItems', 'sessionItemTitles']]
        self.annotation = pd.read_pickle(ann_paths[0] + "_ood2.pkl").reset_index(drop=True)

        self.use_his = False
        self.prompt_flag = False
        self.sas_seq_len = sas_seq_len

        if 'sessionItems' in self.annotation.columns or 'his' in self.annotation.columns:
            used_columns = ['uid', 'iid', 'title', 'his', 'his_title', 'label']
            renamed_columns = ['UserID', 'TargetItemID', 'TargetItemTitle', 'InteractedItemIDs', 'InteractedItemTitles',
                               'label']
            if 'not_cold' in self.annotation.columns:
                used_columns.append("not_cold")
                renamed_columns.append("prompt_flag")
                self.prompt_flag = True

            self.use_his = True
            self.annotation = self.annotation[used_columns]
            self.annotation.columns = renamed_columns

            self.annotation["InteractedItemIDs"] = self.annotation["InteractedItemIDs"].map(list)
            self.annotation["InteractedItemTitles"] = self.annotation["InteractedItemTitles"].map(list)
            self.annotation["InteractedItemTitles"] = self.annotation[
                "InteractedItemTitles"]  # .map(convert_title_list)
        else:
            used_columns = ['uid', 'iid', 'title', 'label']
            renamed_columns = ['UserID', 'TargetItemID', 'TargetItemTitle', 'label']
            if 'not_cold' in self.annotation.columns:
                used_columns.append('not_cold')
                renamed_columns.append("prompt_flag")
                self.prompt_flag = True
            self.annotation = self.annotation[used_columns]
            self.annotation.columns = renamed_columns

        print("data path:", ann_paths[0], "data size:", self.annotation.shape)
        self.user_num = self.annotation['UserID'].max() + 1
        self.item_num = self.annotation['TargetItemID'].max() + 1
        self.text_processor = text_processor

        if self.use_his:
            max_length_ = 0
            for x in self.annotation['InteractedItemIDs'].values:
                max_length_ = max(max_length_, len(x))
            self.max_lenght = min(max_length_, 10)  # average: only 50; 0915: 15
            print("Movie OOD datasets, max history length:", self.max_lenght)
            logging.info("Movie OOD datasets, max history length:" + str(self.max_lenght))

    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation.iloc[index]
        if self.use_his:
            a = ann['InteractedItemIDs']
            InteractedNum = len(a)
            if a[0] == 0:
                InteractedNum -= 1

            if len(a) < self.max_lenght:
                b = [0] * (self.max_lenght - len(a))  # assuming padding idx is zero
                b.extend(a)
            elif len(a) > self.max_lenght:
                b = a[-self.max_lenght:]
                InteractedNum = self.max_lenght
            else:
                b = a

            if len(a) < self.sas_seq_len:  # used for sasrec
                c = [0] * (self.sas_seq_len - len(a))
                c.extend(a)
            elif len(a) >= self.sas_seq_len:
                c = a[-self.sas_seq_len:]

            one_sample = {
                "UserID": ann['UserID'],
                "InteractedItemIDs_pad": np.array(b),
                # "InteractedItemIDs": ann['InteractedItemIDs'],
                "InteractedItemTitles": convert_title_list_v2(ann['InteractedItemTitles'][-InteractedNum:]),
                "TargetItemID": ann["TargetItemID"],
                "TargetItemTitle": "\"" + ann["TargetItemTitle"].strip(' ') + "\"",
                "InteractedNum": InteractedNum,
                "label": ann['label'],
                "sas_seq": np.array(c)
            }
            if self.prompt_flag:
                one_sample['prompt_flag'] = ann['prompt_flag']
            return one_sample
        else:
            one_sample = {
                "UserID": ann['UserID'],
                # "InteractedItemIDs_pad": None,
                # # "InteractedItemIDs": ann['InteractedItemIDs'],
                # "InteractedItemTitles": None,
                "TargetItemID": ann["TargetItemID"],
                "TargetItemTitle": ann["TargetItemTitle"].strip(' '),
                # "InteractedNum": None,
                "label": ann['label']
            }
            if self.prompt_flag:
                one_sample['prompt_flag'] = ann['prompt_flag']
            return one_sample