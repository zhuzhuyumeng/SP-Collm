import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error  # 用于计算RMSE, MAE

# --- 1. 数据定义与预处理 (模拟) ---

# 假设的参数
VOCAB_SIZE_CONTRACTS = 1000  # 合约ID总数
MAX_SEQ_LEN = 50  # 最大序列长度
EMBED_DIM = 64  # 嵌入维度
HIDDEN_DIM = 64  # GRU隐藏层维度
POP_LEVELS = 100  # 流行度离散级别 (1-100)
TIME_INTERVAL_LEVELS = 10  # 时间间隔离散级别 (0-9)
T_SCALE = 3600  # 时间缩放因子 (秒转小时)
K_MAX_INTERVAL = 9  # 时间间隔离散化上限


class DataProcessor:
    def __init__(self, vocab_size_contracts, max_seq_len, pop_levels, time_interval_levels, t_scale, k_max_interval):
        self.vocab_size_contracts = vocab_size_contracts
        self.max_seq_len = max_seq_len
        self.pop_levels = pop_levels
        self.time_interval_levels = time_interval_levels
        self.t_scale = t_scale
        self.k_max_interval = k_max_interval

    def calculate_contract_popularity(self, all_user_interactions):
        """
        模拟计算合约流行度 P_i = sum(C_u for u in U_i)
        C_u = 1 / N_u
        并归一化到 (1, pop_levels) 之间
        """
        user_interaction_counts = {}  # {user_id: count}
        contract_raw_popularity = {}  # {contract_id: raw_pop_sum}

        for user_id, seq_S in all_user_interactions.items():
            user_interaction_counts[user_id] = len(seq_S)

        for user_id, seq_S in all_user_interactions.items():
            C_u = 1.0 / user_interaction_counts[user_id]
            for contract_id in seq_S:
                contract_raw_popularity[contract_id] = contract_raw_popularity.get(contract_id, 0) + C_u

        # 归一化和离散化
        if not contract_raw_popularity:
            return {i: 1 for i in range(self.vocab_size_contracts)}  # 默认值

        max_raw_pop = max(contract_raw_popularity.values())
        min_raw_pop = min(contract_raw_popularity.values())

        if max_raw_pop == min_raw_pop:  # 所有流行度都一样
            normalized_pop = {c_id: 1 for c_id in contract_raw_popularity}
        else:
            normalized_pop = {
                c_id: int(1 + (self.pop_levels - 1) * (raw_pop - min_raw_pop) / (max_raw_pop - min_raw_pop))
                for c_id, raw_pop in contract_raw_popularity.items()
            }

        # 确保所有可能的合约ID都有流行度，未交互的设为最低流行度
        for i in range(self.vocab_size_contracts):
            if i not in normalized_pop:
                normalized_pop[i] = 1  # 设为最低流行度

        return normalized_pop  # {contract_id: discrete_popularity_level}

    def calculate_time_interval_matrix(self, seq_T_u):
        """
        计算时间间隔矩阵 T_delta^u
        T_delta^u(i, j) = min(floor(log2(Delta t / T_scale)) + 3, k)
        Delta t 以秒为单位
        """
        seq_len = len(seq_T_u)
        time_delta_matrix = torch.zeros((self.max_seq_len, self.max_seq_len), dtype=torch.long)

        for i in range(seq_len):
            for j in range(seq_len):
                if i == j:
                    # 对于自身交互，时间间隔为0，映射到离散值0
                    time_delta_matrix[i, j] = 0
                    continue

                delta_t = abs(seq_T_u[i] - seq_T_u[j])  # 秒
                if delta_t == 0:
                    discretized_val = 0
                else:
                    discretized_val = min(math.floor(math.log2(delta_t / self.t_scale)) + 3, self.k_max_interval)
                    discretized_val = max(0, discretized_val)  # 确保不小于0
                time_delta_matrix[i, j] = discretized_val
        return time_delta_matrix

    def preprocess_sequence(self, S_u_raw, T_u_raw, contract_popularities):
        """
        处理单个用户的原始序列，进行填充/截断，并生成流行度序列和时间间隔矩阵
        S_u_raw: list of contract_ids
        T_u_raw: list of timestamps (seconds)
        contract_popularities: dict {contract_id: discrete_popularity_level}
        """
        seq_len_raw = len(S_u_raw)

        # 截断或填充 S_u
        if seq_len_raw > self.max_seq_len:
            S_u = S_u_raw[-self.max_seq_len:]
            T_u = T_u_raw[-self.max_seq_len:]
        else:
            S_u = [0] * (self.max_seq_len - seq_len_raw) + S_u_raw
            # 填充时间戳通常用0或序列中第一个时间戳，这里用0
            T_u_padded = [0] * (self.max_seq_len - seq_len_raw)
            T_u = T_u_padded + T_u_raw

        # 生成流行度序列 P_u
        P_u = [contract_popularities.get(c_id, 1) for c_id in S_u]  # 填充0的合约或未见过的合约给最低流行度

        # 计算时间间隔矩阵 T_delta^u
        T_delta_u = self.calculate_time_interval_matrix(T_u)

        return (
            torch.tensor(S_u, dtype=torch.long),
            torch.tensor(P_u, dtype=torch.long),
            T_delta_u  # T_delta_u 已经处理成 max_seq_len x max_seq_len
        )


# --- 2. 嵌入层 (Embedding Layer) ---

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size_contracts, embed_dim, pop_levels, time_interval_levels):
        super(EmbeddingLayer, self).__init__()
        self.contract_embedding = nn.Embedding(vocab_size_contracts, embed_dim, padding_idx=0)  # 0 for padding
        self.time_interval_embedding_K = nn.Embedding(time_interval_levels, embed_dim)  # K for Key
        self.time_interval_embedding_V = nn.Embedding(time_interval_levels, embed_dim)  # V for Value
        self.pop_embedding_Q = nn.Embedding(pop_levels + 1, embed_dim)  # Q for Query (+1 for level 0/1)
        self.pop_embedding_K = nn.Embedding(pop_levels + 1, embed_dim)  # K for Key
        self.pop_embedding_V = nn.Embedding(pop_levels + 1, embed_dim)  # V for Value

    def forward(self, seq_S_u, seq_P_u, T_delta_u):
        """
        Args:
            seq_S_u: (batch_size, max_seq_len) - 采购商交互序列 (合约ID)
            seq_P_u: (batch_size, max_seq_len) - 采购商交互序列对应的流行度
            T_delta_u: (batch_size, max_seq_len, max_seq_len) - 时间间隔矩阵
        Returns:
            E: (batch_size, max_seq_len, embed_dim) - 交互序列嵌入
            T_K_emb, T_V_emb: (batch_size, max_seq_len, max_seq_len, embed_dim) - 时间间隔矩阵的K,V嵌入
            P_Q_emb, P_K_emb, P_V_emb: (batch_size, max_seq_len, embed_dim) - 流行度序列的Q,K,V嵌入
        """
        E = self.contract_embedding(seq_S_u)  # (batch_size, max_seq_len, embed_dim)

        # 展开 T_delta_u 以进行嵌入查找
        batch_size, seq_len, _ = T_delta_u.shape
        T_delta_u_flat = T_delta_u.view(-1)  # (batch_size * seq_len * seq_len)
        T_K_flat = self.time_interval_embedding_K(T_delta_u_flat)
        T_V_flat = self.time_interval_embedding_V(T_delta_u_flat)
        T_K_emb = T_K_flat.view(batch_size, seq_len, seq_len, -1)  # (batch_size, seq_len, seq_len, embed_dim)
        T_V_emb = T_V_flat.view(batch_size, seq_len, seq_len, -1)  # (batch_size, seq_len, seq_len, embed_dim)

        P_Q_emb = self.pop_embedding_Q(seq_P_u)  # (batch_size, max_seq_len, embed_dim)
        P_K_emb = self.pop_embedding_K(seq_P_u)  # (batch_size, max_seq_len, embed_dim)
        P_V_emb = self.pop_embedding_V(seq_P_u)  # (batch_size, max_seq_len, embed_dim)

        return E, T_K_emb, T_V_emb, P_Q_emb, P_K_emb, P_V_emb


# --- 3. 合约流行度自注意网络 (LPSAN) ---
class LPSAN(nn.Module):
    def __init__(self, embed_dim, max_seq_len):
        super(LPSAN, self).__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        # 简化版 Q,K,V 线性变换
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)

    def forward(self, E, P_K_emb, P_V_emb, pad_mask=None):
        """
        Args:
            E: (batch_size, max_seq_len, embed_dim) - 交互序列嵌入
            P_K_emb: (batch_size, max_seq_len, embed_dim) - 流行度嵌入 P^K
            P_V_emb: (batch_size, max_seq_len, embed_dim) - 流行度嵌入 P^V
            pad_mask: (batch_size, max_seq_len) - 填充掩码 (True for padding positions)
        Returns:
            O_I: (batch_size, max_seq_len, embed_dim) - LPSAN输出
        """
        Q = self.W_q(E)  # (batch_size, max_seq_len, embed_dim)
        K = self.W_k(E)  # (batch_size, max_seq_len, embed_dim)
        V = self.W_v(E)  # (batch_size, max_seq_len, embed_dim)

        # Hadamard 内积融合流行度
        K_p = K * P_K_emb  # (batch_size, max_seq_len, embed_dim)
        V_p = V * P_V_emb  # (batch_size, max_seq_len, embed_dim)

        # 自注意力 (简化版)
        # Q * K_p.transpose(-2, -1) -> (batch_size, max_seq_len, max_seq_len)
        scores = torch.matmul(Q, K_p.transpose(-2, -1)) / math.sqrt(self.embed_dim)

        if pad_mask is not None:
            # 填充位置得分设为极小值
            scores = scores.masked_fill(pad_mask.unsqueeze(1), -1e9)  # (batch_size, 1, max_seq_len)
            # 确保未来的信息不会被看到 (自回归)
            causal_mask = torch.triu(torch.ones(self.max_seq_len, self.max_seq_len, dtype=torch.bool), diagonal=1).to(
                E.device)
            scores = scores.masked_fill(causal_mask, -1e9)

        attention_weights = F.softmax(scores, dim=-1)  # (batch_size, max_seq_len, max_seq_len)
        O_I = torch.matmul(attention_weights, V_p)  # (batch_size, max_seq_len, embed_dim)

        return O_I


# --- 4. 采购商偏好自注意网络 (BPSAN) ---
class BPSAN(nn.Module):
    def __init__(self, embed_dim, max_seq_len, num_heads=1):  # 简化为单头注意力
        super(BPSAN, self).__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads

        assert embed_dim % num_heads == 0
        self.head_dim = embed_dim // num_heads

        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)

        self.gru = nn.GRU(embed_dim, embed_dim, batch_first=True)

    def _generate_pop_positional_encoding(self, P_u):
        """
        流行度位置编码 (Pop-PE)
        P_u: (batch_size, max_seq_len) - 流行度序列 (作为位置信息)
        """
        batch_size, seq_len = P_u.shape
        # 将 P_u 转换为浮点数，并用作位置信息。
        # 这里需要注意P_u的范围，通常PE的div_term基于位置索引，但此处专利描述是基于流行度值
        # 为简化，直接将P_u作为位置信息，假设其在合理范围内
        position = P_u.float().unsqueeze(-1)  # (batch_size, seq_len, 1)

        div_term = torch.exp(torch.arange(0, self.embed_dim, 2).float() * (-math.log(10000.0) / self.embed_dim)).to(
            P_u.device)

        pe = torch.zeros(batch_size, seq_len, self.embed_dim).to(P_u.device)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe  # (batch_size, max_seq_len, embed_dim)

    def forward(self, E, P_u, T_K_emb, T_V_emb, pad_mask=None):
        """
        Args:
            E: (batch_size, max_seq_len, embed_dim) - 交互序列嵌入
            P_u: (batch_size, max_seq_len) - 采购商交互序列对应的流行度
            T_K_emb: (batch_size, max_seq_len, max_seq_len, embed_dim) - 时间间隔K嵌入
            T_V_emb: (batch_size, max_seq_len, max_seq_len, embed_dim) - 时间间隔V嵌入
            pad_mask: (batch_size, max_seq_len) - 填充掩码 (True for padding positions)
        Returns:
            O_U: (batch_size, max_seq_len, embed_dim) - BPSAN输出
        """
        # 流行度位置编码
        pop_pe = self._generate_pop_positional_encoding(P_u)
        E_prime = E + pop_pe  # E' = E + Pop-PE (batch_size, max_seq_len, embed_dim)

        Q = self.W_q(E_prime)  # (batch_size, max_seq_len, embed_dim)
        K = self.W_k(E_prime)  # (batch_size, max_seq_len, embed_dim)
        V = self.W_v(E_prime)  # (batch_size, max_seq_len, embed_dim)

        # 简化版多头注意力，这里只实现单头，但保留多头维度计算逻辑
        Q = Q.view(Q.shape[0], Q.shape[1], self.num_heads, self.head_dim).transpose(1,
                                                                                    2)  # (batch_size, num_heads, seq_len, head_dim)
        K = K.view(K.shape[0], K.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(V.shape[0], V.shape[1], self.num_heads, self.head_dim).transpose(1, 2)

        # 将T_K_emb, T_V_emb 调整到 (batch_size, num_heads, seq_len, seq_len, head_dim)
        # 这里需要针对每个head进行 repeat，然后 view
        batch_size, seq_len, _, embed_dim = T_K_emb.shape
        T_K_emb_per_head = T_K_emb.view(batch_size, seq_len, seq_len, self.num_heads, self.head_dim).permute(0, 3, 1, 2,
                                                                                                             4)  # (batch_size, num_heads, seq_len, seq_len, head_dim)
        T_V_emb_per_head = T_V_emb.view(batch_size, seq_len, seq_len, self.num_heads, self.head_dim).permute(0, 3, 1, 2,
                                                                                                             4)  # (batch_size, num_heads, seq_len, seq_len, head_dim)

        # (batch_size * num_heads, seq_len, head_dim)
        Q_flat = Q.reshape(-1, self.max_seq_len, self.head_dim)
        K_flat = K.reshape(-1, self.max_seq_len, self.head_dim)
        V_flat = V.reshape(-1, self.max_seq_len, self.head_dim)

        # (batch_size * num_heads, seq_len, seq_len, head_dim)
        T_K_flat = T_K_emb_per_head.reshape(-1, self.max_seq_len, self.max_seq_len, self.head_dim)
        T_V_flat = T_V_emb_per_head.reshape(-1, self.max_seq_len, self.max_seq_len, self.head_dim)

        # 计算注意力得分 (Q * K_T) + 时间间隔融合
        # Q_flat (B*H, S, HD) @ K_flat.transpose (B*H, HD, S) -> (B*H, S, S)
        scores = torch.matmul(Q_flat, K_flat.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 融入时间间隔 (更符合原始专利描述的融合方式)
        # Q_flat (B*H, S, HD) @ T_K_flat.transpose (B*H, HD, S, S)
        # 这里需要更精细的维度处理，简化为将时间K嵌入与Q进行点积后加到分数上
        # (B*H, S, HD) @ (B*H, HD, S) -> (B*H, S, S)
        # 这里的 T_K_flat 是 (B*H, S, S, HD)
        # 我们需要 q_i * k_j + q_i * T_K(i,j)
        # 简化为：Q_flat (B*H, S, HD) 和 K_flat_with_time (B*H, HD, S)
        # (Q_flat @ K_flat.transpose(-2,-1)) + (Q_flat @ T_K_flat.transpose(-2,-1)) # 这种需要T_K_flat的S轴匹配
        # 更合理的做法是在scores上直接加一个 T_delta_u 相关的分数
        # 例如，可以 Q_i @ (K_j + T_K_emb_ij)

        # 简化处理：将时间间隔的K和V作为额外的加性偏置融入
        # (B*H, S, S) + (B*H, S, HD) @ (B*H, HD, S)
        # scores += (Q_flat.unsqueeze(2) * T_K_flat).sum(dim=-1) # Q_i * T_K(i,j)

        # 再次简化，只是为了模拟融入，在点积后加入一个时间相关的偏置
        # (batch_size*num_heads, seq_len, seq_len)
        # T_K_flat的维度是 (B*H, S, S, HD)，这里直接用它的均值作为偏置，非常简化
        # scores += T_K_flat.mean(dim=-1)

        # 掩码应用 (与LPSAN类似)
        if pad_mask is not None:
            mask_expanded = pad_mask.unsqueeze(1).unsqueeze(1).repeat(1, self.num_heads, self.max_seq_len, 1).view(-1,
                                                                                                                   self.max_seq_len,
                                                                                                                   self.max_seq_len)
            scores = scores.masked_fill(mask_expanded, -1e9)
            causal_mask = torch.triu(torch.ones(self.max_seq_len, self.max_seq_len, dtype=torch.bool), diagonal=1).to(
                E.device)
            scores = scores.masked_fill(causal_mask, -1e9)

        attention_weights = F.softmax(scores, dim=-1)  # (batch_size*num_heads, seq_len, seq_len)
        attended_values = torch.matmul(attention_weights, V_flat)  # (batch_size*num_heads, seq_len, head_dim)

        # 重组多头输出
        attended_values = attended_values.view(-1, self.num_heads, self.max_seq_len, self.head_dim).transpose(1,
                                                                                                              2).reshape(
            -1, self.max_seq_len, self.embed_dim)

        O_attention = self.W_o(attended_values)  # (batch_size, max_seq_len, embed_dim)

        # GRU网络
        O_U, _ = self.gru(O_attention)  # (batch_size, max_seq_len, embed_dim)

        return O_U


# --- 5. 融合层 (Fusion Layer) ---
class FusionLayer(nn.Module):
    def __init__(self, embed_dim):
        super(FusionLayer, self).__init__()
        self.ffn_oi = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.ffn_ou = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        # 融合后的全连接层
        self.W_f = nn.Linear(embed_dim * 2, embed_dim)  # Concat后维度翻倍
        self.b_f = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, O_I, O_U):
        """
        Args:
            O_I: (batch_size, max_seq_len, embed_dim) - LPSAN输出
            O_U: (batch_size, max_seq_len, embed_dim) - BPSAN输出
        Returns:
            O_final: (batch_size, max_seq_len, embed_dim) - 最终融合向量
        """
        ffn_oi_out = self.ffn_oi(O_I)
        ffn_ou_out = self.ffn_ou(O_U)

        # 横向拼接
        concatenated = torch.cat((ffn_oi_out, ffn_ou_out), dim=-1)  # (batch_size, max_seq_len, embed_dim*2)

        # O_final = F.relu(torch.matmul(concatenated, self.W_f.weight.T) + self.b_f) # (batch_size, max_seq_len, embed_dim)
        # Simplified for clarity, direct linear transformation after concat
        O_final = self.W_f(concatenated) + self.b_f
        O_final = F.relu(O_final)

        return O_final


# --- 6. B2B-PPSASRec 整体模型 ---
class B2B_PPSASRec(nn.Module):
    def __init__(self, vocab_size_contracts, embed_dim, max_seq_len,
                 pop_levels, time_interval_levels):
        super(B2B_PPSASRec, self).__init__()

        self.embedding_layer = EmbeddingLayer(vocab_size_contracts, embed_dim, pop_levels, time_interval_levels)
        self.lpsan = LPSAN(embed_dim, max_seq_len)
        self.bpsan = BPSAN(embed_dim, max_seq_len)
        self.fusion_layer = FusionLayer(embed_dim)

        # 预测层：线性输出层用于预测采购量
        # Prediction_head now takes the dot product as input (1 value) and outputs 1 value
        self.prediction_head_linear = nn.Linear(1, 1)

        # 用于计算预测时与候选合约的M_j点积，假设M_j就是合约嵌入本身
        self.contract_embedding_lookup = self.embedding_layer.contract_embedding  # 共享嵌入层

    def forward(self, seq_S_u, seq_P_u, T_delta_u, pad_mask=None, target_contract_id=None):
        """
        Args:
            seq_S_u: (batch_size, max_seq_len) - 采购商交互序列
            seq_P_u: (batch_size, max_seq_len) - 流行度序列
            T_delta_u: (batch_size, max_seq_len, max_seq_len) - 时间间隔矩阵
            pad_mask: (batch_size, max_seq_len) - 填充掩码 (True for padding positions)
            target_contract_id: (batch_size,) - 目标合约ID (用于预测该合约的采购量)
        Returns:
            predicted_purchase_volume: (batch_size,) - 预测的采购量
        """
        # 嵌入层
        E, T_K_emb, T_V_emb, P_Q_emb, P_K_emb, P_V_emb = self.embedding_layer(seq_S_u, seq_P_u, T_delta_u)

        # LPSAN
        O_I = self.lpsan(E, P_K_emb, P_V_emb, pad_mask)

        # BPSAN
        O_U = self.bpsan(E, seq_P_u, T_K_emb, T_V_emb, pad_mask)

        # 融合层
        O_final = self.fusion_layer(O_I, O_U)  # (batch_size, max_seq_len, embed_dim)

        # 找到最后一个非填充位置 (或取序列的最后一个)
        if pad_mask is not None:
            actual_lens = (~pad_mask).sum(dim=1)  # (batch_size,)
            last_output_indices = actual_lens - 1
            last_output_indices = torch.clamp(last_output_indices, min=0)

            # 使用gather获取每个batch中最后一个有效时间步的输出
            final_user_representation = O_final[torch.arange(O_final.shape[0]), last_output_indices]
        else:
            final_user_representation = O_final[:, -1, :]  # (batch_size, embed_dim)

        # 预测特定合约的采购量
        if target_contract_id is not None:
            # 获取目标合约的嵌入 M_j
            M_j = self.contract_embedding_lookup(target_contract_id)  # (batch_size, embed_dim)

            # 点积 (final_user_representation . M_j)
            dot_product_score = (final_user_representation * M_j).sum(dim=-1)  # (batch_size,)

            # 经过一个线性层预测采购量
            predicted_purchase_volume = self.prediction_head_linear(dot_product_score.unsqueeze(-1))  # (batch_size, 1)
        else:
            # 如果没有指定target_contract_id，这可能不是一个回归任务的典型路径
            # 但为了完整性，这里可以返回一个默认值或引发错误
            raise ValueError("target_contract_id must be provided for purchase volume prediction.")

        return predicted_purchase_volume.squeeze(-1)  # (batch_size,)


# --- 评价指标函数 (RMSE, MAE) ---

def calculate_regression_metrics(predictions, actuals):
    """
    计算 RMSE 和 MAE

    Args:
        predictions (torch.Tensor): 模型预测值 (batch_size,)
        actuals (torch.Tensor): 真实值 (batch_size,)

    Returns:
        dict: 包含 RMSE 和 MAE 的字典
    """
    # 将 Tensor 转换为 NumPy 数组
    predictions_np = predictions.detach().cpu().numpy()
    actuals_np = actuals.detach().cpu().numpy()

    rmse = np.sqrt(mean_squared_error(actuals_np, predictions_np))
    mae = mean_absolute_error(actuals_np, predictions_np)

    return {'RMSE': rmse, 'MAE': mae}


# --- 测试样例 ---

# 1. 模拟数据生成
dp = DataProcessor(VOCAB_SIZE_CONTRACTS, MAX_SEQ_LEN, POP_LEVELS, TIME_INTERVAL_LEVELS, T_SCALE, K_MAX_INTERVAL)

# 模拟所有用户的交互数据，用于计算全局流行度
all_user_interactions_for_popularity = {
    0: [10, 20, 30, 10, 50, 60, 10],
    1: [15, 25, 35, 15, 15],
    2: [70, 71, 72, 73, 70, 71],
    3: [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
}
contract_popularities = dp.calculate_contract_popularity(all_user_interactions_for_popularity)

# 模拟一批用户的输入数据 (batch_size = 2)
batch_size = 2

# 用户1： 交互序列 (合约ID) 和时间戳 (秒)
S_u_raw_1 = [10, 20, 30, 40]
T_u_raw_1 = [1678886400, 1678886400 + 3600, 1678886400 + 7200, 1678886400 + 86400 * 5]  # 间隔有小时，也有天

# 用户2： 交互序列 (合约ID) 和时间戳 (秒)
S_u_raw_2 = [70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
             90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
             110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120]  # 超过MAX_SEQ_LEN
T_u_raw_2 = [1678886400 + i * 1200 for i in range(len(S_u_raw_2))]  # 每20分钟一次交互

# 预处理数据
seq_S_u_1, seq_P_u_1, T_delta_u_1 = dp.preprocess_sequence(S_u_raw_1, T_u_raw_1, contract_popularities)
seq_S_u_2, seq_P_u_2, T_delta_u_2 = dp.preprocess_sequence(S_u_raw_2, T_u_raw_2, contract_popularities)

# 合并成Batch
seq_S_u_batch = torch.stack([seq_S_u_1, seq_S_u_2])
seq_P_u_batch = torch.stack([seq_P_u_1, seq_P_u_2])
T_delta_u_batch = torch.stack([T_delta_u_1, T_delta_u_2])

# 创建填充掩码 (0是填充，这里设置为True表示是填充位置)
pad_mask = (seq_S_u_batch == 0)  # (batch_size, max_seq_len)

# 目标合约ID和实际采购量 (模拟)
target_contract_id_batch = torch.tensor([50, 120], dtype=torch.long)  # 预测用户1对合约50，用户2对合约120的采购量
actual_purchase_volume_batch = torch.tensor([150.0, 300.0], dtype=torch.float)

print(f"seq_S_u_batch shape: {seq_S_u_batch.shape}")
print(f"seq_P_u_batch shape: {seq_P_u_batch.shape}")
print(f"T_delta_u_batch shape: {T_delta_u_batch.shape}")
print(f"pad_mask shape: {pad_mask.shape}")

# 2. 初始化模型
model = B2B_PPSASRec(
    vocab_size_contracts=VOCAB_SIZE_CONTRACTS,
    embed_dim=EMBED_DIM,
    max_seq_len=MAX_SEQ_LEN,
    pop_levels=POP_LEVELS,
    time_interval_levels=TIME_INTERVAL_LEVELS
)

# 3. 模拟训练过程 (简化版)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()  # 均方误差损失

num_epochs = 500
print("\n--- Starting simulated training (Regression Task) ---")
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    predicted_volume = model(seq_S_u_batch, seq_P_u_batch, T_delta_u_batch, pad_mask, target_contract_id_batch)

    loss = criterion(predicted_volume, actual_purchase_volume_batch)

    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")

# 4. 模拟评估
model.eval()
with torch.no_grad():
    # 假设我们有一批测试数据
    test_S_u_batch = seq_S_u_batch  # (用训练数据模拟)
    test_P_u_batch = seq_P_u_batch
    test_T_delta_u_batch = T_delta_u_batch
    test_pad_mask = pad_mask
    test_target_contract_id_batch = target_contract_id_batch  # (用训练标签模拟)
    test_actual_purchase_volume_batch = actual_purchase_volume_batch  # (用训练标签模拟)

    # 获取模型预测
    test_predicted_volume = model(test_S_u_batch, test_P_u_batch, test_T_delta_u_batch, test_pad_mask,
                                  test_target_contract_id_batch)

    # 计算评价指标
    metrics = calculate_regression_metrics(test_predicted_volume, test_actual_purchase_volume_batch)

    print("\n--- Simulated Evaluation (Regression Metrics) ---")
    print(f"  Predicted Volumes: {test_predicted_volume.tolist()}")
    print(f"  Actual Volumes:    {test_actual_purchase_volume_batch.tolist()}")
    print(f"  RMSE: {metrics['RMSE']:.4f}")
    print(f"  MAE: {metrics['MAE']:.4f}")

# 5. 模拟一个应用场景：预测某个用户对“热门”合约的采购量
print("\n--- Simulated Application: Predict volumes for 'Hot' contracts ---")
with torch.no_grad():
    # 假设用户1 (test_S_u_batch[0])
    user_idx = 0
    user_S_u = seq_S_u_batch[user_idx].unsqueeze(0)
    user_P_u = seq_P_u_batch[user_idx].unsqueeze(0)
    user_T_delta_u = T_delta_u_batch[user_idx].unsqueeze(0)
    user_pad_mask = pad_mask[user_idx].unsqueeze(0)

    # 假设我们关注最热门的几个合约 (例如，流行度最高的合约)
    # 找到流行度最高的5个合约ID
    sorted_popularities = sorted(contract_popularities.items(), key=lambda item: item[1], reverse=True)
    hot_contract_ids = [c_id for c_id, _ in sorted_popularities[:5]]

    print(f"Top 5 hottest contract IDs (by popularity score): {hot_contract_ids}")

    predicted_volumes_for_hot_contracts = []
    for c_id in hot_contract_ids:
        target_c_id_tensor = torch.tensor([c_id], dtype=torch.long)
        predicted_volume = model(user_S_u, user_P_u, user_T_delta_u, user_pad_mask, target_c_id_tensor)
        predicted_volumes_for_hot_contracts.append((c_id, predicted_volume.item()))

    print(f"\nPredicted purchase volumes for user {user_idx} on hot contracts:")
    for c_id, volume in predicted_volumes_for_hot_contracts:
        print(f"  Contract ID: {c_id}, Predicted Volume: {volume:.2f}")