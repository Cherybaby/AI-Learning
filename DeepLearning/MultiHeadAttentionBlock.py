class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model  # embedding特征大小
        self.h = h  # 头的个数
        # 确保d_model可以被h整除
        assert d_model % h == 0, "d_model 不能被 h整除"

        self.d_k = d_model // h  # 每个头特征大小
        self.w_q = nn.Linear(d_model, d_model, bias=False)  # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False)  # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False)  # Wv
        self.w_o = nn.Linear(d_model, d_model, bias=False)  # Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        # 获取d_k的值。
        d_k = query.shape[-1]
        # Q乘以K的转置，除以根号下d_k。
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # 给mask为0的位置填入一个很大的负值，这样在进行softmax，注意力就为0。
            attention_scores.masked_fill_(mask == 0, -1e9)
        # 进行softmax，归一化。得到注意力权重
        # (batch, h, seq_len, seq_len)
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # 注意力权重乘以V，得到更新后的embedding。
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        # 通过3个全连接层，获取Q、K、V矩阵
        query = self.w_q(q)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # 对多头进行拆分
        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # 计算注意力
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # 多个头合并
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # 乘以输出层
        return self.w_o(x)