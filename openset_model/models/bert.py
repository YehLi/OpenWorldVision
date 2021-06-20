import math
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from .registry import register_model

def _gelu_python(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initially created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        This is now written in C in torch.nn.functional
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
gelu = getattr(F, "gelu", _gelu_python)

ACT2FN = {
    "relu": F.relu,
    "gelu": gelu,
    "tanh": F.tanh,
}

class BertEmbeddings(nn.Module):
    def __init__(self, in_dim, hidden_size, dropout):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Linear(in_dim, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        embeddings = self.word_embeddings(x)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class BertSelfOutput(nn.Module):
    def __init__(self, hidden_size, dropout):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob):
        super(BertSelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads)
            )
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)

        shape_list = list(range(len(new_x_shape)))
        shape_list[-2], shape_list[-3] = shape_list[-3], shape_list[-2]
        return x.permute(shape_list)
        #return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)   
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        shape_list = list(range(len(context_layer.shape)))
        shape_list[-2], shape_list[-3] = shape_list[-3], shape_list[-2]
        context_layer = context_layer.permute(shape_list).contiguous()
        #context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        return context_layer

class BertAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, dropout, att_dropout):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(hidden_size, num_attention_heads, att_dropout)
        self.output = BertSelfOutput(hidden_size, dropout)

    def forward(self, input_tensor):
        self_output = self.self(input_tensor)
        attention_output = self.output(self_output, input_tensor)
        return attention_output

class BertIntermediate(nn.Module):
    def __init__(self, hidden_size, intermediate_size, dropout):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = ACT2FN["gelu"]
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

class BertOutput(nn.Module):
    def __init__(self, intermediate_size, hidden_size, dropout):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertLayer(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_attention_heads, dropout, att_dropout):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(hidden_size, num_attention_heads, dropout, att_dropout)
        self.intermediate = BertIntermediate(hidden_size, intermediate_size, dropout)
        self.output = BertOutput(intermediate_size, hidden_size, dropout)

    def forward(self, hidden_states):
        attention_output = self.attention(hidden_states)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class BertAttPooler(nn.Module):
    def __init__(self, hidden_size, pooler_dropout):
        super(BertAttPooler, self).__init__()
        sequential = [
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True)
        ]
        if pooler_dropout > 0:
            sequential.append(nn.Dropout(p=pooler_dropout))
        sequential.append(nn.Linear(hidden_size, 1))
        self.linear = nn.Sequential(*sequential)
        self.embed = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states):
        score = self.linear(hidden_states).squeeze(-1)
        score = F.softmax(score, dim=-1)
        output = score.unsqueeze(1).matmul(hidden_states).squeeze(1)
        output = self.embed(output)
        return output


class SingleBert(nn.Module):
    def __init__(self, layer_num, hidden_size, intermediate_size, num_attention_heads, dropout, att_dropout, num_classes):
        super(SingleBert, self).__init__()
        self.word_embed = BertEmbeddings(num_classes, hidden_size, dropout)
        layers = []
        for _ in range(layer_num):
            layer = BertLayer(hidden_size, intermediate_size, num_attention_heads, dropout, att_dropout)
            layers.append(layer)
        self.layers = nn.ModuleList(layers)
        self.pooler = BertAttPooler(hidden_size, dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        hidden_states = self.word_embed(x)
        for layer_module in self.layers:
            hidden_states = layer_module(hidden_states)
        hidden_state = self.pooler(hidden_states)
        logit = self.fc(hidden_state)
        return logit


class Bert(nn.Module):
    def __init__(self, layer_num, hidden_size, intermediate_size, num_attention_heads, dropout, att_dropout, num_classes, num_berts, **kwargs):
        super(Bert, self).__init__()
        self.num_berts = num_berts
        berts = []
        for i in range(self.num_berts):
            berts += [SingleBert(layer_num, hidden_size, intermediate_size, num_attention_heads, dropout, att_dropout, num_classes)]
        self.berts = nn.ModuleList(berts)

    def forward(self, x):
        logits_all = []
        for i in range(self.num_berts):
            logits_all.append(self.berts[i](x).unsqueeze(-1))
        logits_all = torch.cat(logits_all, dim=-1)
        logits = torch.mean(logits_all, dim=-1)

        return logits


@register_model
def bert(layer_num, hidden_size, intermediate_size, num_attention_heads, dropout, att_dropout, num_classes, num_berts, **kwargs):
    return Bert(layer_num, hidden_size, intermediate_size, num_attention_heads, dropout, att_dropout, num_classes, num_berts)
