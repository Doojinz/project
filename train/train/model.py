from transformers import BertModel, BertTokenizer, BertPreTrainedModel
import opencc
import torch
from torch import nn
from torch.nn import CrossEntropyLoss



def _is_chinese_char(cp): # 检查给定的 Unicode 码点 (cp) 是否属于中文汉字的范围
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
                (cp >= 0x3400 and cp <= 0x4DBF) or  #
                (cp >= 0x20000 and cp <= 0x2A6DF) or  #
                (cp >= 0x2A700 and cp <= 0x2B73F) or  #
                (cp >= 0x2B740 and cp <= 0x2B81F) or  #
                (cp >= 0x2B820 and cp <= 0x2CEAF) or
                (cp >= 0xF900 and cp <= 0xFAFF) or  #
                (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
        return True
    return False

class SpellBert(BertPreTrainedModel):
    def __init__(self, config):
        super(SpellBert, self).__init__(config)

        self.vocab_size = config.vocab_size
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.vocab_size)

        self.init_weights()

    def tie_cls_weight(self): # 将分类器的权重与 BERT 的词嵌入权重绑定
        self.classifier.weight = self.bert.embeddings.word_embeddings.weight

    @staticmethod
    def build_batch(batch, tokenizer):
        return batch

    def forward(self, batch):
        input_ids = batch['src_idx']  # 输入的token ID
        attention_mask = batch['masks']  # 注意力掩码
        loss_mask = batch['loss_masks']  # 损失掩码，用于指示哪些位置需要计算损失
        label_ids = batch['tgt_idx'] if 'tgt_idx' in batch else None  # 标签ID

        outputs = self.bert(input_ids, attention_mask=attention_mask)

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)

        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # 添加隐藏状态和注意力（如果存在）
        
        if label_ids is not None:  # 如果 label_ids 存在，计算交叉熵损失，并只保留活动部分的损失
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            active_loss = loss_mask.view(-1) == 1
            active_logits = logits.view(-1, self.vocab_size)[active_loss]
            active_labels = label_ids.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)
            outputs = (loss,) + outputs

        return outputs 
    
