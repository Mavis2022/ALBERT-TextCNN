# ALBERT-TextCNN
A classification model to classify products to the appropriate category 
## ALBERT:
3 main changes of ALBERT from BERT:
1. Factorized embedding parameterization (词嵌入向量参数的因式分解)：之前上亿条参数，现在只有几百万
2. Cross-Layer Parameter Sharing（跨层参数共享）
3. Inter-sentence coherence loss （段落连续性任务）
######
预计albert_zh会有比roberta_zh更好的性能表现，并且能更好处理较长的文本
######
为加快训练速度，使用LAMB做为优化器 Use LAMB as optimizer, to train with big batch size
使用了大的batch_size来训练(4096)。 LAMB优化器使得我们可以训练，特别大的批次batch_size，如高达6万。
