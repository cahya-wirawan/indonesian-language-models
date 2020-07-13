# How to train a language model with multiple GPUs

Following is just a short tutorial how to train a language model from scratch or fine tuning using 
Pytorch distributed training module. BERT language model and Transformers 
version 3.02 are used as example for this tutorial.

## Tokenizing the dataset
Before we start with the training, we have to tokenize the dataset using the huggingface tokenizer. This is only 
done once for a specific dataset. Once you have done it, you can use it for pre-training or fine-tuning later. 
Following is a short python code to do this task:
```
from tokenizers import BertWordPieceTokenizer

data = ["/dataset/train.txt"]
output_dir = "/output/bert/tokens"
vocab_size = 30522

tokenizer = BertWordPieceTokenizer()
tokenizer.train(data, vocab_size=vocab_size)

tokenizer.save_model(output_dir)
```
## Preparing LM configuration
We need also a configuration file to describe the language model  we want to train. Following is a json 
configuration file for BERT base model and save it in the directory <i>output_dir</i> set above ("/output/bert/tokens"):
``` 
{
  "architectures": [
    "BertForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "gradient_checkpointing": false,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 0,
  "type_vocab_size": 2,
  "vocab_size": 30522
}
```
For any other models configuration, you can check some examples from [huggingface's model collection](https://huggingface.co/models)  
For example, if you want to train "bert-base-uncased", you can download its config.json using wget command line:
```
wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-config.json -O $output_dir/config.json
```
and change its values appropriately.

## Distributed Training 
### Training Language Model from scratch
Download the language model training script [run_language_modeling.py](https://raw.githubusercontent.com/huggingface/transformers/v3.0.2/examples/language-modeling/run_language_modeling.py)
from Huggingface and then run following commands (update the values appropriately).     
```
model_source="/output/bert/tokens"
model_target="/output/bert/base"
target_train="/dataset/train.txt"
target_test="/dataset/test.txt"
#learning_rate="1.0e-4"

python -m torch.distributed.launch --nproc_per_node=4 ./run_language_modeling.py \
  --output_dir $model_target \
  --config_name $model_source \
  --tokenizer_name $model_source \
  --train_data_file $target_train \
  --eval_data_file $target_test \
  --save_total_limit 5 \
  --block_size 128 \
  --overwrite_output_dir \
  --fp16 \
  --num_train_epochs 2 \
  --do_train \
  --per_device_train_batch_size 128 \
  --per_device_eval_batch_size 4 \
  --mlm
```

### Fine Tuning the Language Model
We can fine tune an existing language model (or the LM created above) with a new dataset
```
model_source="/output/bert/base"
model_target="/output/bert/base-finetuning"
target_train="/dataset/train_new.txt"
target_test="/dataset/test_new.txt"
#learning_rate="1.0e-4"

python -m torch.distributed.launch --nproc_per_node=4 ./run_language_modeling.py \
  --output_dir $model_target \
  --config_name $model_source \
  --model_name_or_path $model_source \ 
  --tokenizer_name $model_source \
  --train_data_file $target_train \
  --eval_data_file $target_test \
  --save_total_limit 5 \
  --block_size 128 \
  --overwrite_output_dir \
  --fp16 \
  --num_train_epochs 2 \
  --do_train \
  --per_device_train_batch_size 128 \
  --per_device_eval_batch_size 4 \
  --mlm
```
