# DC-NeuralConversation
OpenNMT based Neural Conversation model which implements Topic and Semantic Distributional Constraints to improve quality of generated responses

## Installing dependencies
- Install the following pip packages - `six`,`tqdm`, `torchtext=0.2.1`, `future`
- Install pytorch 0.3.0; Choose the appropriate link from the website based on your cuda version (our model was trained on CUDA 9)
- Create an empty directory inside the main directory for caching - `syntax_topic_cache`

## How to train and evaluate models
### Preprocessing the data
- Download the train data (sources and targets) and save it in the `data/opensubtitles_data/` directory along with the validation and test files.
- Run the following preprocessing command for T&#124;S model - `python preprocess.py -train_src data/opensubtitles_data/s_train_dialogue_length2_6.txt -train_tgt data/opensubtitles_data/t_train_dialogue_length2_6.txt -valid_src data/opensubtitles_data/s_val_dialogue_length2_6.txt -valid_tgt data/opensubtitles_data/t_val_dialogue_length2_6.txt -save_data data/opensubtitles_2_6 -dynamic_dict -share_vocab`
- Run the following preprocessing command for S&#124;T model - `python preprocess.py -train_src data/opensubtitles_data/t_train_dialogue_length2_6.txt -train_tgt data/opensubtitles_data/s_train_dialogue_length2_6.txt -valid_src data/opensubtitles_data/t_val_dialogue_length2_6.txt -valid_tgt data/opensubtitles_data/s_val_dialogue_length2_6.txt -save_data data/opensubtitles_2_6_s_given_t -dynamic_dict -share_vocab`

### Training the model
- Run the training command for T&#124;S model - `python train.py -data data/opensubtitles_2_6 -save_model saved_adadelta_opensubtitles_models/opensubtitles_2_6_t_given_s -gpuid 2 -encoder_type brnn -param_init 0.08 -batch_size 256 -learning_rate 0.1 -optim adadelta -max_grad_norm 1 -word_vec_size 1000 -layers 4 -rnn_size 1000 -epochs 10
`
- Run the training command for S&#124;T model - `python train.py -data data/opensubtitles_2_6_s_given_t -save_model saved_adadelta_opensubtitles_s_given_t_models/opensubtitles_2_6_s_given_t -gpuid 1 -encoder_type brnn -param_init 0.08 -batch_size 256 -learning_rate 0.1 -optim adadelta -max_grad_norm 1 -word_vec_size 1000 -layers 4 -rnn_size 1000 -epochs 10`

### Evaluating on the sample of Cornell Movie Dialogue Dataset
- To run evaluate DC-10 run - `python translate.py -model saved_adadelta_opensubtitles_models/opensubtitles_2_6_t_given_s_<10th epoch model file> -src data/cornell_test/s_cornell_dev_test.txt -tgt data/cornell_test/t_cornell_dev_test.txt -output DC_10_decoding_cornell_dev_test_predictions.txt -attn_debug -beam_size 10 -n_best 10 -batch_size 1 -verbose`
- To evaluate DC-MMI10 run - `python translate.py -model saved_adadelta_opensubtitles_models/opensubtitles_2_6_t_given_s_<10th epoch model file> -mmi_model saved_adadelta_opensubtitles_s_given_t_models/opensubtitles_2_6_s_given_t_<10th epoch model file> -src data/cornell_test/s_cornell_dev_test.txt -tgt data/cornell_test/t_cornell_dev_test.txt -output DC_MMI10_decoding_cornell_dev_test_predictions.txt -attn_debug -beam_size 10 -n_best 10 -batch_size 1 -verbose`

## Changing the \alpha and \beta (Topic and Semantic constraint weights)
Go the file `/onmt/translate/Beam.py` and update adjust `self.alpha` at line 93(for \alpha) and `self.gamma` at line 99(for \beta)