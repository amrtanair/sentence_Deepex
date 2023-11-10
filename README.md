# sentence_deepex

Finetuning the CoLa dataset on two models, BERT and GPT-2. 
The metrics used to evaluate, in addition to accuracy is the MCC score. 

### Date: 10/11/2023

Updated and cleaned the code. Changed file names. 
For the BERT model, the initial MCC score, while using cross entropy loss was 0.55. The score is decent, however, the validation loss was increasing while training, which seemed odd. 
The F1 score for the OIE-2016 dataset using this finetuned model was 45.7. 
Since the validation loss kept increasing, despite tuning the hyperparameters extensively, a different loss function(focal loss) was introduced. The MCC score was 0.57 and the validation loss
reduced through the epochs as expected. However, the training loss was much much smaller compared to the validation loss, which points to overfitting. 
The F1 score for the OIE-2016 dataset using this finetuned model was 44.7

For the GPT-2 model, I have written the boilerplate code. MCC score is bad as expected. 
