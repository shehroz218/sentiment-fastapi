import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


class predicter():
    def __init__(self,
                model_ckpt = "distilbert-base-uncased",
                num_labels=2,
                data_labels=['negative','positive'],
                load_path='finetuned-emotion-model\pytorch_model.bin'
                ):
        
        self.model_ckpt=model_ckpt
        self.num_labels=num_labels
        self.tokenizer=AutoTokenizer.from_pretrained(model_ckpt)
        self.data_labels=data_labels
        self.load_path=load_path
        self.model=self.load_model()
    
    def load_model(self):
        """
        loads model from either local repository or from hugging face pipeline
        """ 
        #script for when model is loaded from storage 
        # model = (AutoModelForSequenceClassification
        # .from_pretrained(self.model_ckpt, num_labels=2))
        # checkpoint=torch.load(self.load_path,
        # map_location=torch.device('cpu'))
        # self.new_model=model.load_state_dict(checkpoint)

        model_id = "shozi218/finetuned-emotion-model"
        model = pipeline("text-classification", model=model_id)   
        return model

    def predict(self, text:str)->int:
        """
        creates prediction from either transformer pipeline or locally loaded model

        Attributes:
        ----------
        text: str
        input text/tweet for sentiment classification
        """
        #script for when model is loaded from storage 
        # encode_text=self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        # output=self.model(input_ids=encode_text['input_ids'], attention_mask=encode_text['attention_mask'])
        # probabilities= torch.nn.functional.softmax(output.logits, dim=1).flatten().detach().numpy().tolist()
        # probabilities=dict(zip(self.data_labels, probabilities))
        # final_emotion=self.data_labels[torch.argmax(output[0])]

        preds = self.model(text, return_all_scores=True)
        prediction_probs=[preds[i]['score'] for i in range(len(preds))]
        final_emotion=self.data_labels[np.argmax(prediction_probs)]
        probabilities=dict(zip(self.data_labels, prediction_probs))
        return final_emotion,probabilities  

predicter = predicter()
def load_class():
    return predicter
