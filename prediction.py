import torch
from transformers import AutoTokenizer ,AutoModelForSequenceClassification

class trainer():
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
        model = (AutoModelForSequenceClassification
        .from_pretrained(self.model_ckpt, num_labels=2))
        checkpoint=torch.load(self.load_path,
        map_location=torch.device('cpu'))
        self.new_model=model.load_state_dict(checkpoint)   
        return model

    def predict(self, text:str)->int:



        encode_text=self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        label_idx=torch.argmax(self.model(input_ids=encode_text['input_ids'], attention_mask=encode_text['attention_mask'])[0])
        final_emotion=self.data_labels[label_idx]

        return final_emotion


