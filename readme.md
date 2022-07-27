### Usage and Replicating. 

1. Install dependencies:
  pip install -r requirements.txt
 
2. Both the trainer class and the prediction class have two capabilities either to save/load the model from local storage or to upload/use the transformers pipeline api to perform.

3. The pipeline is available in the transformers library and can be used by calling "shozi218/finetuned-emotion-model"


4. The predicter class was used to create Fast-API class and can be locally deployed using uvicorn

5. The model API has been containerized and deployed on heroku. To deploy this model into your own heroku instance follow these steps: (PS: app name is the name of the app created on heroku)
<br> - i. heroku login
<br> - ii. heroku container:login
<br> - iii. heroku create
<br> - iv. docker build -t registry.heroku.com/[[app-name]]/web .
<br> - v. docker push  registry.heroku.com/[[app-name]]/web
<br> - vi. heroku container:release web -a [[app-name]]
   
### Model Selection and Fine Tuning
