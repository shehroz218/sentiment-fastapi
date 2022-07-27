### Usage and Replicating. 

1. Install dependencies:
  pip install -r requirements.txt
 
2. Both the trainer class and the prediction class have two capabilities either to save/load the model from local storage or to upload/use the transformers pipeline api to perform.

3. The pipeline is available in the transformers library and can is available  as "shozi218/finetuned-emotion-model"


4. The predicter class was used to create Fast-API class and can be locally deployed using uvicorn

5. The model API has been containerized and deployed on heroku. To deploy this model into your own heroku instance follow these steps: (PS: app name is the name of the app created on heroku)
    i. heroku login
    ii. heroku container:login
    iii. heroku create
    iv. docker build -t registry.heroku.com/[[app-name]]/web .
    v. docker push  registry.heroku.com/[[app-name]]/web
    vi. heroku container:release web -a [[app-name]]
   
### Model Selection and Fine Tuning
