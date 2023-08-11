# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # 1) Settings

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# In this part:
# - we import the packages that we need
# - we define our parameters in particular the pre-trained model we want to use, the prediction type, the labels...
# - we check the device available and set the default device accordingly (cpu or cuda)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## a) Packages

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# general packages
import dataiku
from dataikuapi.dss.ml import DSSPredictionMLTaskSettings

import torch
import itertools
import os
from datetime import datetime
from sklearn.metrics import roc_auc_score, accuracy_score

# transformers packages
import datasets
import transformers
import evaluate

#import mlflow
import mlflow
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## b) Parameters

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# This part should be modified to change parameters and adjust to your use case.

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
MODEL_REVISION = "1c4513b2eedbda136f57676a34eea67aba266e5c"
HF_MODEL="distilbert-base-uncased"
#PREDICTION_TYPE = "MULTICLASS"



# HUGGING FACE PARAMETERS
#HF_MODEL = "bert-base-uncased" # model name
#REVISION = "0a6aa9128b6194f4f3c4db429b6cb4891cdb421b" # model revision (can be found in the commit)
INF_BATCH_SIZE = 16

# PREDICTION PARAMETERS
PREDICTION_TYPE = "MULTICLASS" # Could be BINARY or REGRESSION
train_data = dataiku.Dataset("train").get_dataframe()
CLASSES = sorted(list(train_data['label'].unique())) # labels should be sorted
NUM_LABELS = len(CLASSES)
LABEL2ID = {CLASSES[i]: i for i in range(NUM_LABELS)}
ID2LABEL = {el:key for key,el in LABEL2ID.items()}
TARGET = 'label'
#train_data["labels"] = train_data["label"].apply(lambda s: LABEL2ID[s])
#print("Training dataset was loaded with classes: {} and target: {}".format(CLASSES,TARGET))

# ML-flow parameteters
EXPERIMENT_FOLDER_ID = "4N8J9RH2"
EXPERIMENT_NAME = "Experiment_a"
MLFLOW_CODE_ENV_NAME = "py_37_hx_nlp"
SAVED_MODEL_NAME = "classification_model"
ARTIFACTS = {SAVED_MODEL_NAME: "hf_model.pth"}
TRAIN_DATASET = "train"
EVAL_DATASET = "test"
TARGET_NAME = 'label' # in eval dataset
DEPLOYMENT_METRIC = 'eval_accuracy' # could be eval_roc_auc any other metric computed and logged in X tracking
AUTO_DEPLOY=True

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## c) Setting up ML_Flow

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# This part does not need to be modified.

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Create a mlflow_extension object
client = dataiku.api_client()
project = client.get_default_project()
mlflow_extension = project.get_mlflow_extension()

# Get a handle on a Managed Folder to store the experiments.
managed_folder = project.get_managed_folder(EXPERIMENT_FOLDER_ID)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## d) (Opt) Garbage collect experiments - this will delete experiments that were removed

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# This part does not need to be modified.

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
mlflow_extension.garbage_collect()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## e) Check Device

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# This part does not need to be modified.

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
if torch.cuda.is_available():
    print("Running on GPU")
    DEVICE = 'cuda'
else:
    print('Running on CPU')
    DEVICE = 'cpu'

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # 2) Load Data

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# In this part:
# - we load the training and validation data and convert it to the [Hugging Face dataset](https://huggingface.co/docs/datasets/tabular_load#pandas-dataframes) format
# - we tokenize the dataset

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## a) Load data

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# This part can be modified if you want to change the way training and validation sets are defined.

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Divide between train and test
train_df = train_data.sample(frac = 0.8,random_state=42)
test_df = train_data.drop(train_df.index)

# reset index
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# Convert to Hugging Face Dataset format - convert to torch for GPU
hf_train_dataset = datasets.Dataset.from_pandas(train_df).class_encode_column(TARGET).with_format("torch")
hf_test_dataset = datasets.Dataset.from_pandas(test_df).class_encode_column(TARGET).with_format("torch")
print("Training and validation datasets loaded in Hugging Face Dataset format resp {} and {} rows".format(len(hf_train_dataset),len(hf_test_dataset)))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## b) Tokenize

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# This part can be modified if you want to change the way the text is tokenized.

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# define tokenizer + tokenizing function
#tokenizer = transformers.AutoTokenizer.from_pretrained(HF_MODEL)
tokenizer = transformers.DistilBertTokenizer.from_pretrained(HF_MODEL)

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True,padding="max_length")

# apply tokenizing function on train and test set
tokenized_train = hf_train_dataset.map(preprocess_function, batched=True)
tokenized_test = hf_test_dataset.map(preprocess_function, batched=True)
print("Train and test sets were tokenized")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # 3) Train Model + log in Experiment Tracking

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# In this part:
# - we define a hyperparameters grid
# - we add a metric of interest that we want to log at training time (could be anything but here we chose ROC-AUC)
# - we start the training with a layer of experiment tracking and in the end we log the model as a ML-Flow model.
# 
# On the last point as there is no ML-flow flavor for Hugging Face we use [this approach](https://julsimon.medium.com/using-mlflow-with-hugging-face-transformers-4f69093a6c04) with callbacks to make sure the logs are properly sent to the Experiment Tracking Section. To log the model, we define a pyfunc variant that relies on a [Hugging Face pipeline](https://huggingface.co/docs/transformers/pipeline_tutorial) and that was inspired by [this tutorial](https://developer.dataiku.com/latest/tutorials/machine-learning/experiment-tracking/keras-nlp/index.html). This class will embed the preprocessing, prediction and post-processing.
# 
# During this section, all the experiments are logged in the Experiment Tracking section associated with the Experiments folder.

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## a) Define hyperparameter grid- change grid

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# This part should be modified to define the hyperparameters you wish you test in your experiment.

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
param_grid = {
    'batch_size':[8,16],
    'learning_rate':[2e-5,4e-5],
    'num_train_epochs':[5,20,50],
    'training_dir':"_"
}

all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]

print("{} combinations will be tried. \n This corresponds to this grid: {}".format(len(all_params),param_grid))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## b) Define training metric

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# This part can be modified if you want to change the metrics that will be logged. You could use any training metric you like to chose the model you want to deploy, here we go for ROC AUC and accuracy.

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def compute_metrics(model,tokenized_test_dataset):
    # retrieve model prediction
    pred=model.predict(tokenized_test_dataset)
    # true labels
    true_labels=pred.label_ids
    # predictions
    predictions=pred.predictions
    predictions=torch.from_numpy(predictions)

    # retrieve scores and predicted labels from predictions
    pred_scores=torch.nn.functional.softmax(input=predictions, dim=-1)
    pred_labels=torch.argmax(predictions, dim=1)

    # compute scores
    #auc=roc_auc_score(y_true=true_labels, y_score=pred_scores,multi_class='ovr', average='macro')
    accuracy=accuracy_score(y_true=true_labels,y_pred=pred_labels)
    return {"accuracy": accuracy} #"auc": auc,

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## c) Training

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# This part can be modified if you want to change the training arguments or the metrics that are logged.

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
with project.setup_mlflow(managed_folder) as mlflow:

    # (1) SET-UP THE EXPERIMENT
    '''If the experiment did not already exist, we create it'''
    if len(mlflow_extension.list_experiments()) >= 1:
        if EXPERIMENT_NAME in [exp['name'] for exp in mlflow_extension.list_experiments()['experiments']]:
            print("Experiment already existed with name : {}".format(EXPERIMENT_NAME))
        else:
            experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
            print("Experiment created with name : {}".format(EXPERIMENT_NAME))
    else:
        experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
        print("Experiment created with name : {}".format(EXPERIMENT_NAME))

    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    experiment_id = experiment.experiment_id
    print("Experiment name is: {} and ID is: {} ".format(EXPERIMENT_NAME,experiment_id))

    mlflow.tracking.MlflowClient().set_experiment_tag(experiment_id, "library", "hugging_face")
    print('Tags added to the experiment')


    # (2) START LOOPING
    for i,params in enumerate(all_params):
        # (a) Start a run
        with mlflow.start_run(experiment_id=experiment_id) as run:
            run_id = run.info.run_id
            print(f'Starting run {run_id} ...\n{params}')
            print("Iteration {} out of {}".format(i+1,len(all_params)))

            #HF callbacks
            # retrieve newly created experiment
            os.environ["MLFLOW_EXPERIMENT_NAME"] = EXPERIMENT_NAME
            # flatten the parameters dictionary before logging
            os.environ["MLFLOW_FLATTEN_PARAMS"] = "1"
            # we do not need to log the checkpoint as we will save the best model right after
            os.environ["HF_MLFLOW_LOG_ARTIFACTS"] = "0"

            # instantiate the model
            model = transformers.AutoModelForSequenceClassification.from_pretrained(HF_MODEL,num_labels=NUM_LABELS,id2label=ID2LABEL,label2id=LABEL2ID,revision=MODEL_REVISION).to(DEVICE)
            print('Model instantiated on device {}'.format(model.device))

            # define training arguments - can be changed
            training_args = transformers.TrainingArguments(output_dir=params['training_dir'],
                                          learning_rate=params['learning_rate'],
                                          logging_strategy='epoch',
                                          per_device_train_batch_size=params['batch_size'],
                                          per_device_eval_batch_size=params['batch_size'],
                                          num_train_epochs=params['num_train_epochs'],
                                          weight_decay=0.01,
                                          evaluation_strategy="epoch",
                                          save_strategy="epoch",
                                          load_best_model_at_end=True,
                                          push_to_hub=False,metric_for_best_model='eval_loss')

            # implement early stopping
            early_stopper = transformers.EarlyStoppingCallback(early_stopping_patience=3,
                                                             early_stopping_threshold=0.05)

            # define trainer
            trainer = transformers.Trainer(model=model,
                                           args=training_args,
                                           train_dataset=tokenized_train,
                                           eval_dataset=tokenized_test,
                                           tokenizer=tokenizer,
                                           callbacks=[early_stopper]
                                           )
            # this will start a run
            trainer.train()

            # we log the roc auc on the eval set - can be changed
            metrics = compute_metrics(trainer,tokenized_test)
            #mlflow.log_metric("eval_roc_auc",metrics['auc'])
            mlflow.log_metric("eval_accuracy",metrics['accuracy'])

            # save the best model (best model was loaded at end)
            trainer.save_model(ARTIFACTS.get(SAVED_MODEL_NAME))

            # this class based on the PythonModel flavor allows us to easily package our pipeline
            class HF_Wrapper(mlflow.pyfunc.PythonModel):

                def load_context(self, context):
                    import transformers

                    # retrieve model path
                    model_path=context.artifacts[SAVED_MODEL_NAME]
                    # load model and tokenizer
                    self.tokenizer=transformers.DistilBertTokenizer.from_pretrained(model_path) # tokenizer
                    self.model = transformers.AutoModelForSequenceClassification.from_pretrained(model_path) # model


                def predict(self, context, model_input):
                    import tqdm
                    import numpy as np
                    import transformers
                    import datasets
                    from transformers.pipelines.base import KeyDataset

                    # convert the data to hf format
                    hf_data = datasets.Dataset.from_pandas(model_input)

                    # pipe with scores - tokenizer arguments can be changed
                    pipe = transformers.TextClassificationPipeline(model=self.model, tokenizer=self.tokenizer, return_all_scores=True)
                    tokenizer_kwargs = {'padding':True,'truncation':True,'max_length':512}

                    # create output proba
                    probas = np.empty([len(hf_data),NUM_LABELS])

                    for i,proba_array in tqdm.tqdm(enumerate(pipe(KeyDataset(hf_data,'text'),batch_size=INF_BATCH_SIZE,**tokenizer_kwargs))):
                        probas[i]=[class_dict['score'] for class_dict in proba_array]

                    return np.array(probas)

            # define signature of the model
            input_schema = Schema([ColSpec("string", "text")]) # our input is a text column
            output_schema = Schema([ColSpec("float")])  # our output is a probability
            signature = ModelSignature(inputs=input_schema, outputs=output_schema)

            # define the artifact path where ml_flow model will be stored
            artifact_path = str(run_id)+'/'
            mlflow.pyfunc.log_model(artifact_path=artifact_path,python_model=HF_Wrapper(),artifacts=ARTIFACTS,
                                    signature=signature)
            print("Model logged")
            mlflow_extension.set_run_inference_info(run_id=run_id,
                                                    prediction_type=PREDICTION_TYPE,
                                                    classes=CLASSES,
                                                    code_env_name=MLFLOW_CODE_ENV_NAME,target=TARGET_NAME)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # 4) Deploy Model with best Deployment metric

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# This part does not need to be modified.

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# In this part:
# - we find the run in the experiment that resulted in the best accuracy;
# - we deploy the corresponding model as a  saved model in the Flow. When clicking on the last active version, you will have access to performance assets such as confusion matrix, lift charts etc.
# 
# This choice could be overriden by the user as it is possible to deploy any of the runs directly from the Experiment Tracking Section.

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
if AUTO_DEPLOY:
    mlflow_handle = project.setup_mlflow(managed_folder=EXPERIMENT_FOLDER_ID)

    # Get experiment
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    print("Experiment: ", experiment)

    # List the runs and get the one with the best accuracy score
    print("Looking for the best run within the experiment")
    best_run = None
    for run_info in mlflow.list_run_infos(experiment.experiment_id):
        run = mlflow.get_run(run_info.run_id)
        if best_run is None:
            best_run = mlflow.get_run(run_info.run_id)
        elif run.data.metrics.get(DEPLOYMENT_METRIC, 0) > best_run.data.metrics.get(DEPLOYMENT_METRIC, 0):
            best_run = run
    print(f"Run id {best_run.info.run_id} with {DEPLOYMENT_METRIC}={best_run.data.metrics.get(DEPLOYMENT_METRIC)}")

    # Deploy the model on the flow
    run_id = best_run.info.run_id
    prediction_type = DSSPredictionMLTaskSettings.PredictionTypes.MULTICLASS

    # Get or create the Saved Model
    sm_id = None
    for sm in project.list_saved_models():
        if sm["name"] != SAVED_MODEL_NAME:
            continue
        else:
            sm_id = sm["id"]
            print(f"Found Saved Model {sm['name']} with id {sm['id']}")
            break
    if sm_id:
        saved_model = project.get_saved_model(sm_id)
    else:
        saved_model = project.create_mlflow_pyfunc_model(SAVED_MODEL_NAME, PREDICTION_TYPE)
        sm_id = saved_model.id
        print(f"Saved Model not found, created new one with id {sm_id}")

    # Define model version
    model_versions=[model['id'] for model in saved_model.list_versions()]

    if model_versions is None:
        version_id = "finetuning_v1"
    else:
        i=1
        while "finetuning_v{}".format(i) in model_versions:
            i+=1
        version_id = "finetuning_v{}".format(i)

    print(f"Deploying the model {SAVED_MODEL_NAME} on the flow and running evaluation with dataset {EVAL_DATASET}")
    sm_external_model_version_handler = mlflow_extension.deploy_run_model(
        run_id,
        sm_id,
        evaluation_dataset=EVAL_DATASET,
        version_id=version_id,
        target_column_name=TARGET_NAME
    )
else:
    print('No model deployed - experiments are available in the X-Tracking Section')
