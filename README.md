# FinalProject

## Our final project purpose to examine the impact of Surgeon and Hospital Procedural Volume on Outcomes after CABG procedure . <br />


### python files: 

#### Bins.py -  Divided each line of yearly experience to bins. 
#### GLMM.py -  GLMM python model.
#### Plots_trend.py -  Generates our various plots depends on selected experience.
#### Reop SHAP output.py - Run the final models on Reop Data and generates confusion matrix and SHAP plots.
#### RetrainModelComplics.py -  Retrain model for Complication models.
#### RetrainModels.py -  Retrain model for Mortality models.
#### SHAP output.py -  Run the final models and generates confusion matrix and SHAP plots.
#### SelectedYears.py -  generates summaries tables of the experience of selected target variable.  
#### tables.py - create six tables with analysis. each type of mortality : 'total_surgery_count', 'total_CABG', 'Reop' 
####                with each outcome: 'Mortalty', 'Complics'. The data for creating the tables is in Tables directory.
####                and its name is HospID/surgid+'_allyears_expec_surgid_STSRCOM.csv'

### <br />
### Directories:
### Tables dir -  Contains all the data we used on the various scripts. 
### ClusterFiles -  Script that we run on cluster, for grid search and SMOTE SHAP outputs. 

