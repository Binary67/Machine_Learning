import pandas as pd
import numpy as np
import featuretools as ft
import warnings

warnings.filterwarnings('ignore')

PASSANGER = pd.read_csv('Dataset/Train.csv')

# Data Cleaning
PASSANGER = PASSANGER.drop(columns = ['Cabin'])
PASSANGER = PASSANGER.fillna(PASSANGER.mean())
PASSANGER['Sex'].replace(['male', 'female'], [1, 0], inplace = True)
PASSANGER['Embarked'].replace(['C', 'Q', 'S'], [2, 1, 0], inplace = True)
Y_TRAIN = PASSANGER['Survived']
PASSANGER = PASSANGER.drop(columns = ['Survived'])


# Deep Feature Synthesis
ENTITY_SET = ft.EntitySet(id = 'Passanger_Data')

ENTITY_SET = ENTITY_SET.entity_from_dataframe(entity_id = 'Passanger', dataframe = PASSANGER, index = 'PassengerId')
ENTITY_SET = ENTITY_SET.normalize_entity(base_entity_id = 'Passanger', new_entity_id = 'Package', index = 'Ticket',
                                         additional_variables = ['Parch', 'Fare', 'Embarked']) 

FEATURES_1, FEATURE_NAMES_1 = ft.dfs(entityset = ENTITY_SET, target_entity = 'Passanger', max_depth = 3)

print(FEATURE_NAMES_1)