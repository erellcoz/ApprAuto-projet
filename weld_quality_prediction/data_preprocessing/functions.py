import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re 
import json
from sklearn.linear_model import LogisticRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from scipy.stats import pearsonr
from scipy.stats import shapiro
from sklearn.decomposition import PCA
from typing import Literal
from sklearn.preprocessing import Normalizer, StandardScaler


column_names = [
    "Carbon_concentration",
    "Silicon_concentration",
    "Manganese_concentration",
    "Sulphur_concentration",
    "Phosphorus_concentration",
    "Nickel_concentration",
    "Chromium_concentration",
    "Molybdenum_concentration",
    "Vanadium_concentration",
    "Copper_concentration",
    "Cobalt_concentration",
    "Tungsten_concentration",
    "Oxygen_concentration",
    "Titanium_concentration",
    "Nitrogen_concentration",
    "Aluminium_concentration",
    "Boron_concentration",
    "Niobium_concentration",
    "Tin_concentration",
    "Arsenic_concentration",
    "Antimony_concentration",
    "Current",
    "Voltage",
    "AC_or_DC",
    "Electrode_positive_or_negative",
    "Heat_input",
    "Interpass_temperature",
    "Type_of_weld",
    "Post_weld_heat_treatment_temperature",
    "Post_weld_heat_treatment_time",
    "Yield_strength",
    "Ultimate_tensile_strength",
    "Elongation",
    "Reduction_of_Area",
    "Charpy_temperature",
    "Charpy_impact_toughness",
    "Hardness",
    "50%_FATT",
    "Primary_ferrite_in_microstructure",
    "Ferrite_with_second_phase",
    "Acicular_ferrite",
    "Martensite",
    "Ferrite_with_carbide_aggregate",
    "Weld_ID"
]

sulphur_and_phosphorus_columns = ["Sulphur_concentration","Phosphorus_concentration"]

other_concentration_columns = ["Carbon_concentration",
        "Silicon_concentration",
        "Manganese_concentration",
        "Nickel_concentration",
        "Chromium_concentration",
        "Molybdenum_concentration",
        "Vanadium_concentration",
        "Copper_concentration",
        "Cobalt_concentration",
        "Tungsten_concentration",
        "Oxygen_concentration",
        "Titanium_concentration",
        "Nitrogen_concentration",
        'Nitrogen_concentration_residual',
        "Aluminium_concentration",
        "Boron_concentration",
        "Niobium_concentration",
        "Tin_concentration",
        "Arsenic_concentration",
        "Antimony_concentration"]

label_names = ['Yield_strength', 'Ultimate_tensile_strength', 'Elongation', 'Reduction_of_Area', 'Charpy_temperature', 
                   'Charpy_impact_toughness', 'Hardness', '50%_FATT', 'Primary_ferrite_in_microstructure', 'Ferrite_with_second_phase', 
                   'Acicular_ferrite', 'Martensite', 'Ferrite_with_carbide_aggregate', 'Hardness_load']

physical_ordinal_properties_columns = [
        'Current', 
        'Voltage',
        'Heat_input',
        'Interpass_temperature',
        'Post_weld_heat_treatment_temperature',
        'Post_weld_heat_treatment_time', 
    ]

physical_categorical_properties_columns = [
    'AC_or_DC',
    'Electrode_positive_or_negative',
    'Type_of_weld'
]


def replace_data(df):
    df.replace('N', np.nan, inplace=True)
    return df


# Copy the initial dataset to apply transformations
def choose_labels(df, labels_chosen):
    # Create a copy of the DataFrame to avoid SettingWithCopyWarning
    df = df.copy()  
    
    # Initialize new columns
    df.loc[:, "Hardness_load"] = np.nan
    df.loc[:, "Nitrogen_concentration_residual"] = 0
    
    # Select the chosen labels
    labels = df[labels_chosen]
    
    # Drop specified columns
    inputs = df.drop(columns=label_names + ['Weld_ID'])
    
    return inputs, labels


def sum_values_inferior_to_value(data):
    # Dictionary to store for each the maximum value of the column sum_less_than
    mean_values = {}
    values = data.unique()
    for value in values:
        mean_values[value] = data[data <= value].sum() / data[data <= value].shape[0]
    
    return mean_values


# Values < N
def replace_less_than_values(*, df, column, strategy: Literal['max', 'mean']):
    if df[column].any() and type(df[column].dropna().iloc[0]) == str:  # Check if the column contains string values:
        new_column = column + '_<'  # Create a new column name
        df[new_column] = np.nan  # Create a new column to store the boolean values
        
        # Apply the transformation using .loc to avoid SettingWithCopyWarning
        mask = df[column].apply(lambda x: isinstance(x, str))
        df.loc[mask, column + '_<'] = df.loc[mask, column].str.contains('<')
        df.loc[mask, column] = df.loc[mask, column].replace('<', '', regex=True)
        
        if strategy=='max':
            # Replace the values in the column by the max value
            df.loc[df[new_column] == True, column] = df.loc[df[new_column] == True, column].apply(lambda x: float(x))
        elif strategy=='mean' and df.loc[df[new_column] == True, column].shape[0] > 0:
            # Replace the values in the column by the mean value of the values inferior to max value
            mean_values = sum_values_inferior_to_value(df.loc[df[column].notna(), column].apply(lambda x: float(x)))
            df.loc[df[new_column] == True, column].map(mean_values)

        df.drop(column + '_<', axis=1, inplace=True)
    return df


# Values like 158(Hv30) or 67tot33res
def split_res_values(value, pattern):
    if isinstance(value, str) and pattern in value:
        # Looking for two numbers in the string 
        numbers = re.findall(r'\d+', value)
        if len(numbers) > 1:
            return float(numbers[0]), float(numbers[1])
        else:
            return float(numbers[0]), np.nan  # If only one number is before "res"
    else:
        try:
            return float(value), np.nan  # If there is no "res", return NaN
        except ValueError:
            return np.nan, np.nan


# Values like 150 - 200
def process_interpass_temperature(value):
        if isinstance(value, str) and '-' in value:
            # Split the range, convert to integers, and calculate the mean
            low, high = map(int, value.split('-'))
            return int((low + high) / 2)
        else:
            # Try converting the value to int, or return NaN if not possible
            try:
                return int(value)
            except (ValueError, TypeError):
                return np.nan
            
            
# Handling columns with two numerical values 
def process_string_values(*, inputs, outputs, strategy, labels_chosen):
    for column in inputs.columns:
        inputs = replace_less_than_values(df=inputs, column=column, strategy=strategy)
    inputs['Nitrogen_concentration'], inputs['Nitrogen_concentration_residual'] =  zip(*inputs['Nitrogen_concentration'].apply(lambda x:split_res_values(x, 'res')))
    if 'Hardness' in labels_chosen:
        outputs['Hardness'], outputs['Hardness_load'] =  zip(*outputs['Hardness'].apply(lambda x:split_res_values(x, 'Hv')))
    inputs['Interpass_temperature'] = inputs['Interpass_temperature'].apply(process_interpass_temperature)
    return inputs, outputs


# Converting string values that are actually numeric
def convert_to_numeric_values(df):
    for column in df.columns:
        try:
            df[column] = pd.to_numeric(df[column])
        except Exception:
            continue
    return df


# Instantiate a transformer for mean imputation.
def missing_values_sulphur_and_phosphorus(df_train, df_test):

    imp_mean = SimpleImputer(strategy='mean')
    df_train[sulphur_and_phosphorus_columns] = imp_mean.fit_transform(df_train[sulphur_and_phosphorus_columns])
    df_test[sulphur_and_phosphorus_columns] = imp_mean.transform(df_test[sulphur_and_phosphorus_columns])

    return df_train, df_test


def missing_values_other_concentration(df_train, df_test):

    # Initialize SimpleImputer with constant strategy to fill missing values with 0
    imputer_zero = SimpleImputer(strategy='constant', fill_value=0)
    df_train[other_concentration_columns] = imputer_zero.fit_transform(df_train[other_concentration_columns])
    df_test[other_concentration_columns] = imputer_zero.transform(df_test[other_concentration_columns])

    return df_train, df_test


def univariate_imputation(train_data, test_data, strategy):
    # Create a SimpleImputer object with the specified strategy
    imp = SimpleImputer(strategy=strategy)
    # Fit the imputer on the training data and transform it
    train_imputed = imp.fit_transform(train_data)
    # Transform the test data using the same imputer
    test_imputed = imp.transform(test_data)
    return train_imputed, test_imputed


def multivariate_imputation_ordinal(train_data, test_data):
    # Create an IterativeImputer for ordinal imputation
    imp = IterativeImputer()
    # Fit the imputer on the training data and transform it
    train_imputed = imp.fit_transform(train_data)
    # Transform the test data using the same imputer
    test_imputed = imp.transform(test_data)
    return train_imputed, test_imputed


def multivariate_imputation_categorical(train_data, test_data):
    # Create an IterativeImputer for categorical imputation with a logistic regression estimator
    imp = IterativeImputer(estimator=LogisticRegression())
    # Fit the imputer on the training data and transform it
    train_imputed = imp.fit_transform(train_data)
    # Transform the test data using the same imputer
    test_imputed = imp.transform(test_data)
    return train_imputed, test_imputed


def one_hot_encoding(training_set, testing_set):
    # Store ordinal columns
    ordinal_columns = training_set.select_dtypes(include=[np.number]).columns
    
    # One Hot Encoding on the training_set without dummy_na
    training_encoded = pd.get_dummies(training_set, drop_first=False, dummy_na=True, dtype=float)
    
    # Get the final columns of the training_set
    final_columns = training_encoded.columns

    # Handle NaN values manually
    for column in final_columns:
        if '_nan' in column:
            dummy_columns = [col for col in final_columns if col.startswith(column[:-4])]
            for dummy_col in dummy_columns:
                training_encoded.loc[training_encoded[column] == 1, dummy_col] = np.nan
            training_encoded.drop(column, axis=1, inplace=True)

    # Apply the same One-Hot Encoding to the testing_set
    testing_encoded = pd.get_dummies(testing_set, drop_first=False, dummy_na=True, dtype=float)

    # Ensure that the testing_set has the same columns as the training_set
    # Align the columns of testing_encoded to those of training_encoded
    testing_encoded = testing_encoded.reindex(columns=final_columns, fill_value=0)

    # Final columns after encoding
    final_columns = training_encoded.columns

    # New categorical columns
    new_categorical_columns = list(set(final_columns) - set(ordinal_columns))

    return training_encoded, testing_encoded, new_categorical_columns


def missing_values_physical_properties(train_data, test_data, ordinal_strategy, categorical_strategy, categorical_columns):
    # Distinguish ordinal columns
    ordinal_columns = train_data.columns.difference(categorical_columns)
    
    # Impute ordinal values
    if len(ordinal_columns) > 0:
        if ordinal_strategy == 'mean':
            # Replace missing values with the mean value
            train_data[ordinal_columns], test_data[ordinal_columns] = univariate_imputation(train_data[ordinal_columns], test_data[ordinal_columns], 'mean')
        elif ordinal_strategy == 'linear':
            # Replace missing values with a linear regression
            train_data[ordinal_columns], test_data[ordinal_columns] = multivariate_imputation_ordinal(train_data[ordinal_columns], test_data[ordinal_columns])

    # Impute categorical values
    if len(categorical_columns) > 0:
        if categorical_strategy == 'most_frequent':
            # Replace missing values with the most frequent value
            train_data[categorical_columns], test_data[categorical_columns] = univariate_imputation(train_data[categorical_columns], test_data[categorical_columns], 'most_frequent')
        elif categorical_strategy == 'logistic':
            # Replace missing values with a logistic regression
            train_data[categorical_columns], test_data[categorical_columns] = multivariate_imputation_categorical(train_data[categorical_columns], test_data[categorical_columns])
    
    return train_data, test_data


#After the one hot encoding and the imputation, we need to be sure that the data that has been imputed respects the incompatibility rule (either AC or DC etc...)
def handle_incompatibility_categorical_features(df, incompatible_features_list):
    
    for incompatible_features in incompatible_features_list:
        # Check the rows where all specified columns are equal to zero
        zero_rows = df.index[df[incompatible_features].sum(axis=1) == 0]

        # Check the rows where there is more than one '1' in the specified columns
        more_one_rows = df.index[df[incompatible_features].sum(axis=1) > 1]

        # For the rows where all the specified columns are zero
        for idx in zero_rows:
            # Randomly select one of the incompatible features
            random_col = np.random.choice(incompatible_features)
            # Set the selected column to 1 for the current index
            df.at[idx, random_col] = 1  

        # For the rows where there is more than one '1'
        for idx in more_one_rows:
            # Get the columns that have a value of '1' for the current index
            columns_with_one = df.columns[df.loc[idx] == 1]  # Corrected here
            
            # Ensure there are columns with '1'
            if len(columns_with_one) > 0:
                # Randomly select one of the columns with '1'
                random_col = np.random.choice(columns_with_one)
                # Set all columns with '1' to 0 for the current index
                df.loc[idx, columns_with_one] = 0
                # Set the randomly selected column back to 1 for the current index
                df.at[idx, random_col] = 1  

    return df


def scaler(train_df: pd.DataFrame, test_df: pd.DataFrame, columns_to_normalise: list, strategy: Literal['standard', 'normalizer']):
    if strategy == 'standard':
        # Instantiate a StandardScaler
        scaler = StandardScaler()
    elif strategy == 'normalizer':
        # Instantiate a Normalizer
        scaler = Normalizer()
    else:
        raise ValueError("Invalid strategy. Choose either 'standard' or 'normalizer'.")

    # Fit the scaler only on the categorical columns of the training data
    train_scaled = scaler.fit_transform(train_df[columns_to_normalise])
    test_scaled = scaler.transform(test_df[columns_to_normalise])

    # Convert the normalized data to DataFrames
    train_scaled_df = pd.DataFrame(train_scaled, columns=columns_to_normalise, index=train_df.index)
    test_scaled_df = pd.DataFrame(test_scaled, columns=columns_to_normalise, index=test_df.index)

    # Combine scaled categorical columns with the original DataFrames (excluding the original categorical columns)
    train_final = pd.concat([train_df.drop(columns=columns_to_normalise), train_scaled_df], axis=1)
    test_final = pd.concat([test_df.drop(columns=columns_to_normalise), test_scaled_df], axis=1)

    return train_final, test_final


def compute_pca(train_df, test_df, pca_percent_explained_variance):
    # PCA approach
    pca = PCA(n_components=pca_percent_explained_variance, svd_solver="full")

    # Fit PCA on the training DataFrame and transform the data
    train_concentration_vector = pca.fit_transform(train_df)

    # Transform the test DataFrame using the fitted PCA model
    test_concentration_vector = pca.transform(test_df)

    # Results
    explained_variance_ratio = pca.explained_variance_ratio_  # Variance explained by each component
    n_components = pca.n_components_  # Number of components chosen to explain the specified variance
    
    # Convert the transformed data back to DataFrames, keeping the original index
    train_concentration_data = pd.DataFrame(train_concentration_vector, 
                                            columns=[f'PC{i+1}' for i in range(n_components)],
                                            index=train_df.index)  # Preserve original index
    test_concentration_data = pd.DataFrame(test_concentration_vector, 
                                           columns=[f'PC{i+1}' for i in range(n_components)],
                                           index=test_df.index)  # Preserve original index

    # Print results
    print(f"Number of components chosen by PCA: {n_components}")
    print(f"Explained Variance Ratio: {explained_variance_ratio}")

    return train_concentration_data, test_concentration_data


OrdinalStrategies = Literal["mean", "linear"]
CategoricalStrategies = Literal["most_frequent", "logistic"]
ScalerStrategy = Literal["standard", "normalizer"]
incompatible_features_list = [['AC_or_DC_DC', 'AC_or_DC_AC'], ['Electrode_positive_or_negative_+', 'Electrode_positive_or_negative_0', 'Electrode_positive_or_negative_-'], ['Type_of_weld_MMA', 'Type_of_weld_ShMA', 'Type_of_weld_FCA', 'Type_of_weld_SA', 'Type_of_weld_TSA', 'Type_of_weld_SAA', 'Type_of_weld_GTAA', 'Type_of_weld_GMAA', 'Type_of_weld_NGSAW', 'Type_of_weld_NGGMA']
]
PcaColumns = Literal['concentration', 'all_ordinals']
LessThanList = Literal['max', 'mean']


def pipeline_training_set(*, training_set: pd.DataFrame, training_labels : pd.DataFrame, testing_set: pd.DataFrame, 
                          testing_labels : pd.DataFrame, labels_chosen : list[str], is_PCA: bool, pca_percent_explained_variance: float, 
                          ordinal_strategy: OrdinalStrategies, categorical_strategy: CategoricalStrategies, scaler_strategy: ScalerStrategy,
                          pca_columns: PcaColumns, less_than_strategy: LessThanList):
    

    # Structural errors
    training_set, training_labels = process_string_values(inputs=training_set, outputs=training_labels, labels_chosen=labels_chosen, 
                                                          strategy=less_than_strategy)
    testing_set, testing_labels = process_string_values(inputs=testing_set, outputs=testing_labels, labels_chosen=labels_chosen, 
                                                        strategy=less_than_strategy)

    training_set = convert_to_numeric_values(training_set)
    testing_set = convert_to_numeric_values(testing_set)


    # Missing values
    ## Transform concentrations accordingly to 
    ## "The yield and ultimate tensile strength of steel welds - Tracey Cool a,*, H.K.D.H. Bhadeshia a, D.J.C. MacKay b"
    training_set, testing_set = missing_values_sulphur_and_phosphorus(training_set, testing_set)
    training_set, testing_set = missing_values_other_concentration(training_set, testing_set)

    # One Hot Encoding
    training_set, testing_set, categorical_columns = one_hot_encoding(training_set, testing_set)

    # Missing values
    training_set, testing_set = missing_values_physical_properties(training_set, testing_set, ordinal_strategy=ordinal_strategy, categorical_strategy=categorical_strategy, categorical_columns=categorical_columns)
    
    # Handle the incompatibility concerning the imputation of some categorical values
    training_set = handle_incompatibility_categorical_features(training_set, incompatible_features_list)
    testing_set = handle_incompatibility_categorical_features(testing_set, incompatible_features_list)
    
    # Drop one column for each categorical feature to avoid multicollinearity. Drop the less frequent column
    columns_to_drop = ['AC_or_DC_AC', 'Electrode_positive_or_negative_0', 'Type_of_weld_NGGMA']
    categorical_columns = [col for col in categorical_columns if col not in columns_to_drop]
    training_set = training_set.drop(columns=columns_to_drop)
    testing_set = testing_set.drop(columns=columns_to_drop)
    pd.get_dummies


    # Normalisation
    columns_to_normalise = physical_ordinal_properties_columns + sulphur_and_phosphorus_columns + other_concentration_columns
    training_set_normalised, testing_set_normalised = scaler(training_set, testing_set, columns_to_normalise, strategy=scaler_strategy)


    # Dimension reduction
    if pca_columns == 'concentration':
        pca_columns_list = sulphur_and_phosphorus_columns + other_concentration_columns
        other_columns = physical_ordinal_properties_columns + categorical_columns
        pca_data_training = training_set_normalised[pca_columns_list]
        pca_data_testing = testing_set_normalised[pca_columns_list]
    elif pca_columns == 'all_ordinals':
        pca_columns_list = sulphur_and_phosphorus_columns + other_concentration_columns + physical_ordinal_properties_columns
        other_columns = categorical_columns
        pca_data_training = training_set_normalised[pca_columns_list]
        pca_data_testing = testing_set_normalised[pca_columns_list]

    if is_PCA:
        # Call the PCA function with both training and testing datasets
        train_concentration_data, test_concentration_data = compute_pca(
            pca_data_training, pca_data_testing, pca_percent_explained_variance
        )        

        # Combine the PCA results with the training set
        training_set_processed = pd.concat(
            [train_concentration_data, training_set[other_columns]], 
            axis=1
        )
        
        # Combine the PCA results with the testing set
        testing_set_processed = pd.concat(
            [test_concentration_data, testing_set[other_columns]], 
            axis=1
        )
    
    # Return processed training set and labels
    return training_set_processed, testing_set_processed, training_labels, testing_labels

