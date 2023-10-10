from sklearn.model_selection import train_test_split
from urllib.request import urlretrieve
from sklearn import preprocessing
from tqdm import tqdm
import pandas as pd
import numpy as np
import requests
import zipfile
import random
import shutil
import math
import os


def download_dataset(url, folder, filename="tmp.zip"):
    print(f"Could not find dataset. Downloading from {url}...")
    save_path = os.path.join(folder, filename)

    _ = urlretrieve(url, save_path)

    # Streaming, so we can iterate over the response.
    response = requests.get(url, stream=True)
    
    total_size_in_bytes = int(len(response.content))  # int(response.headers.get('content-length', 0))
    
    block_size = 1024  # 1 Kibibyte
    
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    with open(save_path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)

    progress_bar.close()
    
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")

    # Extract the zip
    print("Unzipping dataset...")
    with zipfile.ZipFile(os.path.join(folder, filename), 'r') as zip_ref:
        zip_ref.extractall(folder)


def folder_cleanup(folder):
    # List all the created files and folders for a later cleanup
    created_files = os.listdir(folder)

    for filename in created_files:
        file_path = os.path.join(folder, filename)
        if os.path.isdir(file_path):
            shutil.rmtree(file_path)
        else:
            if filename != "treated_test.csv" and filename != "treated_train.csv":
                os.remove(file_path)


def download_and_preprocess_dataset(dataset_name):
    data_folder = os.path.join('.', 'data', dataset_name)
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    if dataset_name == 'HumanActivityRecognition':
        download_dataset(url="https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip",
                         folder=data_folder)

        with zipfile.ZipFile(os.path.join(data_folder, "UCI HAR Dataset.zip"), 'r') as zip_ref:
            zip_ref.extractall(data_folder)

        x_train = pd.read_csv(os.path.join(data_folder, "UCI HAR Dataset", "train", "X_train.txt"), header=None, sep="\s+")
        x_train['subject'] = pd.read_csv(os.path.join(data_folder, "UCI HAR Dataset", "train", "subject_train.txt"), header=None, sep="\s+")
        y_train = pd.read_csv(os.path.join(data_folder, "UCI HAR Dataset", "train", "y_train.txt"))

        x_test = pd.read_csv(os.path.join(data_folder, "UCI HAR Dataset", "test", "X_test.txt"), header=None, sep="\s+")
        x_test['subject'] = pd.read_csv(os.path.join(data_folder, "UCI HAR Dataset", "test", "subject_test.txt"), header=None, sep="\s+")
        y_test = pd.read_csv(os.path.join(data_folder, "UCI HAR Dataset", "test", "y_test.txt"))

        d = {1: "WALKING", 2: "WALKING_UPSTAIRS",  3: "WALKING_DOWNSTAIRS", 4: "SITTING", 5: "STANDING", 6: "LAYING"}
        y_train = [d[k] for k in y_train.to_numpy().flatten()]
        y_test = [d[k] for k in y_test.to_numpy().flatten()]

        # Convert the categorical classes to numerical values
        mapper, ind = np.unique(y_train, return_inverse=True)
        mapping_dict = dict(zip(y_train, ind))

        y_train = np.array(list(map(mapping_dict.get, y_train)))
        y_test = np.array(list(map(mapping_dict.get, y_test)))

        # Standardize the datasets:
        # x_train_treated = (x_train - x_train.mean(axis=0)) / x_train.std(axis=0)
        # x_test_treated = (x_test - x_train.mean(axis=0)) / x_train.std(axis=0)

        scaler = preprocessing.StandardScaler().fit(x_train)
        x_train_treated = scaler.transform(x_train)
        x_test_treated = scaler.transform(x_test)

    elif dataset_name == 'LetterRecognition':
        download_dataset(url="https://archive.ics.uci.edu/static/public/59/letter+recognition.zip", folder=data_folder)

        df = pd.read_csv(os.path.join(data_folder, 'letter-recognition.data'))

        x = np.array(df.drop(['T'], axis=1))
        le = preprocessing.LabelEncoder()
        le.fit(df['T'].values)
        y = le.transform(df['T'].values)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)

        # Standardize the datasets:
        x_train_treated = (x_train - x_train.mean(axis=0)) / x_train.std(axis=0)
        x_test_treated = (x_test - x_train.mean(axis=0)) / x_train.std(axis=0)

    elif dataset_name == 'Pendigits':
        download_dataset(url="https://archive.ics.uci.edu/static/public/81/pen+based+recognition+of+handwritten+digits.zip", folder=data_folder)

        train_data = pd.read_csv(os.path.join(data_folder, 'pendigits.tra'), header=None)
        test_data = pd.read_csv(os.path.join(data_folder, 'pendigits.tes'), header=None)

        x_train = train_data.drop([train_data.columns[-1]], axis=1)
        y_train = train_data[train_data.columns[-1]]

        x_test = test_data.drop([test_data.columns[-1]], axis=1)
        y_test = test_data[test_data.columns[-1]]

        # Standardize the datasets:
        x_train_treated = (x_train - x_train.mean(axis=0)) / x_train.std(axis=0)
        x_test_treated = (x_test - x_train.mean(axis=0)) / x_train.std(axis=0)

    elif dataset_name == 'USCensus1990':
        download_dataset(url="https://archive.ics.uci.edu/static/public/116/us+census+data+1990.zip", folder=data_folder)

        data = pd.read_csv(os.path.join(data_folder, 'USCensus1990.data.txt'), dtype=np.int16).drop(['caseid'], axis=1)

        x = data.drop(['iYearsch'], axis=1).values
        y = data['iYearsch'].values

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)

        # Standardize the datasets:
        x_train_treated = (x_train - x_train.mean(axis=0)) / x_train.std(axis=0)
        x_test_treated = (x_test - x_train.mean(axis=0)) / x_train.std(axis=0)

    elif dataset_name == 'multiple_feature':
        download_dataset(url="https://archive.ics.uci.edu/static/public/72/multiple+features.zip", folder=data_folder)

        df = pd.read_csv(os.path.join(data_folder, 'mfeat-fac'), header=None, sep="\s+")

        # According to the paper:
        #    "The first 200 patterns are of class `0', followed by sets of 200 patterns for each of the classes `1' - `9'."
        y = np.array([np.repeat(c, 200) for c in range(0, 10)]).flatten()

        X_fac = np.array(df.drop([0], axis=1))

        x_train, x_test, y_train, y_test = train_test_split(X_fac, y, test_size=0.20, random_state=42)

        # Standardize the datasets:
        x_train_treated = (x_train - x_train.mean(axis=0)) / x_train.std(axis=0)
        x_test_treated = (x_test - x_train.mean(axis=0)) / x_train.std(axis=0)

    elif dataset_name == 'optdigits':
        download_dataset(url="https://archive.ics.uci.edu/static/public/80/optical+recognition+of+handwritten+digits.zip", folder=data_folder)

        train_df = pd.read_csv(os.path.join(data_folder, 'optdigits.tra'), header=None)
        test_df = pd.read_csv(os.path.join(data_folder, 'optdigits.tes'), header=None)

        # Drop the constant columns:
        train_df.drop([0, 39], axis=1, inplace=True)
        test_df.drop([0, 39], axis=1, inplace=True)

        # According to the paper:
        #    "All input attributes are integers in the range 0..16.
        #     The last attribute is the class code 0..9"
        X_train = train_df.drop([64], axis=1).to_numpy()
        y_train = train_df[64].to_numpy()
        X_test = test_df.drop([64], axis=1).to_numpy()
        y_test = test_df[64].to_numpy()

        # Standardize the datasets:
        x_train_treated = (X_train - X_train.mean(axis=0)) / X_train.std(axis=0)
        x_test_treated = (X_test - X_train.mean(axis=0)) / X_train.std(axis=0)

    elif dataset_name == 'cnae_9':
        download_dataset(url="https://archive.ics.uci.edu/static/public/233/cnae+9.zip", folder=data_folder)

        df = pd.read_csv(os.path.join(data_folder, 'CNAE-9.data'), sep=',', header=None)

        X = np.array(df.drop([0], axis=1))
        y = np.array(df[0])

        x_train_treated, x_test_treated, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    else:
        raise ValueError(
            f"Unsupported dataset: {dataset_name}. Please complete this code section and define known/unknown classes.")

    # Export datasets:
    print("Exporting dataset...")
    treated_train_df = pd.concat([pd.DataFrame(x_train_treated), pd.DataFrame({'classes': y_train})], axis=1)
    treated_test_df = pd.concat([pd.DataFrame(x_test_treated), pd.DataFrame({'classes': y_test})], axis=1)
    treated_train_df.to_csv(os.path.join(data_folder, 'treated_train.csv'), index=False)
    treated_test_df.to_csv(os.path.join(data_folder, 'treated_test.csv'), index=False)

    # Clean downloaded files:
    folder_cleanup(data_folder)


def import_train_and_test_datasets(dataset_name):
    dataset_folder_path = os.path.join('.', 'data', dataset_name)

    train_path = os.path.join(dataset_folder_path, 'treated_train.csv')
    test_path = os.path.join(dataset_folder_path, 'treated_test.csv')

    # If the dataset doesn't exists, it must be downloaded
    if not os.path.isfile(train_path) or not os.path.isfile(test_path):
        download_and_preprocess_dataset(dataset_name)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    x_train = np.array(train_df.drop(['classes'], axis=1))
    y_train = np.array(train_df['classes'])

    x_test = np.array(test_df.drop(['classes'], axis=1))
    y_test = np.array(test_df['classes'])

    return x_train, y_train, x_test, y_test


def joint_shuffling(x, y):
    np.random.seed(0)
    p = np.random.permutation(len(x))
    return x[p], y[p]


def shuffle_and_split_dataframe(df, seed=0):
    rand_index = np.arange(len(df))
    random.seed(seed)
    random.shuffle(rand_index)
    
    train_indices = rand_index[:math.floor(0.8*len(df))]
    test_indices = rand_index[math.floor(0.8*len(df)):]

    return df.iloc[train_indices], df.iloc[test_indices]


def select_known_unknown_classes(x_train, y_train, x_test, y_test, unknown_ratio=None, known_classes=None, unknown_classes=None):
    if unknown_ratio is None and known_classes is None and unknown_classes is None:
        raise ValueError(f"Please specify either an unknown_ratio or the known and unknown classes.")
    if unknown_ratio is not None and (known_classes is not None or unknown_classes is not None):
        raise ValueError(f"Please specify either an unknown_ratio or the known and unknown classes.")
    if (known_classes is None and unknown_classes is not None) or (known_classes is None and unknown_classes is not None):
        raise ValueError(f"Please specify both values of known and unknown classes.")
    
    # Important step: We select some of the known classes to act as "unknown"

    classes = np.unique(y_train)
    print(f"There are {len(classes)} classes in total")

    if unknown_ratio is not None:
        rand_index = np.arange(len(classes))
        random.seed(0)
        random.shuffle(rand_index)
        
        known_classes = classes[rand_index[math.floor(len(classes) * unknown_ratio):]]
        unknown_classes = classes[rand_index[:math.floor(len(classes) * unknown_ratio)]]
    else:
        if not (isinstance(known_classes, list) and isinstance(unknown_classes, list)) and not (isinstance(known_classes, np.ndarray) and isinstance(unknown_classes, np.ndarray)):
            raise ValueError(f"known_classes and unknown_classes must be either list or np.array")

    print(f"{len(known_classes)} known classes")
    print(f"{len(unknown_classes)} unknown classes")

    y_train_save = y_train.copy()
    y_train = np.array(pd.Series(y_train).astype('category').cat.codes).astype(np.int16)  # Numerize
    y_train[np.in1d(y_train_save, unknown_classes)] = 9999  # We want the unknown_class_value to be the last class number, since we want the known classes in {0, ..., C}
    
    y_test_save = y_test.copy()
    y_test = np.array(pd.Series(y_test).astype('category').cat.codes).astype(np.int16)  # Numerize
    y_test[np.in1d(y_test_save, unknown_classes)] = 9999
    
    print(f"x_train_df.shape={x_train.shape}")
    print(f"x_test_df.shape={x_test.shape}")

    # Numerize the targets
    classifier_mapper, classifier_ind = np.unique(y_train, return_inverse=True)
    classifier_mapping_dict = dict(zip(y_train, classifier_ind))
    
    y_train = np.array(list(map(classifier_mapping_dict.get, y_train)))
    y_test = np.array(list(map(classifier_mapping_dict.get, y_test)))

    unknown_class_value = classifier_mapping_dict[9999]

    return x_train, y_train, x_test, y_test, unknown_class_value, y_train_save, y_test_save


def import_dataset_with_name(dataset_name):
    x_train, y_train, x_test, y_test = import_train_and_test_datasets(dataset_name)

    if dataset_name == 'HumanActivityRecognition':
        known_classes = [1, 2, 5]
        unknown_classes = [0, 3, 4]
        
    elif dataset_name == 'LetterRecognition':
        known_classes = [1, 2, 4, 5, 6, 8, 9, 10, 11, 13, 14, 16, 17, 18, 19, 20, 22, 24, 25]
        unknown_classes = [0, 3, 7, 12, 15, 21, 23]
        
    elif dataset_name == 'Pendigits':
        known_classes = [1, 2, 4, 8, 9]
        unknown_classes = [0, 3, 5, 6, 7]

    elif dataset_name == 'USCensus1990':
        known_classes = [1, 2, 4, 6, 7, 8, 9, 10, 13, 14, 16, 17]
        unknown_classes = [0, 3, 5, 11, 12, 15]
        
        # Too much train and test data, so we cap and balance them 
        train_indices_to_keep = []
        for c in np.unique(y_train):
            train_indices_to_keep += list(np.arange(len(y_train))[y_train == c][:1000])
        train_indices_to_keep.sort()
        x_train = x_train[train_indices_to_keep]
        y_train = y_train[train_indices_to_keep]

        test_indices_to_keep = []
        for c in np.unique(y_test):
            test_indices_to_keep += list(np.arange(len(y_test))[y_test == c][:1000])
        test_indices_to_keep.sort()
        x_test = x_test[test_indices_to_keep]
        y_test = y_test[test_indices_to_keep]
        
    elif dataset_name == 'multiple_feature':
        known_classes = [0, 2, 5, 4, 8]
        unknown_classes = [7, 3, 1, 9, 6]
        
    elif dataset_name == 'optdigits':
        known_classes = [2, 1, 9, 8, 7]
        unknown_classes = [3, 6, 0, 5, 4]
        
    elif dataset_name == 'cnae_9':
        known_classes = [5, 0, 7, 8, 4]
        unknown_classes = [2, 9, 6, 3, 1]
        
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Please complete this code section and define known/unknown classes.")

    x_train, y_train = joint_shuffling(x_train, y_train)
    x_test, y_test = joint_shuffling(x_test, y_test)
    
    return select_known_unknown_classes(x_train, y_train, x_test, y_test, known_classes=known_classes, unknown_classes=unknown_classes)
