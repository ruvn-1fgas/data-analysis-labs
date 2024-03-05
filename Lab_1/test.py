# %% [markdown]
# # Лабораторная работа №1 - Data Science London + Scikit-learn

# %%
from sklearnex import patch_sklearn
import warnings

warnings.filterwarnings("ignore")

patch_sklearn()

# %% [markdown]
# ### Препроцессинг, функции для загрузки данных

# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.mixture import GaussianMixture

input_path = "input/data-science-london-scikit-learn"
RANDOM_STATE = 1337


def grid_search(model, params, X, y):
    grid_search = GridSearchCV(
        model,
        params,
        refit=True,
        cv=10,
        verbose=1,
        n_jobs=-1,
        scoring="accuracy",
        return_train_score=True,
    )
    grid_search.fit(X, y)

    print(f"\n\033[92mModel:\033[0m {model.__class__.__name__}")
    print(f"\033[92mBest params:\033[0m {grid_search.best_params_}")
    print(
        f"\033[92mTrain score:\033[0m {grid_search.cv_results_['mean_train_score'][grid_search.best_index_] * 100:.2f}"
    )
    print(f"\033[92mBest score:\033[0m {grid_search.best_score_ * 100:.2f}\n")

    return grid_search.best_estimator_, grid_search.best_score_


def get_preprocessed_data():
    train_data = pd.read_csv(f"{input_path}/train.csv", header=None)
    test_data = pd.read_csv(f"{input_path}/test.csv", header=None)

    x_all = np.r_[train_data, test_data]

    lowest_bic = np.infty
    bic = []
    n_components_range = range(1, 5)
    cv_types = ["spherical", "tied", "diag", "full"]
    for cv_type in cv_types:
        for n_components in n_components_range:
            gmm = GaussianMixture(
                n_components=n_components,
                covariance_type=cv_type,
                random_state=RANDOM_STATE,
            )
            gmm.fit(x_all)
            bic.append(gmm.bic(x_all))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm

    gmm = best_gmm
    gmm.fit(x_all)
    gmm_train = gmm.predict_proba(train_data)
    gmm_test = gmm.predict_proba(test_data)

    return gmm_train, gmm_test


# %% [markdown]
# ### Загрузка данных

# %%
train_data, test_data = get_preprocessed_data()
train_labels = pd.read_csv(f"{input_path}/trainLabels.csv", header=None)

# %% [markdown]
# ### Создание сабмита


# %%
def create_submission(model, name):
    predictions = model.predict(test_data)

    submission = pd.DataFrame(
        {"Id": range(1, len(predictions) + 1), "Solution": predictions}
    )
    submission.to_csv(f"{name}.csv", index=False)


# %% [markdown]
# ### KNeighborsClassifier

# %%
from sklearn.neighbors import KNeighborsClassifier as KNC


def model_knn():
    param_grid = {
        "n_neighbors": [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21],
        "weights": ["uniform", "distance"],
        "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
    }
    return grid_search(
        KNC(),
        param_grid,
        train_data,
        train_labels,
    )


# %% [markdown]
# ### RandomForestClassifier

# %%
from sklearn.ensemble import RandomForestClassifier as RFC


def model_rfc():
    param_grid = {
        "n_estimators": [100, 200, 300, 400, 500, 1000],
        "max_depth": [9, 18, 27],
        "max_features": [6, 7, 8, 9],
    }
    return grid_search(
        RFC(random_state=RANDOM_STATE, max_depth=18, max_features=9, n_estimators=200),
        param_grid,
        train_data,
        train_labels,
    )


# %% [markdown]
# ### GaussianProcessClassifier

# %%
from sklearn.gaussian_process import GaussianProcessClassifier as GPC


def model_gpc():
    param_grid = {
        "max_iter_predict": [100, 200, 300, 400, 500, 1000],
    }
    return grid_search(
        GPC(random_state=RANDOM_STATE),
        param_grid,
        train_data,
        train_labels,
    )


# %% [markdown]
# ### Подбор гиперпараметров и создание сабмишеннов

# %%
submissions_folder = "submissions"

items = list(globals().items())

for name, func in items:
    if callable(func) and name.startswith("model_"):
        model, score = func()
        create_submission(
            model, f"{submissions_folder}/{score:.4f}_{model.__class__.__name__}"
        )

# %% [markdown]
# ### Сравниваем с ~100% результатом

# %%
import os

submission_100 = pd.read_csv(f"{input_path}/100_score.csv")

for submission_fn in os.listdir(submissions_folder):
    if submission_fn.endswith(".csv"):
        submission = pd.read_csv(f"{submissions_folder}/{submission_fn}")

        accuracy = (submission["Solution"] == submission_100["Solution"]).mean()
        model_name = submission_fn.split("_")[-1].split(".")[0]

        print(f"{model_name}: {accuracy:.4f}")

        submission.to_csv(f"best/{accuracy:.4f}_{model_name}.csv", index=False)
