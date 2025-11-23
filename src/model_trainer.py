from imblearn.pipeline import Pipeline 
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import precision_score, recall_score, make_scorer

def train_model(X_train, y_train, preprocessor):

    xgb_classifier = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
        tree_method="hist"
    )

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', xgb_classifier)
    ])

    param_dist = {
        "classifier__n_estimators": [400, 500, 650],
        "classifier__learning_rate": [0.02, 0.03, 0.04],
        "classifier__max_depth": [2, 3],
        "classifier__min_child_weight": [6, 7, 8, 9],
        "classifier__gamma": [0.7, 1.0, 1.3, 1.6],
        "classifier__reg_alpha": [0.0, 0.3, 0.5, 0.8],
        "classifier__reg_lambda": [0.5, 1, 2, 3],
        "classifier__subsample": [0.85, 0.9, 0.95],
        "classifier__colsample_bytree": [0.65, 0.7, 0.75],
        "classifier__scale_pos_weight": [0.6, 0.7, 0.8],
        "classifier__max_delta_step": [1, 2, 3],
    }

    def soft_precision_with_recall_floor(y_true, y_pred, min_recall=0.4):
        r = recall_score(y_true, y_pred)
        p = precision_score(y_true, y_pred)
        if r >= min_recall:
            return p
        return p * (r / min_recall)

    scorer = make_scorer(soft_precision_with_recall_floor)


    search = RandomizedSearchCV(
        model_pipeline,
        param_distributions=param_dist,
        n_iter=100,
        scoring=scorer,
        cv=5,
        n_jobs=-1,
        random_state=42,
        refit=True
    )

    search.fit(X_train, y_train)


    print("✅ [model_trainer] 튜닝 완료")
    print("🏷️ Best params:", search.best_params_)
    return search.best_estimator_
