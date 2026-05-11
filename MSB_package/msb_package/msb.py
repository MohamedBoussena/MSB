from sklearn.base import clone, MetaEstimatorMixin, BaseEstimator
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.impute import KNNImputer
from sksurv.base import SurvivalAnalysisMixin
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union, Any

class MSB(MetaEstimatorMixin, SurvivalAnalysisMixin, BaseEstimator):
    """
    Multimodality Stacking with Blockwise Missing Values (MSB) for survival analysis.

    MSB is a late fusion algorithm designed to integrate predictions from multiple data sources
    (modalities) while explicitly handling blockwise missing values. It is particularly useful
    for survival analysis in clinical datasets where missingness is common.

    Attributes:
        estimators (List[tuple]): List of tuples (name, estimator) for base learners.
        final_estimator (BaseEstimator): The meta-learner for combining base predictions.
        folds (int): Number of cross-validation folds.
        blocks (List[List[str]]): List of modality blocks.
        dict_block (pd.DataFrame): DataFrame mapping blocks to feature codes.
        stable_features (List[str]): Features always present in all modalities.
        random_state (int, optional): Random seed for reproducibility.
        impute (bool): Whether to impute missing values.
        id_name (str): Name of the index column (e.g., patient ID).
        top_feature (List[str], optional): Top features to include in predictions.
        missingness (bool): Whether to include missingness indicators.
        missing_matrix (pd.DataFrame, optional): Precomputed missingness matrix.
    """

    def __init__(
        self,
        estimators: List[tuple],
        final_estimator: BaseEstimator,
        dict_block: pd.DataFrame,
        folds: int = 3,
        blocks: List[List[str]] = [
            ['clin_bm'], ["MIIPP"], ["HDX"], ["ACP"], ["MIVBL"],
            ['InnatePharma'], ['Mutations']
        ],
        stable_features: List[str] = [],
        random_state: Optional[int] = None,
        impute: bool = True,
        id_name: str = 'SUBJID',
        top_feature: Optional[List[str]] = None,
        missingness: bool = True,
        missing_matrix: Optional[pd.DataFrame] = None
    ) -> None:
        """
        Initialize the MSB model.

        Args:
            estimators: List of tuples (name, estimator) for base learners.
            final_estimator: The meta-learner for combining base predictions.
            dict_block: DataFrame mapping blocks to feature codes.
            folds: Number of cross-validation folds.
            blocks: List of modality blocks.
            stable_features: Features always present in all modalities.
            random_state: Random seed for reproducibility.
            impute: Whether to impute missing values.
            id_name: Name of the index column (e.g., patient ID).
            top_feature: Top features to include in predictions.
            missingness: Whether to include missingness indicators.
            missing_matrix: Precomputed missingness matrix.
        """
        super().__init__()
        self.estimators = estimators
        self.cv_estimators = [(estimator[0], clone(estimator[1])) for estimator in estimators]
        self.folds = folds
        self.final_estimator = clone(final_estimator)
        self.blocks = blocks
        self.dict_block = dict_block
        self.stable_features = stable_features
        self.random_state = random_state
        self.dict_estimators = {
            block: [(estimator[0], clone(estimator[1])) for estimator in self.estimators]
            for block in self.blocks
        }
        self.impute = impute
        self.dict_imputers = {}
        self.id_name = id_name
        self.top_feature = top_feature
        self.missingness = missingness
        self.missing_matrix = missing_matrix

    def missing_block_matrix(self, X: pd.DataFrame) -> None:
        """
        Compute the missingness matrix for each block.

        Args:
            X: Input feature matrix.
        """
        self.missing_matrix = pd.DataFrame(index=X.index)
        for b in self.blocks:
            self.missing_matrix = self.missing_matrix.merge(
                X.filter(items=list(self.dict_block[self.dict_block.block == b].code))
                .isna().mean(axis=1).rename('R_' + b),
                left_index=True, right_index=True
            )

    def block_splitting(
        self, X: pd.DataFrame, y: Optional[Any], block: str
    ) -> tuple:
        """
        Split data into blocks based on missingness.

        Args:
            X: Input feature matrix.
            y: Target variable.
            block: Current block name.

        Returns:
            Tuple of (X_block, y_block) for the specified block.
        """
        if block == 'all':
            X_block_fit = X
            y_block_fit = y
        else:
            X_block_fit = X.filter(
                items=list(self.dict_block[self.dict_block.block == block].code)
            )[self.missing_matrix['R_' + block] < 0.5]
            if y is not None:
                y_block_fit = y[np.where(self.missing_matrix['R_' + block] < 0.5)]
            else:
                y_block_fit = None
        return X_block_fit, y_block_fit

    def _fit_predict_estimators(self, X: pd.DataFrame, y: Any) -> pd.DataFrame:
        """
        Fit base estimators and generate cross-validated predictions.

        Args:
            X: Input feature matrix.
            y: Target variable.

        Returns:
            DataFrame of cross-validated predictions.
        """
        scores_block_cv = pd.DataFrame(index=X.index)
        self.missing_block_matrix(X)
        for block in self.blocks:
            X_block_fit, y_block_fit = self.block_splitting(X, y, block)
            cv = KFold(n_splits=self.folds, random_state=self.random_state, shuffle=True)
            if self.impute:
                self.dict_imputers[block] = KNNImputer().set_output(transform='pandas')
                self.dict_imputers[block].fit(X_block_fit)
                X_block_fit = self.dict_imputers[block].transform(X_block_fit)
            df_block_cv = pd.DataFrame(index=X_block_fit.index)
            for estimator in self.cv_estimators:
                model = estimator[1]
                score_pred = cross_val_predict(model, X_block_fit, y_block_fit, cv=cv)
                df_block_cv[block + "_" + model.__class__.__name__ + "_"] = score_pred
            for estimator in self.dict_estimators[''.join(block)]:
                model = estimator[1]
                model.fit(X_block_fit, y_block_fit)
            scores_block_cv = scores_block_cv.merge(
                df_block_cv, right_index=True, left_index=True, validate='one_to_one', how='left'
            )
        if self.missingness:
            scores_block_cv = scores_block_cv.merge(
                self.missing_matrix, right_index=True, left_index=True, validate='one_to_one', how='left'
            )
        if self.top_feature:
            scores_block_cv = scores_block_cv.merge(
                X.filter(items=self.top_feature), left_index=True, right_index=True, validate='one_to_one', how='left'
            )
        return scores_block_cv

    def predict_estimators(self, X: pd.DataFrame, y: Optional[Any] = None) -> pd.DataFrame:
        """
        Generate predictions from base estimators.

        Args:
            X: Input feature matrix.
            y: Target variable (optional).

        Returns:
            DataFrame of predictions.
        """
        scores_block = pd.DataFrame(index=X.index)
        self.missing_block_matrix(X)
        for block in self.blocks:
            X_block_fit, y_block_fit = self.block_splitting(X, y, block)
            if self.impute:
                X_block_fit = self.dict_imputers[block].transform(X_block_fit)
            df_block = pd.DataFrame(index=X_block_fit.index)
            for estimator in self.dict_estimators[block]:
                model = estimator[1]
                X_block_fit = X_block_fit.reset_index()
                X_block_fit = X_block_fit.set_index(self.id_name)
                score_pred = model.predict(X_block_fit)
                df_block[block + "_" + model.__class__.__name__ + "_"] = score_pred
            scores_block = scores_block.merge(
                df_block, right_index=True, left_index=True, how='left'
            )
        if self.missingness:
            scores_block = scores_block.merge(
                self.missing_matrix, right_index=True, left_index=True, how='left'
            )
        if self.top_feature:
            scores_block = scores_block.merge(
                X.filter(items=self.top_feature), left_index=True, right_index=True, how='left'
            )
        return scores_block

    def fit(self, X: pd.DataFrame, y: Any) -> 'MSB':
        """
        Fit the MSB model.

        Args:
            X: Input feature matrix.
            y: Target variable.

        Returns:
            Fitted MSB instance.
        """
        scores_block_cv = self._fit_predict_estimators(X, y)
        self.columns = scores_block_cv.columns
        self.final_estimator.fit(scores_block_cv, y)
        return self

    def predict(self, X: pd.DataFrame, y: Optional[Any] = None) -> Any:
        """
        Predict using the MSB model. This outputs the default prediction of the final_estimator.

        Args:
            X: Input feature matrix.
            y: Target variable (optional).

        Returns:
            Predictions from the final estimator.
        """
        scores_predict = self.predict_estimators(X)
        return self.final_estimator.predict(scores_predict)

    def predict_proba(self, X: pd.DataFrame, y: Optional[Any] = None) -> Any:
        """
        Predict class probabilities using the MSB model.

        Args:
            X: Input feature matrix.
            y: Target variable (optional).

        Returns:
            Class probabilities from the final estimator.
        """
        scores_predict = self.predict_estimators(X)
        return self.final_estimator.predict_proba(scores_predict)

    def predict_survival_function(
        self, X: pd.DataFrame, y: Optional[Any] = None
    ) -> Any:
        """
        Predict survival functions using the MSB model.

        Args:
            X: Input feature matrix.
            y: Target variable (optional).

        Returns:
            Survival functions from the final estimator.
        """
        scores_predict = self.predict_estimators(X)
        return self.final_estimator.predict_survival_function(scores_predict)

    def predict_cumulative_hazard_function(
        self, X: pd.DataFrame, y: Optional[Any] = None
    ) -> Any:
        """
        Predict cumulative hazard functions using the MSB model.

        Args:
            X: Input feature matrix.
            y: Target variable (optional).

        Returns:
            Cumulative hazard functions from the final estimator.
        """
        scores_predict = self.predict_estimators(X)
        return self.final_estimator.predict_cumulative_hazard_function(scores_predict)

    def score(self, X: pd.DataFrame, y: Any) -> float:
        """
        Score the MSB model.

        Args:
            X: Input feature matrix.
            y: Target variable.

        Returns:
            Model score.
        """
        scores_predict = self.predict_estimators(X)
        return self.final_estimator.score(scores_predict, y)
