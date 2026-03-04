"""Stepwise subset selection for linear regression models.

This module provides a selector that can run forward or backward stepwise
search using an inner criterion (used during path construction) and an outer
criterion (used to choose the final subset along the path).
"""

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import KFold, cross_val_score

import metrics as mt

class Stepwise_regression_selector:
    """Stepwise feature subset selector for regression.

    Args:
        inner_crit: Criterion used inside each step (`'RSS'` or `'R2'`).
        outer_crit: Criterion used to pick the best subset from the full path
            (`'adjusted-R2'`, `'AIC'`, `'BIC'`, `'Cp'`, or `'cv'`).
        cv_fold: Number of folds for cross-validation when `outer_crit='cv'`.
    """
    
    def __init__(self, inner_crit, outer_crit, cv_fold=None):
        """Initialize selector settings and validate inputs."""
        self._validate_input(inner_crit, outer_crit, cv_fold)
        self.inner_crit = inner_crit
        self.outer_crit = outer_crit
        self.cv_fold = cv_fold
               
        self.subsets = None
        self.selected_subset = None
        
    
    def step_forward(self, X, Y, model):
        """Run forward stepwise selection.

        Starts from the null model and adds one predictor per step based on
        the best inner score. Then applies outer selection to choose the final
        subset.

        Args:
            X: Predictor matrix of shape (n_samples, n_features).
            Y: Response vector of shape (n_samples,).
            model: Scikit-learn compatible regressor with `fit` and `predict`.
        """
        # --- Inner Selection --------------------------------------------------
        n, p = X.shape
        selected_subsets = []
        selected_predictors = []
        remaining_predictors = list(range(p))
        
        null_result = self._record_null_model(Y)
        selected_subsets.append(null_result)
        
        for k in range(p):
            best_rss = None
            best_score = -np.inf
            best_predictor = None
            
            for j in remaining_predictors:
                subset = selected_predictors + [j]
                X_subset = X[:, subset]
                
                m = clone(model)
                m.fit(X_subset, Y)
                preds = m.predict(X_subset)
                score = self._compute_inner_score(Y, preds)
                
                
                if score > best_score:
                    best_rss = mt.RSS(Y, preds)
                    best_score = score
                    best_predictor = j
            
            selected_predictors.append(best_predictor)
            remaining_predictors.remove(best_predictor)
            selected_subsets.append(
                self._record_result(k+1, selected_predictors.copy(), best_rss, best_score)
            )
        
        self.subsets = selected_subsets
        
        # --- Outer selection --------------------------------------------------
        self._outer_selection(X, Y, model)
        
    
    def step_backward(self, X, Y, model):
        """Run backward stepwise selection.

        Starts from the full model and removes one predictor per step based on
        the best inner score. Then applies outer selection to choose the final
        subset.

        Args:
            X: Predictor matrix of shape (n_samples, n_features).
            Y: Response vector of shape (n_samples,).
            model: Scikit-learn compatible regressor with `fit` and `predict`.
        """
        # --- Inner Selection --------------------------------------------------
        n, p = X.shape
        selected_subsets = []
        selected_predictors = list(range(p))

        # Start from the full model.
        m_full = clone(model)
        m_full.fit(X[:, selected_predictors], Y)
        preds_full = m_full.predict(X[:, selected_predictors])
        score_full = self._compute_inner_score(Y, preds_full)
        rss_full = mt.RSS(Y, preds_full)
        selected_subsets.append(
            self._record_result(p, selected_predictors.copy(), rss_full, score_full)
        )

        # Remove one predictor at a time.
        for _ in range(p):
            best_rss = None
            best_score = -np.inf
            predictor_to_remove = None

            for j in selected_predictors:
                subset = [idx for idx in selected_predictors if idx != j]

                if len(subset) == 0:
                    preds = np.full_like(Y, np.mean(Y), dtype=float)
                else:
                    m = clone(model)
                    m.fit(X[:, subset], Y)
                    preds = m.predict(X[:, subset])

                score = self._compute_inner_score(Y, preds)

                if score > best_score:
                    best_rss = mt.RSS(Y, preds)
                    best_score = score
                    predictor_to_remove = j

            selected_predictors.remove(predictor_to_remove)
            selected_subsets.append(
                self._record_result(
                    len(selected_predictors),
                    selected_predictors.copy(),
                    best_rss,
                    best_score
                )
            )

        self.subsets = sorted(selected_subsets, key=lambda s: s["num_predictors"])

        # --- Outer selection --------------------------------------------------
        self._outer_selection(X, Y, model)


    def _outer_selection(self, X, Y, model):
        """Compute outer scores for all recorded subsets and pick the best."""
        n, p = X.shape
        
        for subset in self.subsets:
            k = subset["num_predictors"]
            selected_predictors = subset["selected_predictors"]
            rss_k = subset["RSS"]
            tss = mt.TSS(Y)
            sigma2_full = None
            
            if self.outer_crit == 'cv':
                subset["outer_score"] = self._compute_outer_cv_score(X, Y, model, selected_predictors)
                
            else: 
                if self.outer_crit == "Cp":
                    full = clone(model).fit(X, Y)
                    rss_full = mt.RSS(Y, full.predict(X))
                    denom = n - p - 1
                    if denom <= 0:
                        raise ValueError(
                            f"Cp requires n-p-1 > 0 for full-model variance. Got n={n}, p={p}."
                        )
                    sigma2_full = rss_full / denom
                
                subset["outer_score"] = self._compute_outer_score(rss_k, tss, n, k, sigma2_full)

        best_idx = int(np.argmax([r["outer_score"] for r in self.subsets]))
        self.selected_subset = self.subsets[best_idx]

        
    def _compute_inner_score(self, Y, preds):
        """Return inner score for a fitted subset (`-RSS` or `R2`)."""
        score = 0
        if self.inner_crit == 'RSS':
            score = -mt.RSS(Y, preds)
        else:
            score = mt.R2(Y, preds)

        return float(score)
    
    
    def _compute_outer_score(self, rss_k, tss, n, k, sigma2_full=None):
        """Return outer score for non-CV criteria.

        The value is oriented so larger is better (`AIC`, `BIC`, and `Cp` are
        negated because they are naturally minimized).
        """
        score = 0
        if self.outer_crit == "adjusted-R2":
            score = mt.adjusted_R2(rss_k, tss, n, k)
        if self.outer_crit == "AIC":
            score = -mt.aic(rss_k, n, k)
        if self.outer_crit == "BIC":
            score = -mt.bic(rss_k, n, k)
        if self.outer_crit == "Cp":
            if sigma2_full is None:
                raise ValueError("Cp requires sigma^2 from the full model.")
            score = -mt.Cp(rss_k, sigma2_full, n, k)
        
        return float(score)
       
        
    def _compute_outer_cv_score(self, X, Y, model, subset):
        """Return mean cross-validated negative MSE for a subset."""
        Xsub = X[:, subset] if len(subset) > 0 else np.ones((len(Y), 1))
        if self.cv_fold > len(Y):
            raise ValueError(
                f"`cv_fold` ({self.cv_fold}) cannot exceed number of samples ({len(Y)})."
            )
        cv = KFold(n_splits=self.cv_fold, shuffle=True, random_state=0)
        scores = cross_val_score(clone(model), Xsub, Y, scoring="neg_mean_squared_error", cv=cv)
        return float(np.mean(scores))
        
 
    def _record_null_model(self, Y):
        """Build and return the result record for the intercept-only model."""
        preds0 = np.full_like(Y, np.mean(Y), dtype=float)
        score0 = self._compute_inner_score(Y, preds0)
        rss0 = mt.RSS(Y, preds0)
        return self._record_result(0, [], rss0, score0)
 
 
    def _record_result(self, num_predictors, predictors, rss, inner_score):
        """Create a standardized subset result dictionary."""
        return {"num_predictors": int(num_predictors),
                "selected_predictors": list(predictors),
                "RSS": float(rss),
                "inner_score": float(inner_score),
                "outer_score": -np.inf
        }


    def _validate_input(self, inner_crit, outer_crit, cv_fold):
        """Validate constructor arguments."""
        if inner_crit not in ['RSS', 'R2']:
            raise ValueError(f"Supported inner criteria: ['RSS', 'R2'(R-squared)].",
                             f"Got {inner_crit}"
            )
        
        if outer_crit not in ['adjusted-R2', 'AIC', 'BIC', 'Cp', 'cv']:
            raise ValueError(f"Supported outer criteria: ",
                             f"['adjusted-R2', 'AIC', 'BIC', 'Cp', 'cv']. ",
                             f"Got {outer_crit}."
            )
            
        if outer_crit == 'cv' and cv_fold is None:
            raise ValueError("To use Cross Validation as the outer criteria, ",
                             "Please specify 'cv_fold' as the number of folds ",
                             "run cross validation on."
            )
        
        if outer_crit == 'cv' and cv_fold < 2:
            raise ValueError(f"`cv_fold` needs to be >= 2, got {cv_fold}")
