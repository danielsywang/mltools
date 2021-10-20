import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline

# 2021.09.14 Created by Daniel SY wang


class ClusterScorePlot(Pipeline):
    """
    ClusterScorePlot inherited from Pipeline,score and plot cluster without labels.
    It's same as pipeline+gridseach.


    Parameters
    ----------
    steps : list
        List of (name, transform) tuples (implementing fit/transform) that are
        chained, in the order in which they are chained, with the last object
        a cluster estimator.

    param_grid : list of dict
        usage reference example part.

    scoring : list
        list of 'si','ca','da' for cluster scoring,
        'si' means metrics.silhouette_score,
        'ca' means metrics.calinski_harabasz_score,
        'da' means metrics.davies_bouldin_score.

    memory : str or object with the joblib.Memory interface, default=None
        Used to cache the fitted transformers of the pipeline. By default,
        no caching is performed. If a string is given, it is the path to
        the caching directory. Enabling caching triggers a clone of
        the transformers before fitting. Therefore, the transformer
        instance given to the pipeline cannot be inspected
        directly. Use the attribute ``named_steps`` or ``steps`` to
        inspect estimators within the pipeline. Caching the
        transformers is advantageous when fitting is time consuming.

    verbose : bool, default=False
        If True, the time elapsed while fitting each step will be printed as it
        is completed.

    Attributes
    ----------
    named_steps : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.
        Read-only attribute to access any step parameter by user given name.
        Keys are step names and values are steps parameters.

    Examples
    --------
    Get the best socres and paramters
    ::

        #setup steps:
        cls = [('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('cluster',cluster.AgglomerativeClustering(n_clusters=5))]

        #setup param_grid:
        lr = LinearRegression() 
        imp = IterativeImputer(estimator=lr,missing_values=np.nan, max_iter=10, verbose=2, imputation_order='roman',random_state=0)
        knn = KNNImputer(n_neighbors=2, add_indicator=True)
        param_grid = [ 
            { 'imputer': [SimpleImputer(strategy='median'),imp,knn, None],
            'scaler': [MinMaxScaler(),StandardScaler(),RobustScaler(),None],
            'cluster':[cluster.AgglomerativeClustering(n_clusters=5),
                        cluster.DBSCAN(eps=0.30, min_samples=10),
                        cluster.KMeans(n_clusters=5),
                        GaussianMixture(n_components=5)]}]

        #setup scoring:
        scoring = ['da', 'si', 'ca']

        #get the best scores:
        clustersearch=Cluster_Score_Plot(steps=cls,param_grid=param_grid,scoring=scoring)
        score_df=clustersearch.get_score(data)
        score_df

    Polt scatter diagram for best paramters
    ::

        #param_grid : setup param_grid using best_score_df param list
        best_score_df=clustersearch.get_best_score(score_df,n_best_score=3)
        param_grid=list(best_score_df['param'])
        
        #plot scatter using best_score_df param
        clustersearch=Cluster_Score_Plot(steps=cls,param_grid=best_score_df,scoring=scoring)
        clustersearch.plot(data,param_type='',ncol=1)

    """

    def __init__(self, steps, param_grid, scoring, memory=None, verbose=False):
        self.param_grid = param_grid
        self.scoring = scoring
        self.X_transformed = None
        super().__init__(steps=steps, memory=memory, verbose=verbose)

    def fit(self, X, y=None, **fit_params):
        """Fit the model

        Fit all the transforms one after the other and transform the
        data, then fit the transformed data using the final estimator.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.

        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps of
            the pipeline.

        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        self : Pipeline
            This estimator
        """

        fit_params_steps = self._check_fit_params(**fit_params)
        Xt = self._fit(X, y, **fit_params_steps)
        # 从数组删除含有nan的行
        self.X_transformed = Xt[~np.isnan(Xt).any(axis=1)]

        if self._final_estimator != 'passthrough':
            fit_params_last_step = fit_params_steps[self.steps[-1][0]]
            self._final_estimator.fit(
                self.X_transformed, y, **fit_params_last_step)

        return self

    def predict(self, X=None, **predict_params):
        """Apply transforms to the data, and predict with the final estimator

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        **predict_params : dict of string -> object
            Parameters to the ``predict`` called at the end of all
            transformations in the pipeline. Note that while this may be
            used to return uncertainties from some models with return_std
            or return_cov, uncertainties that are generated by the
            transformations in the pipeline are not propagated to the
            final estimator.

            .. versionadded:: 0.20

        Returns
        -------
        y_pred : array-like
        """

        last_step = self._final_estimator
        fit_params_steps = self._check_fit_params(**predict_params)
        fit_params_last_step = fit_params_steps[self.steps[-1][0]]
        if hasattr(last_step, 'fit_predict'):
            return last_step.fit_predict(self.X_transformed, **fit_params_last_step)
        else:
            return last_step.predict(self.X_transformed, **predict_params)

    def get_score(self, X):
        """Caculate X's cluster score

        Parameters
        ----------
        X : df
            DataFrame

        Returns
        -------
        df_score : df
            DataFrame contain param and scores
        """
        all_score_list = []
        param_grid_list = list(ParameterGrid(self.param_grid))
        for param in param_grid_list:
            self.set_params(**param)
            self.fit(X)
            labels = self.predict()

            score_list = []
            # score_list.append(self.steps)
            score_list.append(param)
            for scorer in self.scoring:

                try:
                    if scorer == 'si':
                        cluster_score = silhouette_score(
                            self.X_transformed, labels)
                    elif scorer == 'ca':
                        cluster_score = calinski_harabasz_score(
                            self.X_transformed, labels)
                    elif scorer == 'da':
                        cluster_score = davies_bouldin_score(
                            self.X_transformed, labels)
                except:
                    cluster_score = np.nan

                score_list.append(cluster_score)

            #score_list.insert(0, self.named_steps)
            all_score_list.append(score_list)

        column_list = self.scoring.copy()
        column_list.insert(0, 'param')
        df_score = pd.DataFrame(all_score_list, columns=column_list)

        return df_score

    def get_best_score(self, data, n_best_score=3):
        """Return the best score DataFrame

        Parameters
        ----------
        data : df
            DataFrame of param and scores

        n_best_score : n best score for each scoring method

        Returns
        -------
        data_best_score : df
            DataFrame of the best score and param        
        """
        score_list = list(data.columns)
        score_list.remove('param')
        str_a = score_list.pop()
        if str_a == 'da':
            idx1 = data.sort_values(str_a, ascending=True).head(n_best_score).index
        else:
            idx1 = data.sort_values(str_a, ascending=False).head(n_best_score).index
        while score_list:
            str_a = score_list.pop()
            if str_a == 'da':
                idx2 = data.sort_values(str_a, ascending=True).head(n_best_score).index
            else:
                idx2 = data.sort_values(str_a, ascending=False).head(n_best_score).index
            idx1 = idx1.union(idx2)
        data_best_score = data.iloc[idx1]
        return data_best_score

    def plot(self, X, param_type='grid', ncol=1, figsize_x=10, scale_y=1):
        """Preprocessing transform X,then TSNE transform, plot scatterplot

        Parameters
        ----------
        X : data
            DataFrame

        param_type : string
            'grid'(default) for param_grid,'' for best_score param.

        ncol: int
            figure number in a row

        figsize_x : float
            sub-figure width
        scale_y : float
            scale height

        Returns
        -------
        return : None
            plot multiple figures

        """
        if param_type == 'grid':
            param_grid_list = list(ParameterGrid(self.param_grid))
        else:
            param_grid_list = self.param_grid

        nrow = math.ceil(len(param_grid_list)/ncol)
        figsize_y = nrow*figsize_x/ncol*scale_y
        fig, axes = plt.subplots(
            nrows=nrow, ncols=ncol, figsize=(figsize_x, figsize_y))

        for param, ax in zip(param_grid_list, axes.flat):

            self.set_params(**param)
            self.fit(X)
            labels = self.predict()
            tsne = TSNE(random_state=42)
            X_tsne = tsne.fit_transform(self.X_transformed)

            ax.set_title(str(param))
            sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=labels, size=labels, style=labels,
                            data=X_tsne, ax=ax)
        return
