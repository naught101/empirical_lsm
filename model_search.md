# Model test bed

## Instantaneous models

### Linear regression
- insensitive to scaling or PCA

```{python}
pipe = make_pipeline(LinearRegression())
```

### Polynomial regression
- Only a slight improvement
    - because non-linearities are localised?

```{python}
pipe = make_pipeline(PolynomialFeatures(2), LinearRegression())
name = get_pipeline_name(pipe, "poly2")
```

```{python}
pipe = make_pipeline(PolynomialFeatures(5), LinearRegression())
name = get_pipeline_name(pipe, "poly5")
```

### SGD
- very sensitive to scaling. Not sensitive to PCA

```{python}
pipe = make_pipeline(StandardScaler(), SGDRegressor())
```

### Support Vector Machines
- Sensitive to scaling, not to PCA

```{python}
pipe = make_pipeline(SVR())
test_pipeline(pipe
```

```{python}
pipe = make_pipeline(StandardScaler(), SVR())
```

```python
pipe = make_pipeline(StandardScaler(), PCA(), SVR())
test_pipeline(pipe
```

```{python}
pipe = make_pipeline(StandardScaler(), SVR(kernel="poly"))
name = get_pipeline_name(pipe, "polykernel"))
```


### Multilayer Perceptron

```{python}
pipe = make_pipeline(MultilayerPerceptronRegressor())
```

```{python}
pipe = make_pipeline(StandardScaler(), MultilayerPerceptronRegressor())
```

```{python}
pipe = make_pipeline(PCA(), MultilayerPerceptronRegressor())
```

```{python}
pipe = make_pipeline(StandardScaler(), PCA(), MultilayerPerceptronRegressor())
```

```{python}
pipe = make_pipeline(StandardScaler(), MultilayerPerceptronRegressor(activation="logistic"))
name = get_pipeline_name(pipe, "logisitic"))
```

```{python}
pipe = make_pipeline(StandardScaler(), MultilayerPerceptronRegressor(hidden_layer_sizes=(20,20,20,)))
name = get_pipeline_name(pipe, "[20,20,20]"))
```

```{python}
pipe = make_pipeline(StandardScaler(), MultilayerPerceptronRegressor(hidden_layer_sizes=(10,10,)))
name = get_pipeline_name(pipe, "[10,10]"))
```

```{python}
pipe = make_pipeline(StandardScaler(), MultilayerPerceptronRegressor(hidden_layer_sizes=(10,30,)))
name = get_pipeline_name(pipe, "[10,30]"))
```

```{python}
pipe = make_pipeline(StandardScaler(), MultilayerPerceptronRegressor(hidden_layer_sizes=(20,20,)))
name = get_pipeline_name(pipe, "[20,20]"))
```


### K-nearest neighbours
- Not sensitive to scaling or PCA

```{python}
pipe = make_pipeline(KNeighborsRegressor())
```

```{python}
pipe = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors = 1000))
name = get_pipeline_name(pipe, "1000 neighbours")
```


### Decision Trees

```{python}
pipe = make_pipeline(DecisionTreeRegressor())
```

```{python}
pipe = make_pipeline(ExtraTreesRegressor())
```

```{python}
pipe = make_pipeline(StandardScaler(), PCA(), ExtraTreesRegressor())
```


### Lagged linear regression

- need to make a proper wrapper for this.

```{python}
pipe = make_pipeline(LagFeatures(), LinearRegression())
name = name=get_pipeline_name(pipe, "lag1"))
```


