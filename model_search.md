# Model test bed

```python
from run_models import fit_and_predict
from evaluate import full_evaluation
```


## Instantaneous models

### Linear regression
- insensitive to scaling or PCA

```python
pipe = make_pipeline(LinearRegression())
name = get_pipeline_name(pipe)
```

### Polynomial regression
- Only a slight improvement
    - because non-linearities are localised?

```python
pipe = make_pipeline(PolynomialFeatures(2), LinearRegression())
name = get_pipeline_name(pipe, "poly2")
```

```python
pipe = make_pipeline(PolynomialFeatures(5), LinearRegression())
name = get_pipeline_name(pipe, "poly5")
```

### SGD
- very sensitive to scaling. Not sensitive to PCA

```python
pipe = make_pipeline(StandardScaler(), SGDRegressor())
name = get_pipeline_name(pipe)
```

### Support Vector Machines
- Sensitive to scaling, not to PCA

```python
pipe = make_pipeline(SVR())
name = get_pipeline_name(pipe)
test_pipeline(pipe)
```

```python
pipe = make_pipeline(StandardScaler(), SVR())
name = get_pipeline_name(pipe)
```

```python
pipe = make_pipeline(StandardScaler(), PCA(), SVR())
name = get_pipeline_name(pipe)
test_pipeline(pipe)
```

```python
pipe = make_pipeline(StandardScaler(), SVR(kernel="poly"))
name = get_pipeline_name(pipe, "polykernel"))
```


### Multilayer Perceptron

```python
pipe = make_pipeline(MultilayerPerceptronRegressor())
name = get_pipeline_name(pipe)
```

```python
pipe = make_pipeline(StandardScaler(), MultilayerPerceptronRegressor())
name = get_pipeline_name(pipe)
```

```python
pipe = make_pipeline(PCA(), MultilayerPerceptronRegressor())
name = get_pipeline_name(pipe)
```

```python
pipe = make_pipeline(StandardScaler(), PCA(), MultilayerPerceptronRegressor())
name = get_pipeline_name(pipe)
```

```python
pipe = make_pipeline(StandardScaler(), MultilayerPerceptronRegressor(activation="logistic"))
name = get_pipeline_name(pipe, "logisitic"))
```

```python
pipe = make_pipeline(StandardScaler(), MultilayerPerceptronRegressor(hidden_layer_sizes=(20,20,20,)))
name = get_pipeline_name(pipe, "[20,20,20]"))
```

```python
pipe = make_pipeline(StandardScaler(), MultilayerPerceptronRegressor(hidden_layer_sizes=(10,10,)))
name = get_pipeline_name(pipe, "[10,10]"))
```

```python
pipe = make_pipeline(StandardScaler(), MultilayerPerceptronRegressor(hidden_layer_sizes=(10,30,)))
name = get_pipeline_name(pipe, "[10,30]"))
```

```python
pipe = make_pipeline(StandardScaler(), MultilayerPerceptronRegressor(hidden_layer_sizes=(20,20,)))
name = get_pipeline_name(pipe, "[20,20]"))
```


### K-nearest neighbours
- Not sensitive to scaling or PCA

```python
pipe = make_pipeline(KNeighborsRegressor())
name = get_pipeline_name(pipe)
```

```python
pipe = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors = 1000))
name = get_pipeline_name(pipe, "1000 neighbours")
```


### Decision Trees

```python
pipe = make_pipeline(DecisionTreeRegressor())
name = get_pipeline_name(pipe)
```

```python
pipe = make_pipeline(ExtraTreesRegressor())
name = get_pipeline_name(pipe)
```

```python
pipe = make_pipeline(StandardScaler(), PCA(), ExtraTreesRegressor())
name = get_pipeline_name(pipe)
```


### Lagged linear regression

- need to make a proper wrapper for this.

```python
pipe = make_pipeline(LagFeatures(), LinearRegression())
name = name=get_pipeline_name(pipe, "lag1"))
```


