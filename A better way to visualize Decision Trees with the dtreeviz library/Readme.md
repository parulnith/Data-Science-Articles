The repository contains the code for the accompanying blogpost titled [A better way to visualize Decision Trees with the dtreeviz library](https://towardsdatascience.com/a-better-way-to-visualize-decision-trees-with-the-dtreeviz-library-758994cdf05e?sk=ad5fcdf665e07388a829bb5320be9a6f)




## Installation

```
#conda
conda uninstall python-graphviz
conda uninstall graphviz

#pip
pip install dtreeviz             # install dtreeviz for sklearn
pip install dtreeviz[xgboost]    # install XGBoost related dependency
pip install dtreeviz[pyspark]    # install pyspark related dependency
pip install dtreeviz[lightgbm]   # install LightGBM related dependency
This should also pull in the graphviz Python library (>=0.9), which we are using for platform specific stuff.

```
For details see: https://github.com/parrt/dtreeviz
