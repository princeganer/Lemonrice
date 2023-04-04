# Project **Lemonrice** *code explained*

------

## Spark Session

```python
# create a sparksession
spark = SparkSession.builder.master('local[4]').appName('Lemonrice').getOrCreate()
```

SparkSession was introduced in version Spark 2.0, It is an entry point  to underlying Spark functionality in order to programmatically create  Spark RDD, DataFrame, and DataSet. SparkSession’s object *`spark`* is the default variable available in `spark-shell` and it can be created programmatically using `SparkSession` builder pattern.

`SparkSession.builder()` – Return `SparkSession.Builder` class. This is a builder for `SparkSession`. master(), appName() and getOrCreate() are methods of [SparkSession.Builder](https://spark.apache.org/docs/latest/api/java/org/apache/spark/sql/SparkSession.Builder.html).

`master()` – If you are running it on the cluster you need to use your master name  as an argument to master(). usually, it would be either [`yarn` ](https://sparkbyexamples.com/hadoop/how-yarn-works/)or `mesos` depends on your cluster setup.

- Use `local[x]` when running in Standalone mode. x should be an integer value and  should be greater than 0; this represents how many partitions it should  create when using RDD, DataFrame, and Dataset. Ideally, x value should  be the number of CPU cores you have.
- For standalone use `spark://master:7077`

`appName()` – Sets a name to the Spark application that shows in the [Spark web UI](https://sparkbyexamples.com/spark/spark-web-ui-understanding/). If no application name is set, it sets a random name.

`getOrCreate()` – This returns a SparkSession object if already exists. Creates a new one if not exist.

To create SparkSession in Scala or Python, you need to use the builder pattern method `builder()` and calling `getOrCreate()` method. If SparkSession already exists it returns otherwise creates a new SparkSession.

------

## Spark Config

```python
# display function for spark dataframe
spark.conf.set('spark.sql.repl.eagerEval.enabled', True)
```

This setting makes the output more like pandas and less like  command-line SQL. After this, you no longer need to specify show() to  see the output. 

------

## Spark Hyper-parameter tuning

### CrossValidator

*class* `pyspark.ml.tuning.CrossValidator`(*, *estimator: Optional[pyspark.ml.base.Estimator] = None*, *estimatorParamMaps: Optional[List[ParamMap]] = None*, *evaluator: Optional[[pyspark.ml.evaluation.Evaluator](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.evaluation.Evaluator.html#pyspark.ml.evaluation.Evaluator)] = None*, *numFolds: int = 3*, *seed: Optional[int] = None*, *parallelism: int = 1*, *collectSubModels: bool = False*, *foldCol: str = ''*)                                        [[source\]](https://spark.apache.org/docs/latest/api/python/_modules/pyspark/ml/tuning.html#CrossValidator)

K-fold cross validation performs model selection by splitting the dataset into a set of non-overlapping randomly partitioned folds which are used as separate training and test datasets e.g., with k=3 folds, K-fold cross validation will generate 3 (training, test) dataset pairs, each of which uses 2/3 of the data for training and 1/3 for testing. Each fold is used as the test set exactly once.

### Param Builder

*class* `pyspark.ml.tuning.``ParamGridBuilder`[[source\]](https://spark.apache.org/docs/latest/api/python/_modules/pyspark/ml/tuning.html#ParamGridBuilder)

Builder for a param grid used in grid search-based model selection.

`addGrid`(*param: [pyspark.ml.param.Param](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.param.Param.html#pyspark.ml.param.Param)[Any]*, *values: List[Any]*) → [pyspark.ml.tuning.ParamGridBuilder](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.tuning.ParamGridBuilder.html#pyspark.ml.tuning.ParamGridBuilder)[[source\]](https://spark.apache.org/docs/latest/api/python/_modules/pyspark/ml/tuning.html#ParamGridBuilder.addGrid)

Sets the given parameters in this grid to fixed values.

param must be an instance of Param associated with an instance of Params (such as Estimator or Transformer).



