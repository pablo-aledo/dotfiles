# Run application locally on 8 cores
spark-submit \
  --class org.apache.spark.examples.SparkPi \
  --master "local[8]" \
  /usr/local/spark/examples/jars/spark-examples_2.11-2.4.5.jar \
  100

# Run on a Spark standalone cluster in client deploy mode
spark-submit \
  --class org.apache.spark.examples.SparkPi \
  --master spark://207.184.161.138:7077 \
  --executor-memory 20G \
  --total-executor-cores 100 \
  /usr/local/spark/examples/jars/spark-examples_2.11-2.4.5.jar \
  1000

# Run on a Spark standalone cluster in cluster deploy mode with supervise
spark-submit \
  --class org.apache.spark.examples.SparkPi \
  --master spark://207.184.161.138:7077 \
  --deploy-mode cluster \
  --supervise \
  --executor-memory 20G \
  --total-executor-cores 100 \
  /usr/local/spark/examples/jars/spark-examples_2.11-2.4.5.jar \
  1000

# Run on a YARN cluster
export HADOOP_CONF_DIR=XXX
spark-submit \
  --class org.apache.spark.examples.SparkPi \
  --master yarn \
  --deploy-mode cluster \  # can be client for client mode
  --executor-memory 20G \
  --num-executors 50 \
  /usr/local/spark/examples/jars/spark-examples_2.11-2.4.5.jar \
  1000

# Run a Python application on a Spark standalone cluster
spark-submit \
  --master spark://207.184.161.138:7077 \
  examples/src/main/python/pi.py \
  1000

# Run on a Mesos cluster in cluster deploy mode with supervise
spark-submit \
  --class org.apache.spark.examples.SparkPi \
  --master mesos://207.184.161.138:7077 \
  --deploy-mode cluster \
  --supervise \
  --executor-memory 20G \
  --total-executor-cores 100 \
  http://path/to/examples.jar \
  1000

# Run on a Kubernetes cluster in cluster deploy mode
spark-submit \
  --class org.apache.spark.examples.SparkPi \
  --master k8s://xx.yy.zz.ww:443 \
  --deploy-mode cluster \
  --executor-memory 20G \
  --num-executors 50 \
  http://path/to/examples.jar \
  1000

# Run on a Kubernetes cluster in cluster deploy mode

k create serviceaccount spark
k policy add-role-to-user admin system:serviceaccount:project:spark
k create clusterrolebinding spark-role --clusterrole=edit --serviceaccount=project:spark --namespace=project

spark-submit \
  --class org.apache.spark.examples.SparkPi \
  --master k8s://https://xx.yy.zz.ww:443 \
  --deploy-mode cluster \
  --name sparkpi \
  --conf spark.executor.instances=1 \
  --conf spark.kubernetes.container.image=registry... \
  --conf spark.kubernetes.authenticate.driver.serviceAccountName=spark \
  --conf spark.kubernetes.authenticate.submission.oauthToken=$(kubectl sa get-token spark) \
  --conf spark.kubernetes.namespace=project \
  --conf spark.kubernetes.timeout=100 \
  --executor-memory 20G \
  --num-executors 50 \
  http://path/to/spark-examples_2.11-2.4.5.jar \
  1000

