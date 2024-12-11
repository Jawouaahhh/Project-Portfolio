# Effacer le repertoire de sortie
hdfs dfs -rm -r output/projet_hadoop_lot1

# Lancement du Job MapReduce
hadoop jar $HADOOP_HOME/share/hadoop/tools/lib/hadoop-streaming-2.7.2.jar -file /root/mapper_lot1.py -mapper "python3 mapper_lot1.py" -file /root/reducer_lot1.py -reducer "python3 reducer_lot1.py" -input /user/root/input/dataw_fro03.csv -output /user/root/output/projet_hadoop_lot1
