import streamlit as st
import pandas as pd
import numpy as np
import os

# PySpark imports (Big Data Distributed Processing Engine)
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, sum as _sum

st.set_page_config(page_title="Big Data Framework - PySpark Analytics Engine", page_icon="⚡", layout="wide")

st.title("⚡ Big Data Framework Analytics Engine (Apache Spark & PySpark)")
st.write("Demonstrating distributed big data processing and aggregation using the **Apache Spark Framework**.")

# 1. Initialize PySpark Local SparkSession Engine
@st.cache_resource
def get_spark_session():
    return SparkSession.builder \
        .appName("BigDataFrameworkProject") \
        .master("local[*]") \
        .getOrCreate()

spark = get_spark_session()

# 2. Ingest and create a Big Data Dataset
@st.cache_data
def generate_and_save_data():
    csv_file = "network_big_data.csv"
    if not os.path.exists(csv_file):
        np.random.seed(42)
        rows = 50000  # Large dataset representing big data log records
        df = pd.DataFrame({
            "Log_ID": range(1001, 1001 + rows),
            "Framework": "Apache Spark",
            "Protocol": np.random.choice(["TCP", "UDP", "HTTP", "HTTPS"], rows),
            "Data_Volume_KB": np.random.randint(100, 50000, rows),
            "Threat_Status": np.random.choice(["Normal", "Malicious", "Suspicious"], rows, p=[0.75, 0.15, 0.10]),
            "Cluster_Node": np.random.choice(["Node-01", "Node-02", "Node-03", "Node-04"], rows)
        })
        df.to_csv(csv_file, index=False)
    return csv_file

data_path = generate_and_save_data()

with st.spinner("Executing PySpark Distributed Framework Jobs..."):
    # Read raw CSV using PySpark Dataframe API
    spark_df = spark.read.csv(data_path, header=True, inferSchema=True)

st.success("PySpark Engine Session Active | Distributed RDD Computation Ready")

# 3. Big Data Framework Pipeline Aggregation
st.subheader("1. Apache Spark Distributed Processing Metrics")
total_records = spark_df.count()
total_volume_mb = spark_df.select(_sum("Data_Volume_KB")).collect()[0][0] / 1024

m1, m2, m3 = st.columns(3)
m1.metric("Big Data Records Processed", f"{total_records:,}")
m2.metric("Total Data Aggregated (MB)", f"{total_volume_mb:.2f} MB")
m3.metric("Spark Framework Engine", "PySpark 3.x")

st.markdown("---")

# 4. PySpark GroupBy Aggregations
st.subheader("2. Framework Query & Execution Performance")

# Grouping by Threat Status using PySpark Engine
threat_summary_spark = spark_df.groupBy("Threat_Status").agg(
    count("Log_ID").alias("Total_Count"),
    _sum("Data_Volume_KB").alias("Total_Volume_KB")
)
threat_pd = threat_summary_spark.toPandas()

col1, col2 = st.columns(2)

with col1:
    st.write("### Threat Counts (Calculated by PySpark)")
    st.bar_chart(threat_pd.set_index("Threat_Status")["Total_Count"])

with col2:
    st.write("### Node Volume Distribution")
    node_summary = spark_df.groupBy("Cluster_Node").agg(_sum("Data_Volume_KB").alias("Volume")).toPandas()
    st.bar_chart(node_summary.set_index("Cluster_Node")["Volume"])

st.markdown("---")
st.subheader("3. PySpark DataFrame Preview (Top 100 Rows)")
st.dataframe(spark_df.limit(100).toPandas(), use_container_width=True)