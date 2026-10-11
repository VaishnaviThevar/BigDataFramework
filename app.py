import streamlit as st
from pyspark.sql import SparkSession
import tempfile
import os

# Set page configuration
st.set_page_config(page_title="Big Data Framework Analytics Engine", layout="wide")

# Initialize Spark Session safely with caching
@st.cache_resource
def get_spark_session():
    return SparkSession.builder \
        .appName("BigDataFrameworkAnalytics") \
        .getOrCreate()

spark = get_spark_session()

st.title("Big Data Framework Analytics Engine (Apache Spark & PySpark)")
st.markdown("Demonstrating distributed big data processing and aggregation using the **Apache Spark Framework**.")

st.sidebar.header("Data Management")
uploaded_file = st.sidebar.file_uploader("Upload a dataset (CSV or Parquet)", type=["csv", "parquet", "pq"])

if uploaded_file is not None:
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    # Save to a temp file so Spark's JVM can read it from path
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        # Load dataset into PySpark DataFrame
        if file_extension == 'csv':
            df = spark.read.option("header", "true").option("inferSchema", "true").csv(tmp_path)
        else:
            df = spark.read.parquet(tmp_path)

        st.success("Dataset successfully loaded into PySpark!")

        # Analytics Overview Section
        st.subheader("Dataset Overview & Metrics")
        total_rows = df.count()
        total_columns = len(df.columns)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Rows", total_rows)
        with col2:
            st.metric("Total Columns", total_columns)

        # Data Preview
        st.subheader("Data Preview (First 10 Rows)")
        st.dataframe(df.limit(10).toPandas())

        # Schema Details
        st.subheader("Schema Definition")
        schema_info = [{"Column": field.name, "Type": field.dataType.simpleString()} for field in df.schema.fields]
        st.table(schema_info)

    except Exception as e:
        st.error(f"Error processing file with PySpark: {e}")
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

else:
    st.info("No dataset uploaded. Please upload a CSV or Parquet file from the sidebar to begin processing.")
