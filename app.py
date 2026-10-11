import streamlit as st
from pyspark.sql import SparkSession
import pandas as pd
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
st.markdown("Demonstrating distributed big data processing, schema analysis, and SQL querying using the **Apache Spark Framework**.")

st.sidebar.header("Data Management")
uploaded_file = st.sidebar.file_uploader("Upload a dataset (CSV or Parquet)", type=["csv", "parquet", "pq"])

if uploaded_file is not None:
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        if file_extension == 'csv':
            df = spark.read.option("header", "true").option("inferSchema", "true").csv(tmp_path)
        else:
            df = spark.read.parquet(tmp_path)

        # Register DataFrame as a Spark SQL table view
        table_name = "dataset_table"
        df.createOrReplaceTempView(table_name)

        st.success("Dataset loaded successfully and registered as a Spark SQL table (`dataset_table`)!")

        # --- Tabbed Navigation ---
        tab1, tab2, tab3 = st.tabs(["Overview & Schema", "Visual Analytics", "Spark SQL Console"])

        # Tab 1: Overview & Schema
        with tab1:
            st.subheader("Dataset Overview & Metrics")
            total_rows = df.count()
            total_columns = len(df.columns)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Rows", total_rows)
            with col2:
                st.metric("Total Columns", total_columns)

            st.subheader("Data Preview (First 10 Rows)")
            st.dataframe(df.limit(10).toPandas())

            st.subheader("Schema Definition")
            schema_info = [{"Column": field.name, "Type": field.dataType.simpleString()} for field in df.schema.fields]
            st.table(schema_info)

        # Tab 2: Visual Analytics
        with tab2:
            st.subheader("Interactive Visual Analytics")
            display_df = df.limit(10000).toPandas() if total_rows > 100000 else df.toPandas()
            
            if not display_df.empty:
                all_columns = list(display_df.columns)
                num_columns = [col for col, dtype in zip(all_columns, display_df.dtypes) if dtype != 'object']
                cat_columns = [col for col, dtype in zip(all_columns, display_df.dtypes) if dtype == 'object']

                col1, col2 = st.columns(2)
                with col1:
                    if cat_columns:
                        selected_cat = st.selectbox("Select Categorical Column", cat_columns, key="cat_sel")
                        spark_counts = df.groupBy(selected_cat).count().orderBy("count", ascending=False).limit(15)
                        st.write(f"**Distribution of {selected_cat}**")
                        st.bar_chart(spark_counts.toPandas().set_index(selected_cat))
                with col2:
                    if num_columns:
                        selected_num = st.selectbox("Select Numerical Column", num_columns, key="num_sel")
                        st.write(f"**Trend View of {selected_num}**")
                        st.line_chart(display_df[selected_num])

        # Tab 3: Spark SQL Query Console (New Feature!)
        with tab3:
            st.subheader("PySpark SQL Query Console")
            st.markdown(f"Write and execute standard SQL queries directly against your uploaded dataset using the table name: **`{table_name}`**")
            
            st.info(f"💡 **Tip:** Your table name is `dataset_table`. Example: `SELECT * FROM dataset_table LIMIT 5`")

            default_query = f"SELECT * FROM {table_name} LIMIT 10"
            query = st.text_area("Enter Spark SQL Query:", value=default_query, height=100)

            if st.button("Execute SQL Query"):
                try:
                    query_result_df = spark.sql(query)
                    result_pd = query_result_df.toPandas()
                    st.success("Query executed successfully!")
                    st.metric("Result Rows", len(result_pd))
                    st.dataframe(result_pd)
                except Exception as sql_err:
                    st.error(f"SQL Error: {sql_err}")

    except Exception as e:
        st.error(f"Error processing file with PySpark: {e}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

else:
    st.info("No dataset uploaded. Please upload a CSV or Parquet file from the sidebar to begin processing.")
