import streamlit as st
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import plotly.express as px
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
st.markdown("Demonstrating distributed big data processing and aggregation using the **Apache Spark Framework**.")

# Custom CSS to improve layout
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #d1d1d1;
    }
    .stPlotlyChart {
        margin-top: 20px;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

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

        st.success("Dataset successfully loaded into PySpark!")

        # --- Tabbed Navigation ---
        tab1, tab2 = st.tabs(["Overview & Schema", "Visual Analytics"])

        # Tab 1: Original View (Metrics and Schema)
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

        # Tab 2: Visual Analytics (New Feature)
        with tab2:
            st.subheader("Interactive Visual Analytics")
            
            if total_rows > 100000:
                st.warning(f"Large dataset detected ({total_rows} rows). Visualizations will be based on a random sample of 10,000 rows for performance.")
                display_df = df.sample(fraction=0.1, seed=42).limit(10000).toPandas()
            else:
                st.info(f"Processing entire dataset ({total_rows} rows).")
                display_df = df.toPandas() # Convert entire small-to-medium DF for Plotly

            if display_df.empty:
                st.error("Dataset appears to be empty or failed sampling.")
            else:
                all_columns = list(display_df.columns)
                num_columns = [col for col, dtype in zip(all_columns, display_df.dtypes) if dtype != 'object' and 'date' not in dtype.name]
                cat_columns = [col for col, dtype in zip(all_columns, display_df.dtypes) if dtype == 'object' or 'date' in dtype.name]

                col1, col2 = st.columns(2)

                # 1. Bar Chart: Categorical Distribution (like Shivam's example)
                with col1:
                    if cat_columns:
                        selected_cat = st.selectbox("Select Categorical Column", cat_columns, key="cat_sel")
                        
                        # Use PySpark to count values, then convert to small Pandas DF for Plotly
                        spark_counts = df.groupBy(selected_cat).count().orderBy(F.col("count").desc()).limit(15)
                        cat_data = spark_counts.toPandas()

                        fig_bar = px.bar(
                            cat_data, 
                            x=selected_cat, 
                            y='count', 
                            title=f"Distribution of {selected_cat} (Top 15)",
                            labels={selected_cat: selected_cat, 'count': 'Record Count'}
                        )
                        fig_bar.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig_bar, use_container_width=True)
                    else:
                        st.write("No categorical columns available for a bar chart.")

                # 2. Histogram: Numerical Distribution (like Shivam's example)
                with col2:
                    if num_columns:
                        selected_num = st.selectbox("Select Numerical Column", num_columns, key="num_sel")
                        fig_hist = px.histogram(
                            display_df, 
                            x=selected_num, 
                            nbins=30, 
                            title=f"Histogram distribution of {selected_num}"
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)
                    else:
                        st.write("No numerical columns available for a histogram.")

                # 3. Correlation Heatmap (More complex big data overview)
                if len(num_columns) > 1:
                    st.subheader("Numerical Correlation Heatmap")
                    correlation_matrix = display_df[num_columns].corr()
                    fig_heatmap = px.imshow(
                        correlation_matrix,
                        text_auto=True,
                        aspect="auto",
                        title="Numerical Feature Correlation Matrix"
                    )
                    st.plotly_chart(fig_heatmap, use_container_width=True)

    except Exception as e:
        st.error(f"Error processing file with PySpark: {e}")
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

else:
    st.info("No dataset uploaded. Please upload a CSV or Parquet file from the sidebar to begin processing.")
