import pandas as pd
import plotly.express as px
import streamlit as st
from prediction_main import Predictor  

def load_data():
    data_path = "Data/data.csv"
    try:
        df = pd.read_csv(data_path)
        return df
    except FileNotFoundError:
        st.error("Error: The dataset file 'data.csv' was not found. Please ensure the file exists in the correct path.")
        return None

def apply_filters(df):
    with st.sidebar.expander("Filter Options", expanded=False):
        filters = {}
        for col in df.columns:
            if df[col].dtype == 'object':
                filters[col] = st.multiselect(f"Select {col}", df[col].dropna().unique(), default=df[col].dropna().unique())
            else:
                min_val, max_val = float(df[col].min()), float(df[col].max())
                filters[col] = st.slider(f"Select range for {col}", min_val, max_val, (min_val, max_val))

    filtered_df = df.copy()
    for col, filter_val in filters.items():
        if isinstance(filter_val, list):
            filtered_df = filtered_df[filtered_df[col].isin(filter_val)]
        else:
            filtered_df = filtered_df[(filtered_df[col] >= filter_val[0]) & (filtered_df[col] <= filter_val[1])]
    return filtered_df

def show_data_overview(df):
    st.write("## Data Overview")

    selected_column = st.selectbox("Select a column to view details", df.columns, key="column_overview")

    if selected_column:
        st.write(f"### Details for Column: `{selected_column}`")

        col_type = df[selected_column].dtype
        st.write(f"**Data Type:** `{col_type}`")

        num_unique = df[selected_column].nunique()
        st.write(f"**Number of Unique Values:** `{num_unique}`")

        num_missing = df[selected_column].isnull().sum()
        st.write(f"**Number of Missing Values:** `{num_missing}`")

        if df[selected_column].dtype == 'object':
            unique_categories = df[selected_column].dropna().unique()
            st.write(f"**Unique Categories:**")
            st.write(unique_categories)

        if pd.api.types.is_numeric_dtype(df[selected_column]):
            st.write("**Summary Statistics:**")
            st.dataframe(df[selected_column].describe().to_frame())

    st.write("### Sample Data")
    st.dataframe(df.head())

def categorical_distribution_section(df):
    st.write("## Categorical Feature Distribution")
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    selected_cat_column = st.selectbox("Select a categorical column", categorical_columns, key="cat_dist")
    if selected_cat_column:
        fig = px.pie(df, names=selected_cat_column, title=f"Distribution of {selected_cat_column}",
                     color_discrete_sequence=px.colors.qualitative.Bold, hole=0.4)
        fig.update_traces(textinfo='percent+label', pull=[0.05]*len(df[selected_cat_column].unique()))
        st.plotly_chart(fig, use_container_width=True)

def drill_down_section(df):
    st.write("## Drill-Down Analysis")
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    selected_cat_column = st.selectbox("Select a categorical column for drill-down", categorical_columns, key="drill_cat")
    if selected_cat_column:
        drilldown_col = st.selectbox("Select a column to drill down into", [col for col in df.columns if col != selected_cat_column])
        if drilldown_col:
            if df[drilldown_col].dtype == 'object':
                fig = px.bar(df, x=drilldown_col, color=selected_cat_column,
                             title=f"Drill-Down: {drilldown_col} by {selected_cat_column}",
                             color_discrete_sequence=px.colors.qualitative.Dark24)
            else:
                fig = px.box(df, x=selected_cat_column, y=drilldown_col,
                             title=f"Drill-Down: {drilldown_col} by {selected_cat_column}",
                             color=selected_cat_column, color_discrete_sequence=px.colors.qualitative.Dark24)
            st.plotly_chart(fig, use_container_width=True)

def bar_chart_section(df):
    st.write("## Bar Chart for Categorical Distribution")
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    selected_bar_column = st.selectbox("Select a categorical column for bar chart", categorical_columns, key="bar_chart")
    if selected_bar_column:
        fig = px.bar(df, x=selected_bar_column, title=f"Bar Chart of {selected_bar_column}",
                     color=selected_bar_column, color_discrete_sequence=px.colors.qualitative.Vivid)
        st.plotly_chart(fig, use_container_width=True)

def histogram_section(df):
    st.write("## Histogram for Numerical Columns")
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    hist_column = st.selectbox("Select a numerical column for histogram", numeric_columns, key="hist_column")
    category_column = st.selectbox("Select a categorical column to stack", categorical_columns, key="hist_category")
    if hist_column and category_column:
        fig = px.histogram(df, x=hist_column, color=category_column, barmode='stack',
                           color_discrete_sequence=px.colors.qualitative.Prism)
        st.plotly_chart(fig, use_container_width=True)

def box_plot_section(df):
    st.write("## Box Plot for Numerical Columns")
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    box_column = st.selectbox("Select a numerical column for box plot", numeric_columns, key="box_plot")
    if box_column:
        fig = px.box(df, y=box_column, title=f"Box Plot of {box_column}",
                     color_discrete_sequence=px.colors.qualitative.Bold, points="all")
        st.plotly_chart(fig, use_container_width=True)

def prediction_section():
    st.write("## Predict New Data")
    uploaded_file = st.file_uploader("Upload a CSV file for prediction", type=["csv"])

    if uploaded_file is not None:
        st.write("File uploaded successfully!")
        input_data = pd.read_csv(uploaded_file)
        st.write("### Uploaded Data")
        st.dataframe(input_data.head())

        predictor = Predictor()
        if st.button("Predict"):
            st.write("### Predicting...")
            predictions = predictor.predict(input_data)
            st.write("### Here you go......")
            if predictions is not None:
                st.write("### Predictions")
                st.dataframe(predictions)
                csv = predictions.to_csv(index=False).encode('utf-8')
                st.download_button(label="Download Predictions as CSV", data=csv, file_name="predictions.csv", mime="text/csv")
            else:
                st.error("Prediction failed due to data validation issues.")

def main():
    st.set_page_config(page_title="E-Commerce Customer Data Dashboard", layout="wide")
    tab1, tab2 = st.tabs(["Data Dashboard", "Predictions"])

    with tab1:
        st.title("E-Commerce Customer Data Dashboard")
        df = load_data()
        if df is not None:
            filtered_df = apply_filters(df)
            show_data_overview(filtered_df)
            categorical_distribution_section(filtered_df)
            drill_down_section(filtered_df)
            bar_chart_section(filtered_df)
            histogram_section(filtered_df)
            box_plot_section(filtered_df)

    with tab2:
        st.title("Customer Churn Predictions")
        prediction_section()

    st.write("---")
    st.write("**Created by Sai Vardhan.M & Bhavana.U**")

if __name__ == "__main__":
    main()
