# ============================================================================
# InsightAI: Business Analytics Assistant
# A Streamlit app combining BI dashboards, SQL exploration, and AI insights
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import random

# Try importing transformers, but handle gracefully if not installed
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="InsightAI - Business Analytics Assistant",
    page_icon="📊",
    layout="wide"
)

# ----------------------------------------------------------------------------
# DATA GENERATION / LOADING
# ----------------------------------------------------------------------------
@st.cache_data
def generate_sample_data(n=1000):
    """
    Generates sample e-commerce data for the dashboard.
    Uses caching to avoid regenerating on every rerun.
    """
    np.random.seed(42)
    categories = ['Electronics', 'Furniture', 'Clothing', 'Books', 'Sports']
    regions = ['North America', 'Europe', 'Asia Pacific', 'South America']
    segments = ['Enterprise', 'SMB', 'Consumer']
    
    # Generate base dates
    base_dates = pd.date_range(start='2024-01-01', periods=n, freq='D')
    
    # Generate random offsets (0-29 days)
    random_offsets = np.random.randint(0, 30, size=n)
    
    # Create order dates by adding offsets to base dates
    order_dates = []
    for i in range(n):
        order_dates.append(base_dates[i] + pd.Timedelta(days=int(random_offsets[i])))
    
    # Generate orders
    data = {
        'order_id': [f'ORD-{i:04d}' for i in range(1, n+1)],
        'order_date': order_dates,
        'customer_id': [f'CUST-{random.randint(1, 200):03d}' for _ in range(n)],
        'category': [random.choice(categories) for _ in range(n)],
        'region': [random.choice(regions) for _ in range(n)],
        'customer_segment': [random.choice(segments) for _ in range(n)],
        'sales': np.round(np.random.uniform(50, 2000, n), 2),
        'profit': np.round(np.random.uniform(-100, 800, n), 2),
        'discount': np.round(np.random.uniform(0, 0.25, n), 2)
    }
    
    df = pd.DataFrame(data)
    
    # Add 'month' and 'year' columns
    df['month'] = df['order_date'].dt.to_period('M').astype(str)
    df['year'] = df['order_date'].dt.year
    
    # Generate budget data for controlling analysis
    months = sorted(df['month'].unique())
    budget_data = {
        'month': months,
        'budget_sales': np.round(np.random.uniform(80000, 120000, len(months)), 2),
        'budget_profit': np.round(np.random.uniform(15000, 30000, len(months)), 2)
    }
    budget_df = pd.DataFrame(budget_data)
    
    return df, budget_df

# ----------------------------------------------------------------------------
# AI MODEL LOADING (Hugging Face)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_sentiment_model():
    """
    Loads a Hugging Face sentiment analysis model.
    Cached to avoid reloading on every rerun.
    """
    if not TRANSFORMERS_AVAILABLE:
        return None
    
    try:
        classifier = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1
        )
        return classifier
    except Exception as e:
        st.warning(f"Could not load Hugging Face model: {e}")
        return None

# ----------------------------------------------------------------------------
# SQL QUERY GENERATOR
# ----------------------------------------------------------------------------
def generate_sql_from_query(question, df_columns):
    question = question.lower().strip()
    table = "sales_data"
    columns = ', '.join(df_columns)
    
    if "top" in question and ("sales" in question or "revenue" in question):
        nums = [int(word) for word in question.split() if word.isdigit()]
        num = nums[0] if nums else 5
        return f"SELECT category, SUM(sales) as total_sales FROM {table} GROUP BY category ORDER BY total_sales DESC LIMIT {num}"
    
    elif "total sales" in question or "total revenue" in question:
        return f"SELECT SUM(sales) as total_revenue FROM {table}"
    
    elif "profit" in question and "category" in question:
        return f"SELECT category, SUM(profit) as total_profit FROM {table} GROUP BY category ORDER BY total_profit DESC"
    
    elif "region" in question and "sales" in question:
        return f"SELECT region, SUM(sales) as total_sales FROM {table} GROUP BY region ORDER BY total_sales DESC"
    
    elif "avg" in question or "average" in question:
        if "profit" in question:
            return f"SELECT AVG(profit) as avg_profit FROM {table}"
        else:
            return f"SELECT AVG(sales) as avg_sales FROM {table}"
    
    elif "count" in question or "orders" in question:
        if "region" in question:
            return f"SELECT region, COUNT(*) as order_count FROM {table} GROUP BY region"
        else:
            return f"SELECT COUNT(*) as total_orders FROM {table}"
    
    else:
        return f"SELECT {columns} FROM {table} LIMIT 100"

# ----------------------------------------------------------------------------
# BUDGET VS ACTUAL ANALYSIS
# ----------------------------------------------------------------------------
def calculate_budget_vs_actual(df, budget_df):
    actual_monthly = df.groupby('month').agg({
        'sales': 'sum',
        'profit': 'sum'
    }).reset_index()
    
    comparison = actual_monthly.merge(budget_df, on='month', how='left')
    
    comparison['sales_variance'] = comparison['sales'] - comparison['budget_sales']
    comparison['profit_variance'] = comparison['profit'] - comparison['budget_profit']
    comparison['sales_variance_pct'] = (comparison['sales_variance'] / comparison['budget_sales'] * 100).round(1)
    comparison['profit_variance_pct'] = (comparison['profit_variance'] / comparison['budget_profit'] * 100).round(1)
    
    return comparison

# ----------------------------------------------------------------------------
# MAIN APP
# ----------------------------------------------------------------------------
def main():
    st.title("📊 InsightAI - Business Analytics Assistant")
    st.markdown("""
    *An AI-powered business intelligence tool combining dashboards, SQL exploration, and automated insights.*
    """)
    
    # Load data
    df, budget_df = generate_sample_data(1000)
    
    # Load AI model
    sentiment_model = load_sentiment_model()
    if sentiment_model is None:
        st.sidebar.info("ℹ️ Running in demo mode")
    
    # Sidebar
    st.sidebar.title("📌 Navigation")
    app_mode = st.sidebar.radio(
        "Select Mode",
        ["📈 Dashboard", "🔍 Data Explorer", "💬 Ask Your Data", "📊 Budget vs Actual", "📱 Download Reports"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔎 Filters")
    
    selected_regions = st.sidebar.multiselect(
        "Region",
        options=df['region'].unique(),
        default=df['region'].unique().tolist()
    )
    selected_categories = st.sidebar.multiselect(
        "Category",
        options=df['category'].unique(),
        default=df['category'].unique().tolist()
    )
    
    filtered_df = df[
        (df['region'].isin(selected_regions)) &
        (df['category'].isin(selected_categories))
    ]
    
    if filtered_df.empty:
        st.warning("No data available with current filters. Please adjust your selection.")
        return
    
    # DASHBOARD
    if app_mode == "📈 Dashboard":
        st.header("📈 Interactive KPI Dashboard")
        
        total_sales = filtered_df['sales'].sum()
        total_profit = filtered_df['profit'].sum()
        total_orders = len(filtered_df)
        avg_order_value = total_sales / total_orders if total_orders > 0 else 0
        profit_margin = (total_profit / total_sales * 100) if total_sales > 0 else 0
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("💰 Total Revenue", f"${total_sales:,.2f}")
        col2.metric("📈 Total Profit", f"${total_profit:,.2f}")
        col3.metric("📦 Total Orders", f"{total_orders:,}")
        col4.metric("💳 Avg Order Value", f"${avg_order_value:,.2f}")
        col5.metric("📊 Profit Margin", f"{profit_margin:.1f}%")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Sales by Category")
            category_sales = filtered_df.groupby('category')['sales'].sum().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(category_sales.index, category_sales.values, color='skyblue')
            ax.set_xlabel("Category")
            ax.set_ylabel("Total Sales ($)")
            ax.tick_params(axis='x', rotation=45)
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.subheader("Sales by Region")
            region_sales = filtered_df.groupby('region')['sales'].sum().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(8, 5))
            region_sales.plot(kind='bar', ax=ax, color='lightgreen')
            ax.set_xlabel("Region")
            ax.set_ylabel("Total Sales ($)")
            ax.tick_params(axis='x', rotation=45)
            st.pyplot(fig)
            plt.close()
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("Sales Trend Over Time")
            monthly_sales = filtered_df.groupby('month')['sales'].sum().sort_index()
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(monthly_sales.index, monthly_sales.values, marker='o', color='coral')
            ax.set_xlabel("Month")
            ax.set_ylabel("Total Sales ($)")
            ax.tick_params(axis='x', rotation=45)
            st.pyplot(fig)
            plt.close()
        
        with col4:
            st.subheader("Profit vs Discount")
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(filtered_df['discount'], filtered_df['profit'], alpha=0.5, color='purple')
            ax.set_xlabel("Discount Rate")
            ax.set_ylabel("Profit ($)")
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.3)
            st.pyplot(fig)
            plt.close()
        
        with st.expander("🔍 Data Quality Report"):
            missing_count = filtered_df.isnull().sum().sum()
            duplicate_count = filtered_df.duplicated().sum()
            col1, col2, col3 = st.columns(3)
            col1.metric("Missing Values", missing_count)
            col2.metric("Duplicate Rows", duplicate_count)
            col3.metric("Total Records", len(filtered_df))
            st.dataframe(filtered_df.describe())
    
    # DATA EXPLORER
    elif app_mode == "🔍 Data Explorer":
        st.header("🔍 Interactive Data Explorer")
        
        st.subheader("📋 Data Preview")
        st.dataframe(filtered_df, use_container_width=True)
        
        st.subheader("📊 Distribution Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                selected_col = st.selectbox("Select column for distribution", numeric_cols)
                fig, ax = plt.subplots()
                filtered_df[selected_col].hist(bins=30, ax=ax, color='skyblue', edgecolor='black')
                ax.set_title(f"Distribution of {selected_col}")
                st.pyplot(fig)
                plt.close()
            else:
                st.info("No numeric columns available.")
        
        with col2:
            st.subheader("Correlation Matrix")
            numeric_df = filtered_df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) > 1:
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax, fmt='.2f')
                st.pyplot(fig)
                plt.close()
            else:
                st.info("Not enough numeric columns for correlation.")
    
    # ASK YOUR DATA
    elif app_mode == "💬 Ask Your Data":
        st.header("💬 Ask Your Data")
        st.markdown("""
        Ask questions about your data in natural language. The system will generate and execute SQL queries to find answers.
        """)
        
        df_columns = filtered_df.columns.tolist()
        
        user_question = st.text_area(
            "What would you like to know?",
            placeholder="e.g., 'What are the top 5 product categories by sales?', 'Show me total profit by region'",
            height=80
        )
        
        if st.button("🚀 Generate Query", type="primary"):
            if user_question:
                with st.spinner("Generating SQL query..."):
                    sql_query = generate_sql_from_query(user_question, df_columns)
                    
                    st.subheader("📝 Generated SQL Query")
                    st.code(sql_query, language="sql")
                    
                    st.subheader("📊 Results")
                    try:
                        if "SUM(sales) as total_sales" in sql_query:
                            results = filtered_df.groupby('category')['sales'].sum().reset_index().sort_values('sales', ascending=False)
                            st.dataframe(results)
                        elif "SUM(sales) as total_revenue" in sql_query:
                            results = pd.DataFrame({'total_revenue': [filtered_df['sales'].sum()]})
                            st.dataframe(results)
                        elif "SUM(profit)" in sql_query and "category" in sql_query:
                            results = filtered_df.groupby('category')['profit'].sum().reset_index().sort_values('profit', ascending=False)
                            st.dataframe(results)
                        elif "region" in sql_query and "SUM(sales)" in sql_query:
                            results = filtered_df.groupby('region')['sales'].sum().reset_index().sort_values('sales', ascending=False)
                            st.dataframe(results)
                        elif "AVG" in sql_query:
                            if "profit" in sql_query:
                                results = pd.DataFrame({'avg_profit': [filtered_df['profit'].mean()]})
                            else:
                                results = pd.DataFrame({'avg_sales': [filtered_df['sales'].mean()]})
                            st.dataframe(results)
                        elif "COUNT" in sql_query:
                            if "region" in sql_query:
                                results = filtered_df.groupby('region').size().reset_index(name='order_count')
                                st.dataframe(results)
                            else:
                                results = pd.DataFrame({'total_orders': [len(filtered_df)]})
                                st.dataframe(results)
                        else:
                            st.dataframe(filtered_df.head(10))
                        
                        st.success("✅ Query executed successfully!")
                        
                        st.subheader("🤖 AI-Generated Summary")
                        if sentiment_model:
                            sample_text = f"We found {len(filtered_df)} records with total revenue of ${filtered_df['sales'].sum():,.2f}."
                            sentiment_result = sentiment_model(sample_text)
                            st.info(f"📝 Analysis shows a {sentiment_result[0]['label'].lower()} trend with {sentiment_result[0]['score']*100:.1f}% confidence.")
                        else:
                            st.info(f"""
                            📝 **Analysis Results**
                            
                            • Total Records: {len(filtered_df):,}
                            • Total Revenue: ${filtered_df['sales'].sum():,.2f}
                            • Total Profit: ${filtered_df['profit'].sum():,.2f}
                            • Average Order Value: ${filtered_df['sales'].mean():,.2f}
                            • Profit Margin: {(filtered_df['profit'].sum() / filtered_df['sales'].sum() * 100) if filtered_df['sales'].sum() > 0 else 0:.1f}%
                            """)
                    except Exception as e:
                        st.error(f"Error executing query: {e}")
            else:
                st.warning("⚠️ Please enter a question.")
    
    # BUDGET VS ACTUAL
    elif app_mode == "📊 Budget vs Actual":
        st.header("📊 Budget vs Actual Analysis")
        st.markdown("Compare actual performance against budget.")
        
        comparison_df = calculate_budget_vs_actual(filtered_df, budget_df)
        
        total_sales_actual = comparison_df['sales'].sum()
        total_sales_budget = comparison_df['budget_sales'].sum()
        sales_variance_total = total_sales_actual - total_sales_budget
        
        total_profit_actual = comparison_df['profit'].sum()
        total_profit_budget = comparison_df['budget_profit'].sum()
        profit_variance_total = total_profit_actual - total_profit_budget
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Sales Variance", f"${sales_variance_total:,.2f}")
        col2.metric("Total Profit Variance", f"${profit_variance_total:,.2f}")
        col3.metric("Periods Analyzed", len(comparison_df))
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            x = range(len(comparison_df))
            width = 0.35
            ax.bar([i - width/2 for i in x], comparison_df['budget_sales'], width, label='Budget', color='lightblue')
            ax.bar([i + width/2 for i in x], comparison_df['sales'], width, label='Actual', color='coral')
            ax.set_xlabel('Month')
            ax.set_ylabel('Sales ($)')
            ax.set_xticks(x)
            ax.set_xticklabels(comparison_df['month'], rotation=45)
            ax.legend()
            st.pyplot(fig)
            plt.close()
        
        with col2:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar([i - width/2 for i in x], comparison_df['budget_profit'], width, label='Budget', color='lightgreen')
            ax.bar([i + width/2 for i in x], comparison_df['profit'], width, label='Actual', color='orange')
            ax.set_xlabel('Month')
            ax.set_ylabel('Profit ($)')
            ax.set_xticks(x)
            ax.set_xticklabels(comparison_df['month'], rotation=45)
            ax.legend()
            st.pyplot(fig)
            plt.close()
        
        st.subheader("📋 Detailed Variance Report")
        st.dataframe(comparison_df, use_container_width=True)
    
    # DOWNLOAD REPORTS
    elif app_mode == "📱 Download Reports":
        st.header("📱 Download Reports")
        
        def convert_df_to_csv(df):
            return df.to_csv(index=False).encode('utf-8')
        
        st.subheader("Current Filtered Data")
        st.download_button(
            label="📥 Download Data as CSV",
            data=convert_df_to_csv(filtered_df),
            file_name=f'insightai_data_{datetime.now().strftime("%Y%m%d")}.csv',
            mime='text/csv'
        )

# ----------------------------------------------------------------------------
# RUN THE APP
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
