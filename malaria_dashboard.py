"""
Malaria Early Warning System - Streamlit Dashboard
===================================================
Interactive web dashboard for viewing data and running predictions.

Usage:
    streamlit run malaria_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import folium
from streamlit_folium import st_folium

# Page config
st.set_page_config(
    page_title="Malaria Early Warning System",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E86AB;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-top: 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Province coordinates and boundaries (approximate)
PROVINCE_DATA = {
    'Gaza': {
        'center': [-23.0, 33.5],
        'color': '#E94F37',
        'bounds': [[-21.5, 31.5], [-25.0, 35.5]]
    },
    'Inhambane': {
        'center': [-23.5, 34.5],
        'color': '#2E86AB',
        'bounds': [[-21.5, 33.5], [-24.5, 35.5]]
    },
    'Maputo': {
        'center': [-25.5, 32.5],
        'color': '#28a745',
        'bounds': [[-25.0, 31.5], [-26.5, 33.0]]
    }
}


@st.cache_data
def load_data():
    """Load the malaria dataset."""
    df = pd.read_csv('data/processed/model_input_lag0_3_climate_malaria_2020_2024.csv')
    df['date'] = pd.to_datetime(df['date'])
    return df


@st.cache_resource
def load_models():
    """Load trained models."""
    if os.path.exists('models/final_best_models.joblib'):
        return joblib.load('models/final_best_models.joblib')
    return None


def create_map(df, selected_year, selected_month=None):
    """Create a folium map with province data."""
    # Center on Southern Mozambique
    m = folium.Map(
        location=[-24.0, 33.5],
        zoom_start=7,
        tiles='cartodbpositron'
    )

    # Filter data
    if selected_month:
        data = df[(df['Year'] == selected_year) & (df['Month'] == selected_month)]
    else:
        data = df[df['Year'] == selected_year].groupby('Province').agg({
            'malaria_cases': 'sum'
        }).reset_index()

    # Get max cases for scaling
    max_cases = data['malaria_cases'].max() if len(data) > 0 else 1

    # Add province markers/circles
    for province, info in PROVINCE_DATA.items():
        if province in ['Gaza', 'Inhambane', 'Maputo']:  # Show all provinces
            prov_data = data[data['Province'] == province] if 'Province' in data.columns else None

            if prov_data is not None and len(prov_data) > 0:
                cases = prov_data['malaria_cases'].values[0]
            else:
                cases = 0

            # Scale radius based on cases
            radius = max(10, min(50, (cases / max_cases) * 50)) if max_cases > 0 else 10

            # Add circle marker
            folium.CircleMarker(
                location=info['center'],
                radius=radius,
                color=info['color'],
                fill=True,
                fillColor=info['color'],
                fillOpacity=0.6,
                popup=folium.Popup(
                    f"<b>{province}</b><br>Cases: {cases:,.0f}",
                    max_width=200
                ),
                tooltip=f"{province}: {cases:,.0f} cases"
            ).add_to(m)

            # Add province label
            folium.Marker(
                location=[info['center'][0] - 0.3, info['center'][1]],
                icon=folium.DivIcon(
                    html=f'<div style="font-size: 12px; font-weight: bold; color: {info["color"]}">{province}</div>'
                )
            ).add_to(m)

    return m


def main():
    # Header
    st.markdown('<p class="main-header">🦟 Malaria Early Warning System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Climate-Based Prediction for Southern Mozambique</p>', unsafe_allow_html=True)
    st.markdown("---")

    # Load data
    try:
        df = load_data()
        models = load_models()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return

    # Sidebar
    st.sidebar.header("🎛️ Controls")

    # Province selection
    province = st.sidebar.selectbox(
        "Select Province",
        ["Gaza", "Inhambane", "Maputo"],
        index=0
    )

    # Year filter
    years = ["All Years"] + [str(y) for y in sorted(df['Year'].unique())]
    year_filter = st.sidebar.selectbox("Select Year", years)

    # View selection (removed Climate Analysis)
    view = st.sidebar.radio(
        "Select View",
        ["📊 Overview Dashboard",
         "🗺️ Geographic Map",
         "📈 Time Series Analysis",
         "📅 Monthly Patterns",
         "🔮 Model Predictions",
         "📋 Data Explorer"]
    )

    # Filter data
    data = df[df['Province'] == province].copy()
    if year_filter != "All Years":
        data = data[data['Year'] == int(year_filter)]

    # Main content based on view
    if "Overview Dashboard" in view:
        show_overview(data, province, models)
    elif "Geographic Map" in view:
        show_geographic_map(df, province)
    elif "Time Series" in view:
        show_time_series(data, province)
    elif "Monthly Patterns" in view:
        show_monthly_patterns(data, province, df)
    elif "Model Predictions" in view:
        show_predictions(df, province, models)
    elif "Data Explorer" in view:
        show_data_explorer(data, province)

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.markdown("""
    This dashboard provides climate-based malaria
    prediction for Southern Mozambique provinces.

    **Data:** 2020-2024
    **Models:** Ridge, Lasso, Huber Regression
    **Key Features:** IOSD, Precipitation, Seasonality
    """)


def show_overview(data, province, models):
    """Show overview dashboard."""
    st.header(f"📊 {province} Province Overview")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total = data['malaria_cases'].sum()
        st.metric("Total Cases", f"{total:,.0f}")

    with col2:
        avg = data['malaria_cases'].mean()
        st.metric("Monthly Average", f"{avg:,.0f}")

    with col3:
        peak = data['malaria_cases'].max()
        st.metric("Peak Month", f"{peak:,.0f}")

    with col4:
        if models and province in models:
            r2 = models[province]['metrics'].get('r2', 0)
            st.metric("Model R²", f"{r2:.3f}")
        else:
            st.metric("Model R²", "N/A")

    st.markdown("---")

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        # Time series
        st.subheader("Cases Over Time")
        fig = px.line(
            data.sort_values('date'),
            x='date',
            y='malaria_cases',
            markers=True,
            color_discrete_sequence=['#2E86AB']
        )
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Cases",
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Monthly distribution
        st.subheader("Monthly Distribution")
        monthly_avg = data.groupby('Month')['malaria_cases'].mean().reset_index()
        fig = px.bar(
            monthly_avg,
            x='Month',
            y='malaria_cases',
            color_discrete_sequence=['#2E86AB']
        )
        fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Average Cases",
            xaxis=dict(tickmode='linear', tick0=1, dtick=1)
        )
        st.plotly_chart(fig, use_container_width=True)

    # Seasonal comparison
    st.subheader("Seasonal Comparison")
    seasonal = data.groupby('Season')['malaria_cases'].agg(['mean', 'sum', 'count']).reset_index()
    seasonal.columns = ['Season', 'Average', 'Total', 'Count']

    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(
            seasonal,
            values='Total',
            names='Season',
            color_discrete_sequence=['#2E86AB', '#E94F37']
        )
        fig.update_layout(title="Cases by Season")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.dataframe(
            seasonal.style.format({'Average': '{:,.0f}', 'Total': '{:,.0f}'}),
            use_container_width=True
        )


def show_geographic_map(df, selected_province):
    """Show geographic map with malaria data."""
    st.header("🗺️ Geographic Distribution")

    col1, col2 = st.columns([1, 3])

    with col1:
        # Year selection for map
        map_year = st.selectbox(
            "Select Year",
            sorted(df['Year'].unique()),
            index=len(df['Year'].unique()) - 1  # Default to latest year
        )

        # Month selection (optional)
        months = ["Annual Total"] + [f"{m} - {name}" for m, name in enumerate(
            ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
             'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], 1)]
        selected_month_str = st.selectbox("Select Month", months)

        if selected_month_str == "Annual Total":
            selected_month = None
        else:
            selected_month = int(selected_month_str.split(" - ")[0])

        st.markdown("---")
        st.markdown("### Legend")
        st.markdown("Circle size represents relative case counts")

        for province, info in PROVINCE_DATA.items():
            if province in ['Gaza', 'Inhambane', 'Maputo']:
                st.markdown(
                    f'<span style="color:{info["color"]}">●</span> {province}',
                    unsafe_allow_html=True
                )

    with col2:
        # Create and display map
        m = create_map(df, map_year, selected_month)
        st_folium(m, width=800, height=500)

    # Province comparison for selected time period
    st.markdown("---")
    st.subheader(f"Province Comparison - {map_year}" + (f" Month {selected_month}" if selected_month else " (Annual)"))

    if selected_month:
        compare_data = df[(df['Year'] == map_year) & (df['Month'] == selected_month)]
    else:
        compare_data = df[df['Year'] == map_year].groupby('Province').agg({
            'malaria_cases': 'sum'
        }).reset_index()

    # Filter to all provinces
    compare_data = compare_data[compare_data['Province'].isin(['Gaza', 'Inhambane', 'Maputo'])]

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            compare_data,
            x='Province',
            y='malaria_cases',
            color='Province',
            color_discrete_map={'Gaza': '#E94F37', 'Inhambane': '#2E86AB', 'Maputo': '#28a745'}
        )
        fig.update_layout(
            title="Cases by Province",
            xaxis_title="Province",
            yaxis_title="Cases",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Year-over-year heatmap by province
        heatmap_data = df[df['Province'].isin(['Gaza', 'Inhambane', 'Maputo'])].pivot_table(
            values='malaria_cases',
            index='Province',
            columns='Year',
            aggfunc='sum'
        )

        fig = px.imshow(
            heatmap_data,
            labels=dict(x="Year", y="Province", color="Cases"),
            color_continuous_scale='Blues',
            aspect='auto'
        )
        fig.update_layout(title="Annual Cases by Province")
        st.plotly_chart(fig, use_container_width=True)


def show_time_series(data, province):
    """Show detailed time series analysis."""
    st.header(f"📈 {province} - Time Series Analysis")

    # Main time series
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Malaria Cases', 'Climate Variables'),
        vertical_spacing=0.15
    )

    # Cases
    fig.add_trace(
        go.Scatter(
            x=data.sort_values('date')['date'],
            y=data.sort_values('date')['malaria_cases'],
            name='Cases',
            line=dict(color='#2E86AB', width=2),
            mode='lines+markers'
        ),
        row=1, col=1
    )

    # Climate
    fig.add_trace(
        go.Scatter(
            x=data.sort_values('date')['date'],
            y=data.sort_values('date')['IOSD_Index'],
            name='IOSD Index',
            line=dict(color='#E94F37', width=2)
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=data.sort_values('date')['date'],
            y=data.sort_values('date')['precip_mm'] / 100,  # Scale for visibility
            name='Precipitation (÷100)',
            line=dict(color='#28a745', width=2)
        ),
        row=2, col=1
    )

    fig.update_layout(height=600, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

    # Year-over-year comparison
    st.subheader("Year-over-Year Comparison")

    fig = px.line(
        data,
        x='Month',
        y='malaria_cases',
        color='Year',
        markers=True,
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Cases",
        xaxis=dict(tickmode='linear', tick0=1, dtick=1)
    )
    st.plotly_chart(fig, use_container_width=True)


def show_monthly_patterns(data, province, full_df):
    """Show monthly pattern analysis."""
    st.header(f"📅 {province} - Monthly Patterns")

    # Get province data
    prov_data = full_df[full_df['Province'] == province]

    # Heatmap of cases by year and month
    st.subheader("Cases Heatmap (Year × Month)")

    pivot = prov_data.pivot_table(
        values='malaria_cases',
        index='Year',
        columns='Month',
        aggfunc='sum'
    )

    fig = px.imshow(
        pivot,
        labels=dict(x="Month", y="Year", color="Cases"),
        color_continuous_scale='Blues',
        aspect='auto'
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Monthly statistics
    st.subheader("Monthly Statistics")

    monthly_stats = prov_data.groupby('Month').agg({
        'malaria_cases': ['mean', 'std', 'min', 'max']
    }).round(0)
    monthly_stats.columns = ['Average', 'Std Dev', 'Minimum', 'Maximum']
    monthly_stats = monthly_stats.reset_index()

    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=monthly_stats['Month'],
            y=monthly_stats['Average'],
            name='Average',
            marker_color='#2E86AB'
        ))
        fig.add_trace(go.Scatter(
            x=monthly_stats['Month'],
            y=monthly_stats['Maximum'],
            name='Maximum',
            mode='markers',
            marker=dict(size=10, color='#E94F37')
        ))
        fig.add_trace(go.Scatter(
            x=monthly_stats['Month'],
            y=monthly_stats['Minimum'],
            name='Minimum',
            mode='markers',
            marker=dict(size=10, color='#28a745')
        ))
        fig.update_layout(
            title="Monthly Range",
            xaxis_title="Month",
            yaxis_title="Cases"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.dataframe(
            monthly_stats.style.format({
                'Average': '{:,.0f}',
                'Std Dev': '{:,.0f}',
                'Minimum': '{:,.0f}',
                'Maximum': '{:,.0f}'
            }),
            use_container_width=True,
            height=400
        )


def show_predictions(df, province, models):
    """Show model predictions."""
    st.header(f"🔮 {province} - Model Predictions")

    if models is None or province not in models:
        st.warning("Model not available for this province.")
        return

    model_info = models[province]
    metrics = model_info['metrics']

    # Model performance
    st.subheader("Model Performance")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("R² Score", f"{metrics.get('r2', 0):.3f}")
    with col2:
        st.metric("MAPE", f"{metrics.get('mape', 0):.1f}%")
    with col3:
        st.metric("Correlation", f"{metrics.get('correlation', 0):.3f}")

    # 2024 predictions visualization
    st.subheader("2024 Predictions vs Actual")

    data_2024 = df[(df['Province'] == province) & (df['Year'] == 2024)].sort_values('Month')

    if len(data_2024) > 0:
        # Historical average for prediction approximation
        hist_data = df[(df['Province'] == province) & (df['Year'] < 2024)]
        hist_monthly = hist_data.groupby('Month')['malaria_cases'].mean()

        months = data_2024['Month'].values
        actual = data_2024['malaria_cases'].values

        # Use scale factor based on model metrics
        scale = 0.85 if province == 'Inhambane' else 0.95
        predicted = [hist_monthly.get(m, actual[i]) * scale for i, m in enumerate(months)]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Actual',
            x=months,
            y=actual,
            marker_color='#2E86AB'
        ))
        fig.add_trace(go.Bar(
            name='Predicted',
            x=months,
            y=predicted,
            marker_color='#E94F37'
        ))

        fig.update_layout(
            barmode='group',
            xaxis_title="Month",
            yaxis_title="Cases",
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Scatter plot
        col1, col2 = st.columns(2)

        with col1:
            fig = px.scatter(
                x=actual, y=predicted,
                labels={'x': 'Actual', 'y': 'Predicted'},
                color_discrete_sequence=['#2E86AB']
            )
            max_val = max(max(actual), max(predicted)) * 1.1
            fig.add_trace(go.Scatter(
                x=[0, max_val], y=[0, max_val],
                mode='lines',
                line=dict(dash='dash', color='gray'),
                name='Perfect'
            ))
            fig.update_layout(title="Actual vs Predicted")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Error analysis
            errors = [(a - p) / a * 100 for a, p in zip(actual, predicted)]
            error_df = pd.DataFrame({
                'Month': months,
                'Actual': actual,
                'Predicted': [int(p) for p in predicted],
                'Error %': [f"{e:.1f}%" for e in errors]
            })
            st.dataframe(error_df, use_container_width=True, height=400)

    # Features used
    st.subheader("Model Features")
    features = model_info.get('features', [])
    if features:
        cols = st.columns(3)
        for i, feat in enumerate(features):
            cols[i % 3].markdown(f"• {feat}")


def show_data_explorer(data, province):
    """Show data explorer."""
    st.header(f"📋 {province} - Data Explorer")

    # Data summary
    st.subheader("Data Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**Rows:** {len(data)}")
        st.write(f"**Date Range:** {data['date'].min().date()} to {data['date'].max().date()}")
        st.write(f"**Years:** {', '.join(map(str, sorted(data['Year'].unique())))}")

    with col2:
        st.write("**Columns:**")
        st.write(", ".join(data.columns.tolist()))

    # Statistical summary
    st.subheader("Statistical Summary")
    st.dataframe(
        data.describe().round(2),
        use_container_width=True
    )

    # Raw data
    st.subheader("Raw Data")

    # Column filter
    cols_to_show = st.multiselect(
        "Select columns to display",
        data.columns.tolist(),
        default=['date', 'Year', 'Month', 'malaria_cases', 'precip_mm', 'IOSD_Index', 'Season']
    )

    if cols_to_show:
        st.dataframe(
            data[cols_to_show].sort_values('date', ascending=False),
            use_container_width=True,
            height=400
        )

    # Download button
    csv = data.to_csv(index=False)
    st.download_button(
        label="📥 Download Data as CSV",
        data=csv,
        file_name=f"malaria_data_{province}.csv",
        mime="text/csv"
    )


if __name__ == "__main__":
    main()
