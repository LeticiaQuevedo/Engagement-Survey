import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

def identify_numeric_columns(df):
    """Identify columns that contain numeric survey responses (1-5 scale)"""
    numeric_cols = []
    
    # Colunas que definitivamente não são numéricas
    exclude_cols = [
        'Timestamp', 
        'Department', 
        'How would you describe the current company culture?',
        'What are some things you want to share with us? Feel free to share, but not limited to: what\'s working well and we should continue to do, what we\'re not doing so great and should aim to change.'
    ]
    
    for col in df.columns:
        if col not in exclude_cols:
            try:
                # Tentar converter para numérico
                numeric_values = pd.to_numeric(df[col], errors='coerce')
                
                # Verificar se tem valores válidos
                valid_values = numeric_values.dropna()
                
                if len(valid_values) > 0:
                    # Verificar se está na faixa esperada (1-5)
                    if valid_values.min() >= 1 and valid_values.max() <= 5:
                        numeric_cols.append(col)
            except:
                continue
    
    return numeric_cols

# Page configuration
st.set_page_config(
    page_title="Employee Engagement Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .warning-card {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
    .success-card {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess the survey data"""
    try:
        # Load data from CSV file
        df = pd.read_csv('engagement_survey.csv')
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Define question mappings for easier analysis
        question_mapping = {
            'I am proud to work at AP': 'Pride',
            'I would recommend AP to my friends or colleagues as a great place to work': 'Recommendation',
            'I rarely think about looking for a job at another company': 'Retention_Intent',
            'I see myself still working here in two years\' time': 'Future_Commitment',
            'AP motivates me to go beyond what I would in a similar role elsewhere': 'Motivation',
            'Communication from leadership is frequent and transparent': 'Leadership_Communication',
            'The leaders in my organization provide clear direction and guidelines for the future': 'Leadership_Direction',
            'I receive regular, constructive feedback from my manager': 'Manager_Feedback',
            'I feel supported and comfortable approaching my manager with concerns and constructive feedback': 'Manager_Support',
            'I am satisfied with my job overall': 'Job_Satisfaction',
            'My workload is manageable and realistic while continuing to challenge and stimulate me': 'Workload_Balance',
            'I often feel like I\'m just checking tasks as done instead of creating meaningful impact (reverse scored)': 'Meaningful_Impact',
            'I feel comfortable with our unlimited PTO policy and taking time off when I need it': 'PTO_Comfort',
            'I have access to the things I need to do my job well': 'Resource_Access',
            'Most of the systems and processes here support us getting our work done effectively': 'Process_Effectiveness',
            'I feel empowered to bring new ideas to the team': 'Innovation_Empowerment',
            'I can voice a contrary opinion without fear of negative consequences': 'Psychological_Safety',
            'I know what I need to do to be successful in my role': 'Role_Clarity',
            'I understand how my work contributes to company goals': 'Purpose_Alignment',
            'Quality and improvement are top priorities when making decisions': 'Quality_Focus',
            'When I need to work with other areas (EPD, GTM, Ops, Support), the collaboration is smooth and productive': 'Cross_Collaboration',
            'I receive appropriate recognition when I do good work': 'Recognition',
            'I am satisfied with the current performance review process and career progression': 'Career_Progression',
            'I have access to the learning and development I need to do my job well': 'Learning_Development',
            'My manager (or someone in management) has shown a genuine interest in my career aspirations': 'Career_Interest',
            'I feel that my voice and opinions are heard and valued by my leadership and peers': 'Voice_Value',
            'I feel psychologically safe to take risks, make mistakes and learn from them': 'Risk_Safety',
            'AP is dedicated to diversity and inclusiveness': 'Diversity_Inclusion',
            'AP demonstrates a commitment to the wellbeing of employees': 'Employee_Wellbeing',
            'I understand the AP values': 'Values_Understanding',
            'I am satisfied with my current compensation': 'Compensation_Satisfaction',
            'I understand our benefits policy': 'Benefits_Understanding',
            'I am satisfied with my current benefits package': 'Benefits_Satisfaction',
            'Which of these best describes your primarly area?': 'Department'
        }
        
        # Rename columns for easier handling
        for old_name, new_name in question_mapping.items():
            if old_name in df.columns:
                df[new_name] = df[old_name]
        
        return df, question_mapping
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None

def calculate_enps(scores):
    """Calculate eNPS from recommendation scores"""
    if len(scores) == 0:
        return 0, 0, 0, 0
    
    promoters = len(scores[scores >= 4])
    passives = len(scores[scores == 3])
    detractors = len(scores[scores <= 2])
    total = len(scores)
    
    enps = ((promoters - detractors) / total) * 100
    
    return enps, promoters, passives, detractors

def create_dimensional_scores(df, question_mapping):
    """Create dimensional scores for different categories"""
    dimensions = {
        'Commitment & Belonging': ['Pride', 'Recommendation', 'Retention_Intent', 'Future_Commitment', 'Motivation'],
        'Leadership': ['Leadership_Communication', 'Leadership_Direction', 'Manager_Feedback', 'Manager_Support'],
        'Professional Satisfaction': ['Job_Satisfaction', 'Workload_Balance', 'Meaningful_Impact'],
        'Resources & Processes': ['Resource_Access', 'Process_Effectiveness', 'PTO_Comfort'],
        'Innovation & Voice': ['Innovation_Empowerment', 'Psychological_Safety', 'Voice_Value', 'Risk_Safety'],
        'Role & Purpose': ['Role_Clarity', 'Purpose_Alignment', 'Quality_Focus'],
        'Collaboration': ['Cross_Collaboration'],
        'Recognition & Development': ['Recognition', 'Career_Progression', 'Learning_Development', 'Career_Interest'],
        'Culture & Values': ['Diversity_Inclusion', 'Employee_Wellbeing', 'Values_Understanding'],
        'Compensation & Benefits': ['Compensation_Satisfaction', 'Benefits_Understanding', 'Benefits_Satisfaction']
    }
    
    dimension_scores = {}
    for dimension, questions in dimensions.items():
        available_questions = [q for q in questions if q in df.columns]
        if available_questions:
            # Handle reverse scored questions
            scores = df[available_questions].copy()
            if 'Meaningful_Impact' in available_questions:
                scores['Meaningful_Impact'] = 6 - scores['Meaningful_Impact']  # Reverse score
            dimension_scores[dimension] = scores.mean(axis=1)
    
    return pd.DataFrame(dimension_scores), dimensions

def main():
    st.title("🎯 Employee Engagement Dashboard")
    st.markdown("**AP - Comprehensive Survey Analysis**")
    
    # Load data
    df, question_mapping = load_data()
    
    if df is None:
        st.error("Unable to load data. Please check the file path and format.")
        return
    
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    
    # Department filter
    departments = ['All'] + sorted(df['Department'].unique().tolist())
    selected_dept = st.sidebar.selectbox("Select Department", departments)
    
    # Filter data
    if selected_dept != 'All':
        filtered_df = df[df['Department'] == selected_dept].copy()
        dept_title = f" - {selected_dept} Department"
    else:
        filtered_df = df.copy()
        dept_title = " - All Departments"
    
    # Create dimensional scores
    dim_scores, dimensions = create_dimensional_scores(filtered_df, question_mapping)
    
    # Main dashboard tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Executive Summary", 
        "🎯 eNPS Analysis", 
        "📈 Dimensional Analysis", 
        "🔍 Deep Dive", 
        "👥 Employee Personas", 
        "📋 Action Plans",
        "🔬 Methodology & Data Insights",
    ])
    
    with tab1:
        st.header(f"Executive Summary{dept_title}")
        
        col1, col2, col3, col4 = st.columns(4)
        
        # Key metrics
        if 'Recommendation' in filtered_df.columns:
            enps, promoters, passives, detractors = calculate_enps(filtered_df['Recommendation'].dropna())
            
            with col1:
                st.metric("eNPS Score", f"{enps:.1f}", delta=None)
            
            with col2:
                st.metric("#️⃣ Responses", f"{len(filtered_df)}", delta=None)
            
            with col3:
                overall_satisfaction = filtered_df['Job_Satisfaction'].mean() if 'Job_Satisfaction' in filtered_df.columns else 0
                st.metric("Avg Job Satisfaction", f"{overall_satisfaction:.2f}/5", delta=None)
            
            with col4:
                retention_intent = filtered_df['Retention_Intent'].mean() if 'Retention_Intent' in filtered_df.columns else 0
                st.metric("Avg Retention Intent", f"{retention_intent:.2f}/5", delta=None)
        
        # Commitment & Belonging heatmap
        st.subheader("📊 Dimensional Scores Overview")
        
        if not dim_scores.empty:
            fig = go.Figure(data=go.Heatmap(
                z=[dim_scores.mean().values],
                x=dim_scores.columns,
                y=['Average Score'],
                colorscale='RdYlGn',
                zmin=1,
                zmax=5,
                text=[[f"{val:.2f}" for val in dim_scores.mean().values]],
                texttemplate="%{text}",
                textfont={"size": 12},
                hoverongaps=False
            ))
            
            fig.update_layout(
                title="Dimensional Scores Heatmap",
                xaxis_title="Dimensions",
                height=300,
                xaxis={'tickangle': 45}
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Top strengths and areas for improvement
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏆 Top Strengths")
            if not dim_scores.empty:
                top_scores = dim_scores.mean().sort_values(ascending=False).head(5)
                for dimension, score in top_scores.items():
                    st.markdown(f"**{dimension}**: {score:.2f}/5")
        
        with col2:
            st.subheader("⚠️ Areas for Improvement")
            if not dim_scores.empty:
                bottom_scores = dim_scores.mean().sort_values(ascending=True).head(5)
                for dimension, score in bottom_scores.items():
                    st.markdown(f"**{dimension}**: {score:.2f}/5")

    with tab2:
        st.header(f"🎯 eNPS Analysis{dept_title}")
        
        if 'Recommendation' in filtered_df.columns:
            # Overall eNPS
            enps, promoters, passives, detractors = calculate_enps(filtered_df['Recommendation'].dropna())
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("""
                <style>
                    .metric-card {
                        background-color: #2e3440;
                        color: #eceff4;
                        padding: 1.5rem;
                        border-radius: 0.5rem;
                        border-left: 4px solid #5e81ac;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    }
                    .metric-card h3 {
                        color: #88c999;
                        margin-top: 0;
                        margin-bottom: 1rem;
                        font-size: 1.5rem;
                    }
                    .metric-card p {
                        color: #d8dee9;
                        margin: 0.5rem 0;
                        font-size: 1rem;
                    }
                    .warning-card {
                        background-color: #3c2e1e;
                        color: #fdf6e3;
                        padding: 1.5rem;
                        border-radius: 0.5rem;
                        border-left: 4px solid #ffc107;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    }
                    .warning-card strong {
                        color: #ffc107;
                    }
                    .success-card {
                        background-color: #1e3c2e;
                        color: #f0fff4;
                        padding: 1.5rem;
                        border-radius: 0.5rem;
                        border-left: 4px solid #28a745;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    }
                    .success-card strong {
                        color: #28a745;
                    }
                    .critical-card {
                        background-color: #3c1e1e;
                        color: #fff5f5;
                        padding: 1.5rem;
                        border-radius: 0.5rem;
                        border-left: 4px solid #dc3545;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    }
                    .critical-card strong {
                        color: #dc3545;
                    }
                </style>
                """, unsafe_allow_html=True)
                
                # eNPS interpretation
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Overall eNPS: {enps:.1f}</h3>
                        <p><strong>Promoters:</strong> {promoters}</p>
                        <p><strong>Passives:</strong> {passives}</p>
                        <p><strong>Detractors:</strong> {detractors}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # eNPS interpretation com cards mais escuros
                    if enps >= 50:
                        st.markdown('<div class="success-card"><strong>Excellent:</strong> World-class employee advocacy</div>', unsafe_allow_html=True)
                    elif enps >= 10:
                        st.markdown('<div class="metric-card"><strong>Good:</strong> Positive employee sentiment</div>', unsafe_allow_html=True)
                    elif enps >= -10:
                        st.markdown('<div class="warning-card"><strong>Needs Improvement:</strong> Mixed employee sentiment</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="critical-card"><strong>Critical:</strong> Negative employee sentiment</div>', unsafe_allow_html=True)

            with col2:
                # eNPS distribution
                labels = ['Detractors (1-2)', 'Passives (3)', 'Promoters (4-5)']
                values = [detractors, passives, promoters]
                colors = ['#ff6b6b', '#feca57', '#48cae4']
                
                fig = go.Figure(data=[go.Pie(
                    labels=labels, 
                    values=values, 
                    hole=0.3,
                    marker_colors=colors
                )])
                
                fig.update_layout(
                    title="eNPS Distribution",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # eNPS by Department
            if selected_dept == 'All':
                st.subheader("📊 eNPS by Department")
                
                dept_enps = []
                for dept in df['Department'].unique():
                    dept_data = df[df['Department'] == dept]['Recommendation'].dropna()
                    if len(dept_data) > 0:
                        dept_enps_score, _, _, _ = calculate_enps(dept_data)
                        dept_enps.append({'Department': dept, 'eNPS': dept_enps_score, 'Count': len(dept_data)})
                
                if dept_enps:
                    dept_df = pd.DataFrame(dept_enps)
                    
                    fig = px.bar(
                        dept_df, 
                        x='Department', 
                        y='eNPS',
                        title="eNPS by Department",
                        color='eNPS',
                        color_continuous_scale='RdYlGn',
                        text='eNPS'
                    )
                    
                    fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
                    fig.update_layout(height=400)
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header(f"📈 Dimensional Analysis{dept_title}")
        
        if not dim_scores.empty:
            # Dimensional scores comparison
            st.subheader("📊 Dimensional Scores Comparison")
            
            dim_means = dim_scores.mean().sort_values(ascending=True)
            
            fig = go.Figure()
            
            colors = ['#ff6b6b' if score < 3 else '#feca57' if score < 4 else '#48cae4' for score in dim_means.values]
            
            fig.add_trace(go.Bar(
                y=dim_means.index,
                x=dim_means.values,
                orientation='h',
                marker_color=colors,
                text=[f"{score:.2f}" for score in dim_means.values],
                textposition='auto'
            ))
            
            fig.update_layout(
                title="Average Scores by Dimension",
                xaxis_title="Average Score (1-5)",
                yaxis_title="Dimensions",
                height=600,
                xaxis=dict(range=[0, 5])
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown('---')

            # Correlation analysis
            st.subheader("🔗 Dimensional Correlations")
            
            corr_matrix = dim_scores.corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmin=-1,
                zmax=1,
                text=corr_matrix.round(2).values,
                texttemplate="%{text}",
                textfont={"size": 10},
                hoverongaps=False
            ))
            
            fig.update_layout(
                title="Correlation Matrix - Dimensional Scores",
                height=600,
                xaxis={'tickangle': 45},
                yaxis={'tickangle': 0}
            )
            
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("""
            #### **Correlation Analysis** 🔗
            **Purpose:** Identify relationships between engagement dimensions
            
            **Method:** Pearson correlation coefficient
            - **Range:** -1 to +1
            - **Interpretation:**
            - `0.7 to 1.0`: Strong positive relationship
            - `0.3 to 0.7`: Moderate positive relationship  
            - `-0.3 to 0.3`: Weak/no relationship
            - `-0.7 to -0.3`: Moderate negative relationship
            - `-1.0 to -0.7`: Strong negative relationship
            """)

            st.markdown('---')

            # Department comparison (if viewing all departments)
            if selected_dept == 'All':
                st.subheader("🏢 Department Comparison")
                
                dept_comparison = []
                for dept in df['Department'].unique():
                    dept_data = df[df['Department'] == dept]
                    dept_dim_scores, _ = create_dimensional_scores(dept_data, question_mapping)
                    if not dept_dim_scores.empty:
                        for dim in dept_dim_scores.columns:
                            dept_comparison.append({
                                'Department': dept,
                                'Dimension': dim,
                                'Score': dept_dim_scores[dim].mean(),
                                'Count': len(dept_data)
                            })
                
                if dept_comparison:
                    comp_df = pd.DataFrame(dept_comparison)

                    fig = px.bar(
                        comp_df,
                        x='Dimension',
                        y='Score',
                        color='Department',
                        title="Dimensional Scores by Department",
                        barmode='group'
                    )

                    fig.update_layout(
                        height=500,
                        xaxis={'tickangle': 45},
                        yaxis_title="Score",
                        xaxis_title="Dimension"
                    )

                    st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header(f"🔍 Deep Dive Analysis{dept_title}")
        
        # Impact prioritization
        st.subheader("🎯 Impact Prioritization")
        if 'Job_Satisfaction' in filtered_df.columns and not dim_scores.empty:
            correlations = {}
            for dim in dim_scores.columns:
                if dim != 'Professional Satisfaction':
                    corr = filtered_df['Job_Satisfaction'].corr(dim_scores[dim])
                    if not pd.isna(corr):
                        correlations[dim] = corr
            
            if correlations:
                corr_df = pd.DataFrame(list(correlations.items()), columns=['Dimension', 'Correlation'])
                corr_df = corr_df.sort_values('Correlation', ascending=True)
                
                fig = go.Figure()
                
                colors = ['#ff6b6b' if corr < 0.3 else '#feca57' if corr < 0.6 else '#48cae4' for corr in corr_df['Correlation']]
                
                fig.add_trace(go.Bar(
                    y=corr_df['Dimension'],
                    x=corr_df['Correlation'],
                    orientation='h',
                    marker_color=colors,
                    text=[f"{corr:.3f}" for corr in corr_df['Correlation']],
                    textposition='auto'
                ))
                
                fig.update_layout(
                    title="Correlation with Overall Professional Satisfaction",
                    xaxis_title="Correlation Coefficient",
                    yaxis_title="Dimensions",
                    height=500,
                    # annotations=[
                    #     dict(
                    #         x=0.5,
                    #         y=-0.15,
                    #         xref='paper',
                    #         yref='paper',
                    #         showarrow=False,
                    #         font=dict(size=10, color="gray")
                    #     )
                    # ]
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                if selected_dept == 'All':
                    # Adicionar explicação
                    st.markdown("""
                    **Key takeaway company-wide**
                    - Commitment & Belonging, Innovation & Voice and Leadership highly impact our employees' Professional Satisfaction and our retention rates
                    - These 3 are the ones we should prioritize in our initiatives

                    **How to Read This Chart:**
                    - **High correlation (>0.6)**: Strong relationship with Professional Satisfaction → High priority for improvement
                    - **Medium correlation (0.3-0.6)**: Moderate relationship → Medium priority  
                    - **Low correlation (<0.3)**: Weak relationship → Lower priority for satisfaction impact
                    
                    **Note**: The "Professional Satisfaction" dimension is excluded from this analysis to avoid self-correlation (which would always be 1.0).
                    """)
                elif selected_dept == 'Commercial, Operations & Support':
                    st.markdown("""
                    **Key takeaway Commercial, Operations & Support**
                    - Commitment & Belonging, Innovation & Voice and Leadership highly impact our employees' Professional Satisfaction and our retention rates.
                    - However, Culture & Values, Recognition & Development are also highly important when compared to company-wide correlations.

                    **How to Read This Chart:**
                    - **High correlation (>0.6)**: Strong relationship with Professional Satisfaction → High priority for improvement
                    - **Medium correlation (0.3-0.6)**: Moderate relationship → Medium priority  
                    - **Low correlation (<0.3)**: Weak relationship → Lower priority for satisfaction impact
                    
                    **Note**: The "Professional Satisfaction" dimension is excluded from this analysis to avoid self-correlation (which would always be 1.0).
                    """)
                elif selected_dept == 'Engineering, Product & Design':
                    st.markdown("""
                    **Key takeaway Engineering, Product & Design**
                    - Commitment & Belonging plays a significant role in EPD when it comes to Professional Satisfaction.
                    - Leadership, Collaboration and Recognition & Development are also important, but the priority should be Commitment & Belonging.

                    **How to Read This Chart:**
                    - **High correlation (>0.6)**: Strong relationship with Professional Satisfaction → High priority for improvement
                    - **Medium correlation (0.3-0.6)**: Moderate relationship → Medium priority  
                    - **Low correlation (<0.3)**: Weak relationship → Lower priority for satisfaction impact
                    
                    **Note**: The "Professional Satisfaction" dimension is excluded from this analysis to avoid self-correlation (which would always be 1.0).
                    """)
        
        st.markdown('---')

        # Culture analysis
        if 'How would you describe the current company culture?' in filtered_df.columns:
            st.subheader("🏛️ Culture Analysis")
            
            culture_responses = filtered_df['How would you describe the current company culture?'].dropna()
            if len(culture_responses) > 0:
                # Split multiple responses by comma and count individually
                all_responses = []
                for response in culture_responses:
                    # Split by comma and clean whitespace
                    individual_responses = [resp.strip() for resp in str(response).split(',')]
                    all_responses.extend(individual_responses)
                
                # Count individual responses
                from collections import Counter
                culture_counts = Counter(all_responses)
                
                # Convert to Series to maintain compatibility with original code
                culture_counts_series = pd.Series(culture_counts).sort_values(ascending=False).head(10)
                
                fig = px.bar(
                    x=culture_counts_series.values,
                    y=culture_counts_series.index,
                    orientation='h',
                    title="Most Common Culture Descriptions",
                    text=culture_counts_series.values  # Add text labels on bars
                )
                
                # Update layout and text formatting
                fig.update_traces(
                    texttemplate='%{text}',
                    textposition='outside'
                )
                
                fig.update_layout(
                    height=400,
                    xaxis_title="Number of Mentions",
                    yaxis_title="Culture Descriptions"
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.header(f"👥 Employee Personas{dept_title}")

        # Explanation of engagement levels
        st.markdown("""
        ### 📊 Understanding Engagement Levels
        
        Our analysis categorizes employees into four distinct engagement levels based on their average scores across all dimensions:
        
        - **🟢 Highly Engaged (4.0 - 5.0)**: Exceptional performers who excel across multiple dimensions and serve as organizational champions
        - **🔵 Engaged (3.5 - 3.9)**: Strong performers with consistent positive attitudes and good overall satisfaction
        - **🟡 Moderately Engaged (3.0 - 3.4)**: Employees with potential who need targeted support and development opportunities
        - **🔴 Disengaged (Below 3.0)**: At-risk employees requiring immediate attention and comprehensive intervention strategies
        
        Each persona is further characterized by their **dominant dimension** - the area where they show relatively stronger performance, 
        which can guide targeted development and retention strategies.
        
        ---
        """)

        if not dim_scores.empty and len(dim_scores) > 3:
            # Perform clustering analysis
            st.subheader("🎭 Engagement Personas")
            
            # Prepare data for clustering
            cluster_data = dim_scores.fillna(dim_scores.mean())
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(cluster_data)
            
            # Perform K-means clustering
            n_clusters = min(4, len(cluster_data))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(scaled_data)
            
            # Add cluster labels to dataframe
            cluster_df = dim_scores.copy()
            cluster_df['Cluster'] = clusters
            
            # Calculate cluster characteristics
            cluster_summary = cluster_df.groupby('Cluster').mean()
            
            # Calculate cluster sizes and sort by size (descending)
            cluster_sizes = cluster_df['Cluster'].value_counts().sort_values(ascending=False)
            sorted_clusters = cluster_sizes.index.tolist()
            
            # Create persona names and descriptions based on characteristics
            persona_names = []
            persona_descriptions = []
            
            for cluster_id in sorted_clusters:
                cluster_scores = cluster_summary.loc[cluster_id]
                top_dimension = cluster_scores.idxmax()
                avg_score = cluster_scores.mean()
                
                if avg_score >= 4:
                    persona_names.append(f"Highly Engaged - {top_dimension} Champions")
                    persona_descriptions.append(
                        f"This persona represents employees with exceptional engagement levels across all dimensions. "
                        f"They excel particularly in **{top_dimension}**, making them natural champions in this area. "
                        f"These employees are highly motivated, satisfied, and likely to be key contributors to organizational success. "
                        f"They can serve as mentors and role models for other team members."
                    )
                elif avg_score >= 3.5:
                    persona_names.append(f"Engaged - {top_dimension} Focused")
                    persona_descriptions.append(
                        f"This persona shows strong engagement with a particular focus on **{top_dimension}**. "
                        f"These employees are generally satisfied and motivated, with this dimension being their primary strength. "
                        f"They demonstrate consistent performance and positive attitudes, though there may be opportunities "
                        f"for growth in other engagement areas."
                    )
                elif avg_score >= 3:
                    persona_names.append(f"Moderately Engaged - {top_dimension} Seekers")
                    persona_descriptions.append(
                        f"This persona represents employees with moderate engagement levels who show the most potential in **{top_dimension}**. "
                        f"While they have room for improvement across various dimensions, their relative strength in {top_dimension} "
                        f"suggests this could be a key area for development and motivation. These employees may benefit from "
                        f"targeted support and development opportunities."
                    )
                else:
                    persona_names.append(f"Disengaged - {top_dimension} Concerned")
                    persona_descriptions.append(
                        f"This persona indicates employees facing significant engagement challenges across multiple dimensions. "
                        f"Even their relatively strongest area, **{top_dimension}**, shows concerning levels that require immediate attention. "
                        f"These employees are at risk and need comprehensive support, intervention, and potentially structural changes "
                        f"to improve their work experience and prevent turnover."
                    )
            
            # Display personas with enhanced information (ordered by size)
            for i, (cluster_id, persona_name, description) in enumerate(zip(sorted_clusters, persona_names, persona_descriptions)):
                st.subheader(f"Persona {i+1}: {persona_name}")
                
                # Add description
                st.markdown(description)
                
                persona_data = cluster_summary.loc[cluster_id]
                persona_count = len(cluster_df[cluster_df['Cluster'] == cluster_id])
                top_dimension = persona_data.idxmax()
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.metric("Employee Count", persona_count)
                    st.metric("Average Score", f"{persona_data.mean():.2f}")
                    st.metric("Dominant Dimension", top_dimension)
                    
                    # Top strengths
                    st.markdown("**Top 3 Dimensions:**")
                    top_3 = persona_data.nlargest(3)
                    for rank, (dim, score) in enumerate(top_3.items(), 1):
                        if rank == 1:
                            st.markdown(f"🥇 **{dim}**: {score:.2f}")
                        elif rank == 2:
                            st.markdown(f"🥈 {dim}: {score:.2f}")
                        else:
                            st.markdown(f"🥉 {dim}: {score:.2f}")
                    
                    # Areas for improvement
                    st.markdown("**Areas for Improvement:**")
                    bottom_2 = persona_data.nsmallest(2)
                    for dim, score in bottom_2.items():
                        st.markdown(f"⚠️ {dim}: {score:.2f}")
                
                with col2:
                    # Radar chart for persona
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatterpolar(
                        r=persona_data.values,
                        theta=persona_data.index,
                        fill='toself',
                        name=persona_name,
                        line_color='rgb(32, 201, 151)'
                    ))
                    
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 5],
                                tickmode='linear',
                                tick0=0,
                                dtick=1
                            )),
                        showlegend=False,
                        title=f"Engagement Profile - {top_dimension} Dominant",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)

    with tab6:
        st.header(f"📋 Action Plans{dept_title}")
        
        if not dim_scores.empty:
            # Priority matrix
            st.subheader("🎯 Priority Action Matrix")
            
            dim_means = dim_scores.mean()
            dim_impact = {}

            for dim in dim_scores.columns:
                factors = []
                
                # Factor 1: Correlation with Job Satisfaction (if available)
                if 'Job_Satisfaction' in filtered_df.columns and dim != 'Job_Satisfaction':
                    corr = filtered_df['Job_Satisfaction'].corr(dim_scores[dim])
                    if not pd.isna(corr):
                        factors.append(abs(corr))
                
                # Factor 2: Improvement potential (inverse of current score)
                improvement_potential = (5 - dim_means[dim]) / 4
                factors.append(improvement_potential)
                
                # Factor 3: Response variability (indicates impact potential)
                if len(dim_scores[dim]) > 1:
                    variability = dim_scores[dim].std() / 2.0  # Normalize to 0-1 scale
                    factors.append(min(variability, 1.0))
                
                # Calculate average impact score
                dim_impact[dim] = np.mean(factors) if factors else 0

            # Create priority matrix
            priority_data = []
            for dim in dim_means.index:
                priority_data.append({
                    'Dimension': dim,
                    'Current_Score': dim_means[dim],
                    'Impact_Score': dim_impact[dim],
                    'Priority': ('High' if dim_means[dim] < 3.8 and dim_impact[dim] > 0.3 else 
                                'Medium' if dim_means[dim] < 4.0 and dim_impact[dim] > 0.2 else 'Low')
                })
            
            priority_df = pd.DataFrame(priority_data)
            
            # Priority matrix visualization
            fig = px.scatter(
                priority_df,
                x='Current_Score',
                y='Impact_Score',
                color='Priority',
                size=[1]*len(priority_df),
                hover_name='Dimension',
                title="Action Priority Matrix",
                color_discrete_map={'High': '#ff6b6b', 'Medium': '#feca57', 'Low': '#48cae4'}
            )
            
            fig.add_vline(x=3.8, line_dash="dash", line_color="red", annotation_text="High Priority Score (3.8)")
            fig.add_vline(x=4.0, line_dash="dash", line_color="orange", annotation_text="Medium Priority Score (4.0)")
            fig.add_hline(y=0.3, line_dash="dash", line_color="red", annotation_text="High Impact (0.3)")
            fig.add_hline(y=0.2, line_dash="dash", line_color="orange", annotation_text="Medium Impact (0.2)")
            
            fig.update_layout(
                xaxis_title="Current Score",
                yaxis_title="Impact Score",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")
            with st.expander("📊 How Impact Score is Calculated", expanded=False):
                st.markdown("""
                ### 🧮 **Multi-Factor Impact Calculation**
                
                Instead of relying on a single metric, our Impact Score combines **three key factors** to provide a more robust and balanced assessment:
                
                #### **📈 Factor 1: Correlation with Job Satisfaction (33%)**
                - Measures how strongly each dimension correlates with overall job satisfaction
                - **Higher correlation** = Changes in this dimension significantly affect employee satisfaction
                - **Range:** 0.0 to 1.0 (absolute correlation coefficient)
                - **Note:** Only calculated when Job Satisfaction data is available
                
                #### **🎯 Factor 2: Improvement Potential (33%)**
                - Calculates room for improvement based on current performance
                - **Formula:** (5 - Current Score) ÷ 4
                - **Logic:** Dimensions with lower scores have higher potential for positive impact
                - **Range:** 0.0 (perfect score) to 1.0 (lowest possible score)
                
                #### **📊 Factor 3: Response Variability (33%)**
                - Measures the spread of responses within the dimension
                - **Higher variability** = More diverse opinions, indicating potential for targeted improvements
                - **Formula:** Standard Deviation ÷ 2 (normalized to 0-1 scale)
                - **Logic:** Consistent low scores suggest systemic issues; varied scores suggest segmented opportunities
                """)

            # Também adicione esta informação resumida na explicação principal:
            st.info("""
            💡 **Impact Score Methodology**: Combines correlation with job satisfaction, improvement potential, 
            and response variability to provide a balanced assessment of each dimension's strategic importance.
            """)

            # Explanation of priority matrix
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("""
                **🎯 Priority Levels:**
                
                **🔥 High Priority (Red):**
                - Score < 3.8 **AND** Impact > 0.3
                
                **⚠️ Medium Priority (Yellow):**
                - Score < 4.0 **AND** Impact > 0.2
                
                **✅ Low Priority (Blue):**
                - Score ≥ 4.0 **OR** Impact ≤ 0.2
                """)

            with col2:
                st.markdown("""
                **📊 Matrix Quadrants:**
                
                **Top-Left:** High impact + Low score = 🔥 **Urgent Priority**
                
                **Top-Right:** High impact + Good score = 👀 **Monitor Closely**
                
                **Bottom-Left:** Low impact + Low score = ⚠️ **Improve When Possible**
                
                **Bottom-Right:** Low impact + Good score = ✅ **Maintain Status**
                """)
            
            high_priority = priority_df[priority_df['Priority'] == 'High'].sort_values('Impact_Score', ascending=False)
            medium_priority = priority_df[priority_df['Priority'] == 'Medium'].sort_values('Impact_Score', ascending=False)
            
            # Action plan templates
            action_plans = {
                'Leadership': {
                    'short_term': [
                        "Replace 1 weekly all-hands with comprehensive monthly town hall. Agenda: Vision & Values refresher, New member welcomes, OKRs tracking, in-person events (off-sites, conferences), AP anniversaries, product roadmap update, sales/CS major wins and updates, employee spotlight. ",
                        "Quarterly career check-ins separate from performance reviews. Focus: career progression, concerns and feedback sharing.",
                        "Enhance AMAs: continue with quarterly frequency, but add a rotation for Product, People and Customer Success."
                    ],
                    'long_term': [
                        "External leadership training program: multi-workshop series for all leaders focused on communication, feedback delivery and team development.",
                    ]
                },
                'Recognition & Development': {
                    'short_term': [
                        "Employee spotlight program: monthly recognition at town halls. Leader-nominated employees embodying company values.",
                        "Long-tenure recognition to celebrate folks who are working with us for a long time. Example: 1 month paid sabbatical after 4 years tenure.",
                        "Employee referral program: bonus for successful hires after 3-month retention."
                    ],
                    'long_term': [
                        "Career progression framework, with company-wide career paths and a promotion structure integrated with performance review cycles.",
                        "Mentorship program beyond onboarding buddy system (future consideration)",
                    ]
                },
                'Compensation & Benefits': {
                    'short_term': [
                        "Comprehensive compensation statements (base + equity + benefits), including clear benefits explanation across Canada, US and Brazil, as well as an equity calculator for growth understanding.",
                        "Minimum PTO policy (3.69 score on comfortable taking time off): set a minimum required PTO days to reduce guilt.",
                    ],
                    'long_term': [
                        "Total compensation review: industry benchmark across all departments, review base salary, equity and benefits. Develop correction plan based on findings.",
                    ]
                }
            }
            
            # Display action plans for high priority areas
            if not high_priority.empty:
                if selected_dept == 'All':

                    st.markdown("---")
                    # Specific action plans
                    st.subheader("🚀 Suggested Action Plans")

                    for _, row in high_priority.head(3).iterrows():
                        dimension = row['Dimension']
                        score = row['Current_Score']
                        
                        st.markdown(f"#### {dimension} (Current Score: {score:.2f})")
                        
                        # Find matching action plan
                        action_key = None
                        for key in action_plans.keys():
                            if key.lower() in dimension.lower():
                                action_key = key
                                break
                        
                        if action_key:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**Quick Wins:**")
                                for action in action_plans[action_key]['short_term']:
                                    st.markdown(f"• {action}")
                            
                            with col2:
                                st.markdown("**Long-term**")
                                for action in action_plans[action_key]['long_term']:
                                    st.markdown(f"• {action}")
                        else:
                            st.markdown("**Recommended Actions:**")
                            st.markdown(f"• Conduct focused analysis on {dimension}")
                            st.markdown(f"• Gather employee feedback on specific pain points")
                            st.markdown(f"• Develop targeted improvement plan")
                            st.markdown(f"• Implement regular monitoring and measurement")
                        
                        st.markdown("---")
            
                    # Success metrics
                    st.subheader("📊 Next Steps")
                    
                    st.markdown("""
                    - Define what Action Plans will be tackled next quarter
                    - Define what KPIs to track
                    - Run Engagement Survey every quarter
                    - Define specific Action Plans for each department
                    
                    **Learnings**
                    - We didn't ask about tenure so we can cross-reference feedback based on how long they're working with us
                    - Commercial, Ops and Support ended up being jeopardized in one department. For next surveys we should split.
                    """)

    with tab7:
        st.header(f"🔬 Methodology & Data Insights{dept_title}")
        
        # Dimensional Groupings
        st.subheader("🎯 Dimensional Groupings & Rationale")
        
        st.markdown("""
        Our analysis groups the 33 survey questions into **10 strategic dimensions** based on organizational psychology research and engagement frameworks:
        """)
        
        # Display dimensional groupings with explanations
        dimensions_detailed = {
            'Commitment & Belonging': {
                'questions': ['Pride', 'Recommendation', 'Retention_Intent', 'Future_Commitment', 'Motivation'],
                'rationale': 'Core engagement indicators that predict employee advocacy, retention, and discretionary effort.'
            },
            'Leadership': {
                'questions': ['Leadership_Communication', 'Leadership_Direction', 'Manager_Feedback', 'Manager_Support'],
                'rationale': 'Leadership effectiveness directly impacts employee engagement, trust, and performance.'
            },
            'Professional Satisfaction': {
                'questions': ['Job_Satisfaction', 'Workload_Balance', 'Meaningful_Impact'],
                'rationale': 'Core job characteristics that influence daily work experience and intrinsic motivation.'
            },
            'Resources & Processes': {
                'questions': ['Resource_Access', 'Process_Effectiveness', 'PTO_Comfort'],
                'rationale': 'Operational enablers that remove barriers and support employee productivity.'
            },
            'Innovation & Voice': {
                'questions': ['Innovation_Empowerment', 'Psychological_Safety', 'Voice_Value', 'Risk_Safety'],
                'rationale': 'Psychological safety and empowerment drive innovation, learning, and continuous improvement.'
            },
            'Role & Purpose': {
                'questions': ['Role_Clarity', 'Purpose_Alignment', 'Quality_Focus'],
                'rationale': 'Clear expectations and meaningful purpose drive performance and engagement.'
            },
            'Collaboration': {
                'questions': ['Cross_Collaboration'],
                'rationale': 'Cross-functional effectiveness indicates organizational health and teamwork quality.'
            },
            'Recognition & Development': {
                'questions': ['Recognition', 'Career_Progression', 'Learning_Development', 'Career_Interest'],
                'rationale': 'Growth opportunities and recognition fulfill higher-level needs and drive retention.'
            },
            'Culture & Values': {
                'questions': ['Diversity_Inclusion', 'Employee_Wellbeing', 'Values_Understanding'],
                'rationale': 'Cultural alignment and inclusive environment create belonging and organizational commitment.'
            },
            'Compensation & Benefits': {
                'questions': ['Compensation_Satisfaction', 'Benefits_Understanding', 'Benefits_Satisfaction'],
                'rationale': 'Total rewards satisfaction addresses hygiene factors and perceived organizational support.'
            }
        }
        
        for dimension, details in dimensions_detailed.items():
            with st.expander(f"📊 {dimension} ({len(details['questions'])} questions)"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**Questions Included:**")
                    for q in details['questions']:
                        # Find original question text
                        original_q = None
                        for orig, mapped in question_mapping.items():
                            if mapped == q:
                                original_q = orig
                                break
                        if original_q:
                            st.markdown(f"• {original_q}")
                        else:
                            st.markdown(f"• {q}")
                    
                    st.markdown(f"**Rationale:** {details['rationale']}")
                
                with col2:
                    if dimension in dim_scores.columns:
                        avg_score = dim_scores[dimension].mean()
                        st.metric("Average Score", f"{avg_score:.2f}/5")
                        
                        # Score interpretation
                        if avg_score >= 4.0:
                            st.success("Strong Performance")
                        elif avg_score >= 3.5:
                            st.info("Good Performance")
                        elif avg_score >= 3.0:
                            st.warning("Needs Attention")
                        else:
                            st.error("Critical Area")
        
        # Statistical Methods
        st.subheader("📊 Statistical Methods")
        
        # Create tabs for different methodologies
        method_tab1, method_tab2 = st.tabs([
            "🎯 eNPS Calculation", 
            "🤖 Personas - Clustering", 
        ])
        
        with method_tab1:
            st.markdown("""
            ### eNPS (Employee Net Promoter Score) Methodology
            
            **Formula:** `eNPS = ((Promoters - Detractors) / Total Responses) × 100`
            
            **Classification:**
            - **Promoters (4-5):** Highly engaged employees who actively recommend the company
            - **Passives (3):** Satisfied but not enthusiastic employees
            - **Detractors (1-2):** Disengaged employees who may discourage others
            
            **Interpretation Scale:**
            - **+50 to +100:** Excellent (World-class)
            - **+10 to +49:** Good (Above average)
            - **-10 to +9:** Needs Improvement (Average)
            - **-100 to -11:** Critical (Below average)
            """)
            
            # Show actual eNPS calculation if data available
            if 'Recommendation' in filtered_df.columns:
                rec_data = filtered_df['Recommendation'].dropna()
                if len(rec_data) > 0:
                    enps, promoters, passives, detractors = calculate_enps(rec_data)
                    
                    st.markdown("### Current Calculation:")
                    st.code(f"""
    Total Responses: {len(rec_data)}
    Promoters (4-5): {promoters}
    Passives (3): {passives}  
    Detractors (1-2): {detractors}
                        
    eNPS = (({promoters} - {detractors}) / {len(rec_data)}) × 100 = {enps:.1f}""")
        
        with method_tab2:
            st.markdown("""
            ### Clustering Method
            
            #### 1. **K-Means Clustering** 🎯
            **Purpose:** Create employee personas based on engagement patterns (grouping of similar employees)
            
            **How it works:**
            ```python
            # 1. Standardize data (mean=0, std=1)
            scaled_data = StandardScaler().fit_transform(dimensional_scores)
            
            # 2. Apply K-Means clustering
            kmeans = KMeans(n_clusters=4, random_state=42)
            clusters = kmeans.fit_predict(scaled_data)
            ```
            
            **Why K-Means:**
            - Identifies natural groupings in employee responses
            - Creates actionable employee segments
            - Enables personalized HR strategies
            - Unsupervised learning (no predefined labels needed)
            
            #### 2. **StandardScaler (Normalization)** ⚖️
            **Purpose:** Ensure all dimensions have equal weight in analysis
            
            **Formula:** `z = (x - μ) / σ`
            - `x` = original value
            - `μ` = mean of dimension
            - `σ` = standard deviation of dimension
            
            **Why Necessary:**
            - Prevents bias toward dimensions with larger scales
            - Ensures fair comparison across all engagement areas
            - Required for distance-based algorithms like K-Means
            """)

if __name__ == "__main__":
    main()