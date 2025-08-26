"""
Universal Autonomous Research Analytics Engine with Enhanced EDA Visualizations
Works with any dataset by intelligently detecting patterns and relationships
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import sys
import argparse
from datetime import datetime
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Enhanced ML imports
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression, RFE

# Try importing XGBoost (optional)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Enhanced plotting style
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.style.use('default')
sns.set_palette("husl")

class ResearchQuestion:
    """Class to represent a research question with its methodology"""
    def __init__(self, question, target_variable, hypothesis, methodology, business_impact):
        self.question = question
        self.target_variable = target_variable
        self.hypothesis = hypothesis
        self.methodology = methodology
        self.business_impact = business_impact
        self.findings = []
        self.statistical_tests = []
        self.model_results = {}
        self.confidence_level = None

class UniversalAutonomousResearchAnalytics:
    """
    Universal Autonomous Research Analytics Engine with Enhanced EDA Visualizations
    Works with any dataset by intelligently detecting patterns and relationships
    """
    
    def __init__(self):
        self.data = None
        self.original_data = None
        self.filename = None
        
        # Research components
        self.research_questions = []
        self.selected_question = None
        self.research_findings = {}
        self.statistical_results = {}
        self.business_recommendations = []
        
        # Analysis results
        self.model_results = {}
        self.best_model = None
        self.best_model_name = None
        self.best_model_score = None
        
        # Store data for analysis
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred_best = None
        
        # Universal research question templates
        self.question_templates = {
            'prediction': "What factors most strongly predict {target}?",
            'optimization': "How can we optimize {target} performance?",
            'segmentation': "What are the key drivers of variation in {target}?",
            'relationship': "Which variables have the strongest relationship with {target}?",
            'improvement': "What changes would most effectively improve {target}?"
        }
        
        # NEW: Plot storage for EDA visualizations
        self.eda_plots_info = []
        self.output_dir = None
        
        # Analysis completion status
        self.analysis_completed = False
        self.comprehensive_report = None
        self.analysis_results = {}
    
    def run_complete_automatic_analysis(self, filepath, output_dir="research_output"):
        """🚀 Run complete automatic analysis with all recommended steps and open results"""
        print("🚀 Starting Complete Automatic Analysis...")
        print("Performing all recommended steps: Data Overview → EDA → Modeling → Insights → Recommendations")
        print("=" * 80)
        
        import os
        import subprocess
        import webbrowser
        
        try:
            # Create output directory with timestamp
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"{output_dir}_{timestamp}"
            os.makedirs(output_dir, exist_ok=True)
            
            # Set output directory for plots
            self.output_dir = output_dir
            
            print(f"📁 Output directory: {output_dir}")
            print("")
            
            # Step 1: Load Data
            print("📊 Step 1: Loading and validating data...")
            if not self.load_data(filepath):
                return {"success": False, "error": "Failed to load data", "output_dir": output_dir}
            print(f"   ✅ Data loaded: {len(self.data)} rows × {len(self.data.columns)} columns")
            print("")
            
            # Step 2: Data Exploration
            print("🔍 Step 2: Comprehensive data exploration...")
            data_overview = self.generate_detailed_data_overview()
            print("   ✅ Data overview complete")
            print("")
            
            # Step 3: Research Questions
            print("🧠 Step 3: Formulating research questions...")
            if not self.formulate_universal_research_questions():
                return {"success": False, "error": "Failed to formulate research questions", "output_dir": output_dir}
            
            if not self.select_primary_research_question():
                return {"success": False, "error": "Failed to select research question", "output_dir": output_dir}
            print(f"   ✅ Selected research focus: {self.selected_question.target_variable}")
            print("")
            
            # Step 4: Exploratory Data Analysis with Visualizations
            print("📈 Step 4: Creating comprehensive EDA visualizations...")
            if not self.conduct_exploratory_data_analysis():
                return {"success": False, "error": "Failed to conduct EDA", "output_dir": output_dir}
            print(f"   ✅ Created {len(self.eda_plots_info)} comprehensive visualizations")
            print("")
            
            # Step 5: Data Preprocessing
            print("🔧 Step 5: Intelligent data preprocessing...")
            if not self.preprocess_for_research():
                return {"success": False, "error": "Failed to preprocess data", "output_dir": output_dir}
            print("   ✅ Data preprocessing complete")
            print("")
            
            # Step 6: Predictive Modeling
            print("🤖 Step 6: Running predictive modeling...")
            if not self.run_predictive_modeling():
                return {"success": False, "error": "Failed to run predictive models", "output_dir": output_dir}
            print(f"   ✅ Best model: {self.best_model_name} (R² = {self.best_model_score:.4f})")
            print("")
            
            # Step 7: Generate Insights
            print("💡 Step 7: Generating research insights...")
            if not self.generate_research_insights():
                return {"success": False, "error": "Failed to generate insights", "output_dir": output_dir}
            print("   ✅ Research insights generated")
            print("")
            
            # Step 8: Create Comprehensive Report
            print("📄 Step 8: Creating comprehensive reports...")
            
            # Generate all reports
            detailed_report = self.generate_research_report()
            executive_dashboard = self.generate_executive_dashboard()
            
            # Save reports
            report_files = []
            
            # Detailed report
            detailed_file = os.path.join(output_dir, "detailed_research_report.txt")
            with open(detailed_file, 'w', encoding='utf-8') as f:
                f.write(detailed_report)
            report_files.append(detailed_file)
            print(f"   📄 Detailed report: {os.path.basename(detailed_file)}")
            
            # Executive dashboard
            dashboard_file = os.path.join(output_dir, "executive_dashboard.txt")
            with open(dashboard_file, 'w', encoding='utf-8') as f:
                f.write(executive_dashboard)
            report_files.append(dashboard_file)
            print(f"   📊 Executive dashboard: {os.path.basename(dashboard_file)}")
            
            # Comprehensive JSON results
            json_results = {
                "metadata": {
                    "filename": self.filename,
                    "analysis_date": datetime.now().isoformat(),
                    "engine_version": "Universal Analytics v2.0 Enhanced with EDA"
                },
                "research_framework": {
                    "primary_question": self.selected_question.question,
                    "target_variable": self.selected_question.target_variable,
                    "hypothesis": self.selected_question.hypothesis,
                    "business_impact": self.selected_question.business_impact
                },
                "model_performance": {
                    "best_model": self.best_model_name,
                    "r2_score": self.best_model_score,
                    "confidence_level": "High" if self.best_model_score > 0.7 else "Moderate" if self.best_model_score > 0.4 else "Limited"
                },
                "key_findings": self.selected_question.findings if hasattr(self.selected_question, 'findings') else [],
                "business_recommendations": self.generate_business_recommendations(),
                "eda_visualizations": {
                    "total_plots": len(self.eda_plots_info),
                    "plots_directory": "eda_plots/",
                    "plot_details": self.eda_plots_info
                },
                "data_quality": {
                    "original_rows": len(self.original_data),
                    "processed_rows": len(self.data_processed),
                    "retention_rate": (len(self.data_processed) / len(self.original_data)) * 100
                }
            }
            
            json_file = os.path.join(output_dir, "comprehensive_results.json")
            import json
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(json_results, f, indent=2)
            report_files.append(json_file)
            print(f"   📈 JSON results: {os.path.basename(json_file)}")
            
            # Feature importance CSV
            if self.model_results and 'feature_importance' in self.model_results[self.best_model_name]:
                import pandas as pd
                feature_df = pd.DataFrame([
                    {'feature': feature, 'importance': importance, 'rank': i+1}
                    for i, (feature, importance) in enumerate(
                        sorted(self.model_results[self.best_model_name]['feature_importance'].items(),
                              key=lambda x: x[1], reverse=True)
                    )
                ])
                
                csv_file = os.path.join(output_dir, "feature_importance.csv")
                feature_df.to_csv(csv_file, index=False)
                report_files.append(csv_file)
                print(f"   📋 Feature importance: {os.path.basename(csv_file)}")
            
            # EDA plots summary
            if self.eda_plots_info:
                import pandas as pd
                eda_df = pd.DataFrame(self.eda_plots_info)
                eda_csv_file = os.path.join(output_dir, "eda_plots_summary.csv")
                eda_df.to_csv(eda_csv_file, index=False)
                report_files.append(eda_csv_file)
                print(f"   📊 EDA plots summary: {os.path.basename(eda_csv_file)}")
            
            print("")
            
            # Step 9: Create Interactive HTML Summary (NEW!)
            print("🌐 Step 9: Creating interactive HTML summary...")
            html_file = self.create_interactive_html_summary(output_dir)
            if html_file:
                report_files.append(html_file)
                print(f"   🌐 Interactive summary: {os.path.basename(html_file)}")
            print("")
            
            # Step 10: Open Results
            print("🎉 Step 10: Opening results...")
            
            self.analysis_completed = True
            self.analysis_results = {
                "success": True,
                "output_dir": output_dir,
                "report_files": report_files,
                "research_question": self.selected_question.question,
                "target_variable": self.selected_question.target_variable,
                "best_model": self.best_model_name,
                "model_performance": self.best_model_score,
                "key_findings": self.selected_question.findings if hasattr(self.selected_question, 'findings') else [],
                "recommendations": self.generate_business_recommendations(),
                "eda_plots_count": len(self.eda_plots_info),
                "plots_directory": os.path.join(output_dir, "eda_plots")
            }
            
            # Open results in system
            self.open_analysis_results(output_dir, report_files)
            
            return self.analysis_results
            
        except Exception as e:
            print(f"❌ Complete analysis failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e), "output_dir": output_dir if 'output_dir' in locals() else "research_output"}
    
    def create_interactive_html_summary(self, output_dir):
        """Create an interactive HTML summary with embedded visualizations"""
        try:
            html_file = os.path.join(output_dir, "interactive_summary.html")
            
            # Get executive dashboard content
            dashboard_content = self.generate_executive_dashboard().replace('\n', '<br>\n')
            
            # Get top recommendations
            recommendations = self.generate_business_recommendations()
            
            # Create HTML content
            html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Data Analysis Results - {self.filename}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #2c3e50, #34495e);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        .header p {{
            margin: 10px 0 0 0;
            opacity: 0.9;
            font-size: 1.1em;
        }}
        .content {{
            padding: 30px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #f8f9fa, #e9ecef);
            border-left: 5px solid #667eea;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 5px;
        }}
        .metric-label {{
            color: #6c757d;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .section {{
            margin-bottom: 30px;
            background: #f8f9fa;
            border-radius: 10px;
            padding: 25px;
            border: 1px solid #e9ecef;
        }}
        .section h2 {{
            color: #2c3e50;
            margin-top: 0;
            padding-bottom: 10px;
            border-bottom: 2px solid #667eea;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .recommendations {{
            list-style: none;
            padding: 0;
        }}
        .recommendations li {{
            background: white;
            margin: 10px 0;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #28a745;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            transition: transform 0.2s;
        }}
        .recommendations li:hover {{
            transform: translateX(5px);
        }}
        .files-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px;
        }}
        .file-item {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #dee2e6;
            transition: all 0.2s;
            cursor: pointer;
        }}
        .file-item:hover {{
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            transform: translateY(-2px);
        }}
        .plots-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }}
        .plot-item {{
            background: white;
            padding: 10px;
            border-radius: 8px;
            text-align: center;
            border: 1px solid #dee2e6;
        }}
        .plot-preview {{
            width: 100%;
            height: 120px;
            background: linear-gradient(135deg, #f8f9fa, #e9ecef);
            border-radius: 4px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 10px;
            font-size: 2em;
        }}
        .confidence-high {{ color: #28a745; }}
        .confidence-medium {{ color: #ffc107; }}
        .confidence-limited {{ color: #dc3545; }}
        .footer {{
            text-align: center;
            padding: 20px;
            background: #f8f9fa;
            color: #6c757d;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Data Analysis Results</h1>
            <p>Dataset: {self.filename} | Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="content">
            <!-- Key Metrics -->
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value confidence-{'high' if self.best_model_score > 0.7 else 'medium' if self.best_model_score > 0.4 else 'limited'}">
                        {self.best_model_score:.2%}
                    </div>
                    <div class="metric-label">Predictive Accuracy</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{len(self.eda_plots_info)}</div>
                    <div class="metric-label">Visualizations Created</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{len(self.generate_business_recommendations())}</div>
                    <div class="metric-label">Business Recommendations</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{len(self.data_processed) if hasattr(self, 'data_processed') else 0:,}</div>
                    <div class="metric-label">Data Points Analyzed</div>
                </div>
            </div>
            
            <!-- Research Focus -->
            <div class="section">
                <h2>🎯 Research Focus</h2>
                <p><strong>Question:</strong> {self.selected_question.question}</p>
                <p><strong>Target Variable:</strong> {self.selected_question.target_variable}</p>
                <p><strong>Best Model:</strong> {self.best_model_name}</p>
                <p><strong>Hypothesis:</strong> {self.selected_question.hypothesis}</p>
            </div>
            
            <!-- Key Recommendations -->
            <div class="section">
                <h2>💡 Key Recommendations</h2>
                <ul class="recommendations">
"""
            
            # Add recommendations
            for i, rec in enumerate(recommendations[:5], 1):
                html_content += f"                    <li><strong>{i}.</strong> {rec}</li>\n"
            
            html_content += f"""
                </ul>
            </div>
            
            <!-- Visualizations -->
            <div class="section">
                <h2>📊 EDA Visualizations Created</h2>
                <p>Comprehensive visual analysis including distribution plots, correlation analysis, and data quality assessment.</p>
                <div class="plots-grid">
"""
            
            # Add plot previews
            plot_icons = {
                'distribution': '📈',
                'correlation': '🔗', 
                'scatter': '📊',
                'outlier': '🎯',
                'categorical': '📂',
                'quality': '🔍',
                'missing': '🕳️'
            }
            
            for plot in self.eda_plots_info:
                icon = '📊'  # default
                for key, value in plot_icons.items():
                    if key in plot['filename'].lower():
                        icon = value
                        break
                
                html_content += f"""
                    <div class="plot-item">
                        <div class="plot-preview">{icon}</div>
                        <strong>{plot['title']}</strong>
                        <p style="font-size: 0.8em; color: #6c757d;">{plot['filename']}</p>
                    </div>
"""
            
            html_content += f"""
                </div>
            </div>
            
            <!-- Generated Files -->
            <div class="section">
                <h2>📁 Generated Files</h2>
                <div class="files-grid">
                    <div class="file-item">
                        <strong>📄 Detailed Research Report</strong>
                        <p>Comprehensive analysis with methodology, findings, and technical details</p>
                        <small>detailed_research_report.txt</small>
                    </div>
                    <div class="file-item">
                        <strong>📊 Executive Dashboard</strong>
                        <p>High-level summary with key metrics and actionable insights</p>
                        <small>executive_dashboard.txt</small>
                    </div>
                    <div class="file-item">
                        <strong>📈 Comprehensive Results</strong>
                        <p>Complete analysis results in JSON format for further processing</p>
                        <small>comprehensive_results.json</small>
                    </div>
                    <div class="file-item">
                        <strong>🎨 EDA Visualizations</strong>
                        <p>{len(self.eda_plots_info)} high-quality plots for visual analysis</p>
                        <small>eda_plots/ directory</small>
                    </div>
                </div>
            </div>
            
            <!-- Next Steps -->
            <div class="section">
                <h2>🚀 Next Steps</h2>
                <ol>
                    <li><strong>Review the detailed report</strong> for comprehensive analysis methodology and findings</li>
                    <li><strong>Examine EDA visualizations</strong> to understand data patterns and relationships</li>
                    <li><strong>Implement priority recommendations</strong> based on business impact assessment</li>
                    <li><strong>Monitor outcomes</strong> and validate model predictions with real-world results</li>
                    <li><strong>Iterate and improve</strong> by collecting additional relevant data</li>
                </ol>
            </div>
        </div>
        
        <div class="footer">
            <p>Generated by Universal Autonomous Research Analytics Engine v2.0 Enhanced with EDA</p>
            <p>For questions or support, refer to the detailed technical documentation</p>
        </div>
    </div>
</body>
</html>
"""
            
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            return html_file
            
        except Exception as e:
            print(f"❌ Failed to create HTML summary: {e}")
            return None
    
    def open_analysis_results(self, output_dir, report_files):
        """Open analysis results in system applications"""
        import os
        import subprocess
        import webbrowser
        import platform
        
        try:
            print("   🖥️  Opening analysis results...")
            
            # Open the output directory in Finder (macOS)
            if platform.system() == "Darwin":
                subprocess.run(["open", output_dir])
                print(f"   📁 Opened output folder in Finder")
            elif platform.system() == "Windows":
                subprocess.run(["explorer", output_dir])
                print(f"   📁 Opened output folder in Explorer")
            else:
                subprocess.run(["xdg-open", output_dir])
                print(f"   📁 Opened output folder")
            
            # Open HTML summary in web browser
            html_file = os.path.join(output_dir, "interactive_summary.html")
            if os.path.exists(html_file):
                webbrowser.open('file://' + os.path.abspath(html_file))
                print(f"   🌐 Opened interactive summary in browser")
            
            # Open key visualizations
            plots_dir = os.path.join(output_dir, "eda_plots")
            if os.path.exists(plots_dir) and self.eda_plots_info:
                # Open first few key plots
                key_plots = []
                for plot in self.eda_plots_info[:3]:  # First 3 plots
                    plot_path = plot['filepath']
                    if os.path.exists(plot_path):
                        key_plots.append(plot_path)
                
                if key_plots and platform.system() == "Darwin":
                    # Open plots in Preview on macOS
                    subprocess.run(["open"] + key_plots[:2])  # Open first 2 plots
                    print(f"   📊 Opened key visualizations in Preview")
            
            print("")
            print("✅ ANALYSIS COMPLETE! All results opened successfully.")
            print(f"📁 Full results available in: {output_dir}")
            print("")
            
            # Show summary
            print("📊 ANALYSIS SUMMARY:")
            confidence = "High" if self.best_model_score > 0.7 else "Moderate" if self.best_model_score > 0.4 else "Limited"
            print(f"   🎯 Research Focus: {self.selected_question.target_variable}")
            print(f"   📈 Predictive Accuracy: {self.best_model_score:.2%} ({confidence} confidence)")
            print(f"   📊 Visualizations: {len(self.eda_plots_info)} comprehensive plots")
            print(f"   💡 Recommendations: {len(self.generate_business_recommendations())} actionable insights")
            
            return True
            
        except Exception as e:
            print(f"⚠️  Could not auto-open results: {e}")
            print(f"📁 Please manually open: {output_dir}")
            return False
    
    def load_data(self, filepath):
        """Enhanced data loading with universal compatibility"""
        try:
            if not os.path.exists(filepath):
                print(f"❌ File not found: {filepath}")
                return False
            
            self.filename = os.path.basename(filepath)
            file_extension = os.path.splitext(filepath)[1].lower()
            
            print(f"📁 Loading: {self.filename}")
            
            # Load data based on file type
            if file_extension == '.csv':
                for encoding in ['utf-8', 'latin1', 'cp1252']:
                    for sep in [',', ';', '\t']:
                        try:
                            self.data = pd.read_csv(filepath, encoding=encoding, sep=sep)
                            if len(self.data.columns) > 1:
                                break
                        except:
                            continue
                    if self.data is not None and len(self.data.columns) > 1:
                        break
                        
            elif file_extension in ['.xlsx', '.xls']:
                self.data = pd.read_excel(filepath)
                
            if self.data is None or len(self.data) == 0:
                print("❌ Could not load data")
                return False
            
            # Clean column names
            self.data.columns = self.data.columns.str.strip().str.replace('\n', ' ')
            self.original_data = self.data.copy()
            
            print(f"✅ Loaded: {len(self.data)} rows × {len(self.data.columns)} columns")
            return True
            
        except Exception as e:
            print(f"❌ Error loading file: {str(e)}")
            return False
    
    def analyze_column_characteristics(self, col):
        """Analyze characteristics of a column to determine its suitability as a target"""
        data = self.data[col].dropna()
        
        if len(data) == 0:
            return {'score': 0, 'reason': 'No valid data'}
        
        score = 0
        characteristics = {}
        
        # Data completeness
        completeness = len(data) / len(self.data)
        score += completeness * 30
        characteristics['completeness'] = completeness
        
        # Check if numeric
        if pd.api.types.is_numeric_dtype(data):
            score += 20
            characteristics['is_numeric'] = True
            
            # Variance (normalized)
            if data.std() > 0:
                cv = data.std() / abs(data.mean()) if data.mean() != 0 else 0
                score += min(cv * 10, 20)
                characteristics['coefficient_variation'] = cv
            
            # Distribution properties
            unique_ratio = data.nunique() / len(data)
            score += min(unique_ratio * 20, 15)
            characteristics['unique_ratio'] = unique_ratio
            
            # Avoid binary variables for regression
            if data.nunique() == 2:
                score -= 10
                
        else:
            characteristics['is_numeric'] = False
            # For categorical, check if it can be meaningfully encoded
            unique_count = data.nunique()
            if 2 <= unique_count <= 20:
                score += 10
                characteristics['unique_categories'] = unique_count
            elif unique_count > 50:
                score -= 20  # Too many categories
        
        # Business relevance keywords
        business_keywords = [
            'price', 'cost', 'revenue', 'sales', 'profit', 'income', 'salary', 'wage',
            'rating', 'score', 'satisfaction', 'performance', 'efficiency', 'quality',
            'time', 'duration', 'length', 'count', 'amount', 'quantity', 'volume',
            'rate', 'percentage', 'ratio', 'index', 'value', 'total', 'average'
        ]
        
        col_lower = col.lower()
        for keyword in business_keywords:
            if keyword in col_lower:
                score += 15
                break
        
        characteristics['score'] = score
        return characteristics
    
    def identify_target_variables(self):
        """Intelligently identify potential target variables from any dataset"""
        print("🎯 Identifying potential target variables...")
        
        # Remove obvious ID columns
        potential_targets = []
        for col in self.data.columns:
            col_lower = col.lower()
            
            # Skip ID columns
            if any(id_term in col_lower for id_term in ['id', '_id', 'key', 'index']):
                continue
                
            # Skip date columns
            if self.data[col].dtype == 'datetime64[ns]' or 'date' in col_lower:
                continue
            
            # Analyze column characteristics
            characteristics = self.analyze_column_characteristics(col)
            
            if characteristics['score'] > 20:  # Minimum threshold
                potential_targets.append((col, characteristics))
        
        # Sort by score
        potential_targets.sort(key=lambda x: x[1]['score'], reverse=True)
        
        print(f"   Found {len(potential_targets)} potential target variables:")
        for i, (col, chars) in enumerate(potential_targets[:5], 1):
            print(f"   {i}. {col} (Score: {chars['score']:.1f})")
        
        return potential_targets
    
    def generate_research_hypothesis(self, target_var):
        """Generate intelligent hypothesis based on target variable characteristics"""
        target_lower = target_var.lower()
        
        # Pattern-based hypothesis generation
        if any(term in target_lower for term in ['price', 'cost', 'salary', 'wage', 'income']):
            return f"Multiple economic and demographic factors significantly influence {target_var}"
        elif any(term in target_lower for term in ['rating', 'score', 'satisfaction']):
            return f"Service quality and customer experience factors drive {target_var} variations"
        elif any(term in target_lower for term in ['time', 'duration', 'length']):
            return f"Operational and contextual variables significantly affect {target_var}"
        elif any(term in target_lower for term in ['performance', 'efficiency', 'quality']):
            return f"Process and resource factors are key determinants of {target_var}"
        else:
            return f"Multiple variables in the dataset significantly contribute to predicting {target_var}"
    
    def generate_business_impact_statement(self, target_var):
        """Generate business impact statement based on target variable"""
        target_lower = target_var.lower()
        
        if any(term in target_lower for term in ['price', 'cost', 'revenue', 'sales']):
            return "Optimize pricing strategies and identify revenue growth opportunities"
        elif any(term in target_lower for term in ['satisfaction', 'rating', 'quality']):
            return "Improve customer experience and service quality delivery"
        elif any(term in target_lower for term in ['performance', 'efficiency']):
            return "Enhance operational efficiency and performance optimization"
        elif any(term in target_lower for term in ['time', 'duration']):
            return "Optimize processes and reduce operational time costs"
        else:
            return f"Understand key drivers of {target_var} for strategic decision-making"
    
    def formulate_universal_research_questions(self):
        """Formulate research questions that work with any dataset"""
        print("🧠 Formulating universal research questions...")
        
        # Identify potential target variables
        target_candidates = self.identify_target_variables()
        
        if not target_candidates:
            print("❌ No suitable target variables found")
            return False
        
        self.research_questions = []
        
        # Generate research questions for top candidates
        for target_var, characteristics in target_candidates[:5]:  # Top 5 candidates
            
            # Choose question template based on characteristics
            if characteristics.get('is_numeric', False):
                if 'price' in target_var.lower() or 'cost' in target_var.lower():
                    template = self.question_templates['optimization']
                else:
                    template = self.question_templates['prediction']
            else:
                template = self.question_templates['relationship']
            
            research_question = ResearchQuestion(
                question=template.format(target=target_var),
                target_variable=target_var,
                hypothesis=self.generate_research_hypothesis(target_var),
                methodology="Predictive modeling with feature importance analysis and statistical validation",
                business_impact=self.generate_business_impact_statement(target_var)
            )
            
            self.research_questions.append(research_question)
        
        print(f"✅ Generated {len(self.research_questions)} research questions:")
        for i, q in enumerate(self.research_questions, 1):
            print(f"   {i}. {q.question}")
        
        return len(self.research_questions) > 0
    
    def select_primary_research_question(self):
        """Select the most promising research question for investigation"""
        if not self.research_questions:
            return None
        
        print("\n🎯 Selecting primary research question...")
        
        # Score questions based on data quality and analytical potential
        question_scores = []
        
        for q in self.research_questions:
            score = 0
            target_data = self.data[q.target_variable].dropna()
            
            # Data availability
            completeness = len(target_data) / len(self.data)
            score += completeness * 40
            
            # Data variance (for numeric targets)
            if pd.api.types.is_numeric_dtype(target_data):
                score += 20
                if len(target_data.unique()) > 2:
                    cv = target_data.std() / abs(target_data.mean()) if target_data.mean() != 0 else 0
                    score += min(cv * 15, 20)
            
            # Feature availability
            available_features = len([col for col in self.data.columns 
                                    if col != q.target_variable and 
                                    not any(id_term in col.lower() for id_term in ['id', '_id'])])
            score += min(available_features * 2, 20)
            
            question_scores.append((q, score))
        
        # Select the highest scoring question
        self.selected_question = max(question_scores, key=lambda x: x[1])[0]
        
        print(f"🎯 Selected Research Question:")
        print(f"   Question: {self.selected_question.question}")
        print(f"   Target Variable: {self.selected_question.target_variable}")
        print(f"   Hypothesis: {self.selected_question.hypothesis}")
        print(f"   Business Impact: {self.selected_question.business_impact}")
        
        return self.selected_question
    
    # NEW: Enhanced EDA plotting methods
    def setup_plotting_environment(self, output_dir):
        """Setup plotting environment and directories"""
        self.output_dir = output_dir
        plots_dir = os.path.join(output_dir, 'eda_plots')
        os.makedirs(plots_dir, exist_ok=True)
        return plots_dir
    
    def save_plot(self, filename, title, description=""):
        """Save current plot and track it"""
        if not self.output_dir:
            return
        
        plots_dir = os.path.join(self.output_dir, 'eda_plots')
        filepath = os.path.join(plots_dir, filename)
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        
        self.eda_plots_info.append({
            'filename': filename,
            'title': title,
            'description': description,
            'filepath': filepath
        })
        
        plt.close()
    
    def plot_target_distribution(self, target):
        """Create comprehensive target variable distribution plots"""
        target_data = self.data[target].dropna()
        
        if pd.api.types.is_numeric_dtype(target_data):
            # Numeric target: histogram and box plot
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Histogram
            ax1.hist(target_data, bins=min(50, len(target_data.unique())), alpha=0.7, color='skyblue', edgecolor='black')
            ax1.set_title(f'Distribution of {target}')
            ax1.set_xlabel(target)
            ax1.set_ylabel('Frequency')
            ax1.grid(True, alpha=0.3)
            
            # Box plot
            ax2.boxplot(target_data, vert=True, patch_artist=True, 
                       boxprops=dict(facecolor='lightblue', alpha=0.7))
            ax2.set_title(f'Box Plot of {target}')
            ax2.set_ylabel(target)
            ax2.grid(True, alpha=0.3)
            
            # Q-Q plot for normality
            from scipy import stats
            stats.probplot(target_data, dist="norm", plot=ax3)
            ax3.set_title(f'Q-Q Plot: {target} vs Normal Distribution')
            ax3.grid(True, alpha=0.3)
            
            # Density plot
            ax4.hist(target_data, bins=min(50, len(target_data.unique())), 
                    density=True, alpha=0.7, color='lightgreen', edgecolor='black')
            # Add KDE curve
            from scipy.stats import gaussian_kde
            if len(target_data) > 10:
                kde = gaussian_kde(target_data)
                x_range = np.linspace(target_data.min(), target_data.max(), 100)
                ax4.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
                ax4.legend()
            ax4.set_title(f'Density Plot of {target}')
            ax4.set_xlabel(target)
            ax4.set_ylabel('Density')
            ax4.grid(True, alpha=0.3)
            
            plt.suptitle(f'Target Variable Analysis: {target}', fontsize=16, y=0.98)
            
            self.save_plot(
                f'target_distribution_{target.replace(" ", "_")}.png',
                f'Target Variable Distribution: {target}',
                f'Comprehensive distribution analysis of the target variable {target} including histogram, box plot, Q-Q plot, and density estimation.'
            )
            
        else:
            # Categorical target: bar plot and pie chart
            value_counts = target_data.value_counts().head(20)  # Top 20 categories
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Bar plot
            value_counts.plot(kind='bar', ax=ax1, color='skyblue', edgecolor='black')
            ax1.set_title(f'Distribution of {target}')
            ax1.set_xlabel(target)
            ax1.set_ylabel('Count')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3)
            
            # Pie chart (top 10 only)
            top_10 = value_counts.head(10)
            if len(value_counts) > 10:
                others_count = value_counts.iloc[10:].sum()
                top_10['Others'] = others_count
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(top_10)))
            ax2.pie(top_10.values, labels=top_10.index, autopct='%1.1f%%', colors=colors)
            ax2.set_title(f'Proportion of {target} (Top Categories)')
            
            plt.suptitle(f'Target Variable Analysis: {target}', fontsize=16)
            
            self.save_plot(
                f'target_distribution_{target.replace(" ", "_")}.png',
                f'Target Variable Distribution: {target}',
                f'Distribution analysis of the categorical target variable {target} showing frequency counts and proportions.'
            )
    
    def plot_correlation_heatmap(self, target):
        """Create correlation heatmap focused on target variable"""
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            return
        
        # Calculate correlations
        corr_matrix = self.data[numeric_cols].corr()
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8}, fmt='.3f')
        
        plt.title('Correlation Matrix of Numeric Variables', fontsize=16)
        plt.tight_layout()
        
        self.save_plot(
            'correlation_heatmap.png',
            'Correlation Matrix',
            'Heatmap showing correlations between all numeric variables in the dataset.'
        )
    
    def plot_target_correlations(self, target):
        """Create focused correlation plots with target variable"""
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        if target not in numeric_cols or len(numeric_cols) < 2:
            return
        
        # Get correlations with target
        target_corrs = self.data[numeric_cols].corr()[target].abs().sort_values(ascending=False)
        target_corrs = target_corrs[target_corrs.index != target].head(10)
        
        if len(target_corrs) == 0:
            return
        
        # Create correlation bar plot
        plt.figure(figsize=(12, 8))
        colors = ['red' if abs(self.data[numeric_cols].corr()[target][var]) >= 0.5 else 
                 'orange' if abs(self.data[numeric_cols].corr()[target][var]) >= 0.3 else 'lightblue' 
                 for var in target_corrs.index]
        
        bars = plt.barh(range(len(target_corrs)), target_corrs.values, color=colors, edgecolor='black')
        plt.yticks(range(len(target_corrs)), target_corrs.index)
        plt.xlabel('Absolute Correlation with Target')
        plt.title(f'Top Correlations with {target}')
        plt.grid(True, alpha=0.3, axis='x')
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, target_corrs.values)):
            plt.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.3f}', 
                    va='center', fontweight='bold')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='red', label='Strong (|r| ≥ 0.5)'),
                          Patch(facecolor='orange', label='Moderate (0.3 ≤ |r| < 0.5)'),
                          Patch(facecolor='lightblue', label='Weak (|r| < 0.3)')]
        plt.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        
        self.save_plot(
            f'target_correlations_{target.replace(" ", "_")}.png',
            f'Correlations with {target}',
            f'Bar chart showing the strength of correlations between numeric variables and the target variable {target}.'
        )
    
    def plot_scatter_plots(self, target):
        """Create scatter plots of top correlated variables with target"""
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        if target not in numeric_cols or len(numeric_cols) < 2:
            return
        
        # Get top correlated variables
        correlations = self.data[numeric_cols].corr()[target].abs().sort_values(ascending=False)
        top_vars = correlations[correlations.index != target].head(6).index.tolist()
        
        if len(top_vars) == 0:
            return
        
        # Create scatter plots
        n_plots = min(6, len(top_vars))
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, var in enumerate(top_vars[:n_plots]):
            ax = axes[i]
            
            # Create scatter plot
            ax.scatter(self.data[var], self.data[target], alpha=0.6, color='blue', s=30)
            
            # Add trend line
            try:
                from scipy.stats import linregress
                clean_data = self.data[[var, target]].dropna()
                if len(clean_data) > 5:
                    slope, intercept, r_value, p_value, std_err = linregress(clean_data[var], clean_data[target])
                    line = slope * clean_data[var] + intercept
                    ax.plot(clean_data[var], line, 'r-', alpha=0.8, linewidth=2)
                    
                    # Add correlation info
                    ax.text(0.05, 0.95, f'r = {r_value:.3f}', transform=ax.transAxes, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                           fontsize=10, fontweight='bold')
            except:
                pass
            
            ax.set_xlabel(var)
            ax.set_ylabel(target)
            ax.set_title(f'{var} vs {target}')
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_plots, 6):
            axes[i].set_visible(False)
        
        plt.suptitle(f'Scatter Plots: Top Variables vs {target}', fontsize=16)
        plt.tight_layout()
        
        self.save_plot(
            f'scatter_plots_{target.replace(" ", "_")}.png',
            f'Scatter Plots with {target}',
            f'Scatter plots showing relationships between the most correlated variables and the target variable {target}.'
        )
    
    def plot_missing_data_analysis(self):
        """Create comprehensive missing data visualization"""
        missing_data = self.data.isnull()
        missing_counts = missing_data.sum()
        missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)
        
        if len(missing_counts) == 0:
            # No missing data
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, 'No Missing Data Found!\nDataset is Complete', 
                    ha='center', va='center', fontsize=20, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.axis('off')
            plt.title('Missing Data Analysis', fontsize=16)
            
            self.save_plot(
                'missing_data_analysis.png',
                'Missing Data Analysis',
                'Analysis showing no missing data in the dataset.'
            )
            return
        
        # Create missing data visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Bar plot of missing counts
        missing_pct = (missing_counts / len(self.data)) * 100
        colors = ['red' if pct > 50 else 'orange' if pct > 20 else 'yellow' if pct > 5 else 'lightblue' 
                 for pct in missing_pct]
        
        bars = ax1.barh(range(len(missing_counts)), missing_pct.values, color=colors, edgecolor='black')
        ax1.set_yticks(range(len(missing_counts)))
        ax1.set_yticklabels(missing_counts.index)
        ax1.set_xlabel('Missing Data Percentage')
        ax1.set_title('Missing Data by Variable')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Add percentage labels
        for i, (bar, pct) in enumerate(zip(bars, missing_pct.values)):
            ax1.text(pct + 1, bar.get_y() + bar.get_height()/2, f'{pct:.1f}%', 
                    va='center', fontweight='bold')
        
        # Missing data heatmap (sample)
        if len(self.data) > 1000:
            sample_data = self.data.sample(1000, random_state=42)
        else:
            sample_data = self.data
        
        missing_matrix = sample_data.isnull()
        sns.heatmap(missing_matrix.iloc[:, missing_matrix.any()], 
                   cbar=True, ax=ax2, cmap='viridis_r',
                   yticklabels=False if len(sample_data) > 100 else True)
        ax2.set_title('Missing Data Pattern (Sample)')
        ax2.set_xlabel('Variables')
        
        plt.suptitle('Missing Data Analysis', fontsize=16)
        plt.tight_layout()
        
        self.save_plot(
            'missing_data_analysis.png',
            'Missing Data Analysis',
            'Comprehensive analysis of missing data patterns including percentages and visual patterns.'
        )
    
    def plot_data_quality_overview(self):
        """Create data quality overview dashboard"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Data types distribution
        dtype_counts = self.data.dtypes.value_counts()
        ax1.pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Data Types Distribution')
        
        # Missing data summary
        missing_counts = self.data.isnull().sum()
        complete_cols = (missing_counts == 0).sum()
        partial_cols = ((missing_counts > 0) & (missing_counts < len(self.data))).sum()
        empty_cols = (missing_counts == len(self.data)).sum()
        
        completeness_data = [complete_cols, partial_cols, empty_cols]
        completeness_labels = ['Complete', 'Partial Missing', 'Empty']
        colors = ['green', 'orange', 'red']
        
        ax2.pie(completeness_data, labels=completeness_labels, autopct='%1.1f%%', 
               colors=colors, startangle=90)
        ax2.set_title('Data Completeness Overview')
        
        # Numeric vs categorical variables
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        datetime_cols = self.data.select_dtypes(include=['datetime']).columns
        
        var_types = [len(numeric_cols), len(categorical_cols), len(datetime_cols)]
        var_labels = ['Numeric', 'Categorical', 'DateTime']
        
        ax3.bar(var_labels, var_types, color=['blue', 'green', 'purple'], alpha=0.7, edgecolor='black')
        ax3.set_title('Variable Types Count')
        ax3.set_ylabel('Number of Variables')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, v in enumerate(var_types):
            ax3.text(i, v + 0.1, str(v), ha='center', va='bottom', fontweight='bold')
        
        # Dataset size metrics
        metrics = ['Rows', 'Columns', 'Cells', 'Missing Cells']
        values = [len(self.data), len(self.data.columns), 
                 len(self.data) * len(self.data.columns),
                 self.data.isnull().sum().sum()]
        
        colors_metrics = ['skyblue', 'lightgreen', 'orange', 'red']
        bars = ax4.bar(metrics, values, color=colors_metrics, alpha=0.7, edgecolor='black')
        ax4.set_title('Dataset Size Metrics')
        ax4.set_ylabel('Count')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                    f'{val:,}', ha='center', va='bottom', fontweight='bold')
        
        plt.suptitle('Data Quality Overview Dashboard', fontsize=16)
        plt.tight_layout()
        
        self.save_plot(
            'data_quality_overview.png',
            'Data Quality Overview',
            'Comprehensive dashboard showing data types, completeness, variable types, and dataset metrics.'
        )
    
    def plot_outlier_analysis(self, target):
        """Create outlier analysis plots for numeric variables"""
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        if target in numeric_cols:
            # Include target in analysis
            analysis_cols = [target]
            # Add top correlated variables
            if len(numeric_cols) > 1:
                correlations = self.data[numeric_cols].corr()[target].abs().sort_values(ascending=False)
                top_vars = correlations[correlations.index != target].head(5).index.tolist()
                analysis_cols.extend(top_vars)
        else:
            analysis_cols = numeric_cols[:6]  # Top 6 numeric columns
        
        if len(analysis_cols) == 0:
            return
        
        # Create box plots for outlier detection
        n_cols = min(3, len(analysis_cols))
        n_rows = (len(analysis_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1 or n_cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(analysis_cols):
            if i >= len(axes):
                break
                
            ax = axes[i]
            data_col = self.data[col].dropna()
            
            # Create box plot
            bp = ax.boxplot(data_col, patch_artist=True, vert=True)
            bp['boxes'][0].set_facecolor('lightblue')
            bp['boxes'][0].set_alpha(0.7)
            
            # Calculate outlier statistics
            Q1 = data_col.quantile(0.25)
            Q3 = data_col.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = data_col[(data_col < lower_bound) | (data_col > upper_bound)]
            outlier_pct = (len(outliers) / len(data_col)) * 100
            
            ax.set_title(f'{col}\nOutliers: {len(outliers)} ({outlier_pct:.1f}%)')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(analysis_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Outlier Analysis - Box Plots', fontsize=16)
        plt.tight_layout()
        
        self.save_plot(
            'outlier_analysis.png',
            'Outlier Analysis',
            'Box plots showing outlier detection for key numeric variables including the target variable.'
        )
    
    def plot_categorical_analysis(self, target):
        """Create analysis plots for categorical variables"""
        categorical_cols = self.data.select_dtypes(include=['object']).columns.tolist()
        
        if len(categorical_cols) == 0:
            return
        
        # Focus on columns with reasonable number of categories
        analysis_cols = []
        for col in categorical_cols:
            unique_count = self.data[col].nunique()
            if 2 <= unique_count <= 20:  # Reasonable range for visualization
                analysis_cols.append(col)
        
        if len(analysis_cols) == 0:
            return
        
        # Limit to top 6 for visualization
        analysis_cols = analysis_cols[:6]
        
        n_cols = min(2, len(analysis_cols))
        n_rows = (len(analysis_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 4*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1 or n_cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(analysis_cols):
            if i >= len(axes):
                break
                
            ax = axes[i]
            value_counts = self.data[col].value_counts().head(10)  # Top 10 categories
            
            # Create bar plot
            bars = ax.bar(range(len(value_counts)), value_counts.values, 
                         color='lightcoral', alpha=0.7, edgecolor='black')
            ax.set_xticks(range(len(value_counts)))
            ax.set_xticklabels(value_counts.index, rotation=45, ha='right')
            ax.set_title(f'{col} Distribution')
            ax.set_ylabel('Count')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, val in zip(bars, value_counts.values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(value_counts.values)*0.01,
                       f'{val}', ha='center', va='bottom', fontsize=9)
        
        # Hide unused subplots
        for i in range(len(analysis_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Categorical Variables Distribution', fontsize=16)
        plt.tight_layout()
        
        self.save_plot(
            'categorical_analysis.png',
            'Categorical Variables Analysis',
            'Distribution analysis of categorical variables showing frequency counts for top categories.'
        )
    
    def create_comprehensive_eda_plots(self):
        """Create all EDA plots for the selected research question"""
        if not self.selected_question or not self.output_dir:
            return False
        
        print("📊 Creating comprehensive EDA visualizations...")
        
        target = self.selected_question.target_variable
        plots_dir = self.setup_plotting_environment(self.output_dir)
        
        try:
            # 1. Target variable distribution
            print("   📈 Creating target distribution plots...")
            self.plot_target_distribution(target)
            
            # 2. Data quality overview
            print("   🔍 Creating data quality overview...")
            self.plot_data_quality_overview()
            
            # 3. Missing data analysis
            print("   🕳️  Creating missing data analysis...")
            self.plot_missing_data_analysis()
            
            # 4. Correlation analysis
            print("   🔗 Creating correlation analysis...")
            self.plot_correlation_heatmap(target)
            self.plot_target_correlations(target)
            
            # 5. Scatter plots
            print("   📊 Creating scatter plots...")
            self.plot_scatter_plots(target)
            
            # 6. Outlier analysis
            print("   🎯 Creating outlier analysis...")
            self.plot_outlier_analysis(target)
            
            # 7. Categorical analysis
            print("   📂 Creating categorical analysis...")
            self.plot_categorical_analysis(target)
            
            print(f"✅ Created {len(self.eda_plots_info)} EDA visualizations")
            print(f"   📁 Plots saved to: {plots_dir}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error creating EDA plots: {str(e)}")
            return False
    
    def generate_plots_summary_report(self):
        """Generate a summary report of all created plots"""
        if not self.eda_plots_info:
            return ""
        
        summary = []
        summary.append("EDA VISUALIZATIONS SUMMARY")
        summary.append("=" * 50)
        summary.append("")
        summary.append(f"📊 Total Plots Created: {len(self.eda_plots_info)}")
        summary.append(f"📁 Plots Directory: eda_plots/")
        summary.append("")
        
        summary.append("📈 PLOT INVENTORY:")
        summary.append("-" * 40)
        
        for i, plot_info in enumerate(self.eda_plots_info, 1):
            summary.append(f"{i:2d}. {plot_info['title']}")
            summary.append(f"    📄 File: {plot_info['filename']}")
            if plot_info['description']:
                summary.append(f"    📝 Description: {plot_info['description']}")
            summary.append("")
        
        summary.append("🎯 VISUALIZATION INSIGHTS:")
        summary.append("-" * 40)
        summary.append("These visualizations provide comprehensive insights into:")
        summary.append("• Target variable distribution and characteristics")
        summary.append("• Data quality and completeness assessment") 
        summary.append("• Correlation patterns and relationships")
        summary.append("• Outlier detection and analysis")
        summary.append("• Categorical variable distributions")
        summary.append("• Missing data patterns and impact")
        summary.append("")
        
        return "\n".join(summary)
    
    def conduct_exploratory_data_analysis(self):
        """Conduct comprehensive EDA focused on the research question"""
        if not self.selected_question:
            return False
        
        print(f"\n📊 Conducting EDA for: {self.selected_question.target_variable}")
        
        target = self.selected_question.target_variable
        target_data = self.data[target].dropna()
        
        # Basic statistics
        eda_results = {
            'target_stats': {
                'count': len(target_data),
                'unique_values': target_data.nunique(),
                'missing_rate': (len(self.data) - len(target_data)) / len(self.data) * 100
            }
        }
        
        # Numeric target analysis
        if pd.api.types.is_numeric_dtype(target_data):
            eda_results['target_stats'].update({
                'mean': target_data.mean(),
                'median': target_data.median(),
                'std': target_data.std(),
                'min': target_data.min(),
                'max': target_data.max(),
                'skewness': target_data.skew(),
                'kurtosis': target_data.kurtosis()
            })
            
            print(f"   Target Statistics:")
            print(f"     Mean: {target_data.mean():.4f}")
            print(f"     Std:  {target_data.std():.4f}")
            print(f"     Range: [{target_data.min():.2f}, {target_data.max():.2f}]")
        
        # Correlation analysis with numeric columns
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        if target in numeric_cols and len(numeric_cols) > 1:
            correlations = self.data[numeric_cols].corr()[target].abs().sort_values(ascending=False)
            correlations = correlations[correlations.index != target]
            
            if len(correlations) > 0:
                eda_results['top_correlations'] = correlations.head(10).to_dict()
                
                print(f"   Top correlations with {target}:")
                for var, corr in correlations.head(5).items():
                    print(f"     {var}: {corr:.3f}")
        
        # NEW: Create comprehensive EDA plots
        if self.output_dir:
            self.create_comprehensive_eda_plots()
        
        # Store EDA results
        self.research_findings['eda'] = eda_results
        
        return True
    
    def intelligent_feature_engineering(self, df, target_column):
        """Intelligent feature engineering that works with any dataset"""
        print("🔧 Intelligent feature engineering...")
        
        df_engineered = df.copy()
        
        # Identify and remove ID columns
        id_columns = []
        for col in df_engineered.columns:
            col_lower = col.lower()
            # More comprehensive ID detection
            if (any(id_term in col_lower for id_term in ['id', 'key', 'index', 'code']) and 
                col != target_column and
                df_engineered[col].nunique() > len(df_engineered) * 0.8):  # High cardinality
                id_columns.append(col)
        
        if id_columns:
            df_engineered = df_engineered.drop(columns=id_columns)
            print(f"   Removed {len(id_columns)} ID columns")
        
        # Handle datetime columns
        datetime_cols = df_engineered.select_dtypes(include=['datetime64']).columns
        for col in datetime_cols:
            if col != target_column:
                print(f"   Extracting features from datetime column: {col}")
                df_engineered[f'{col}_year'] = df_engineered[col].dt.year
                df_engineered[f'{col}_month'] = df_engineered[col].dt.month
                df_engineered[f'{col}_day'] = df_engineered[col].dt.day
                df_engineered[f'{col}_dayofweek'] = df_engineered[col].dt.dayofweek
                df_engineered = df_engineered.drop(columns=[col])
        
        # Create interaction features for small datasets
        numeric_cols = df_engineered.select_dtypes(include=[np.number]).columns.tolist()
        if target_column in numeric_cols:
            numeric_cols.remove(target_column)
        
        # For small number of numeric columns, create some interactions
        if len(numeric_cols) >= 2 and len(numeric_cols) <= 5:
            print(f"   Creating interaction features from {len(numeric_cols)} numeric columns")
            for i, col1 in enumerate(numeric_cols[:3]):
                for col2 in numeric_cols[i+1:4]:
                    try:
                        df_engineered[f'{col1}_x_{col2}'] = df_engineered[col1] * df_engineered[col2]
                    except:
                        pass
        
        print(f"   Feature engineering complete. New shape: {df_engineered.shape}")
        return df_engineered
    
    def universal_data_preprocessing(self, df, target_column):
        """Universal data preprocessing that handles any data type"""
        print("🧹 Universal data preprocessing...")
        
        # Start with feature engineering
        df_processed = self.intelligent_feature_engineering(df, target_column)
        
        # Handle missing values in target
        initial_rows = len(df_processed)
        df_processed = df_processed.dropna(subset=[target_column])
        final_rows = len(df_processed)
        
        if final_rows < initial_rows:
            print(f"   Removed {initial_rows - final_rows} rows with missing target values")
        
        if len(df_processed) < 30:
            print(f"❌ Insufficient data after cleaning: {len(df_processed)} rows")
            return None
        
        # Separate features and target
        feature_cols = [col for col in df_processed.columns if col != target_column]
        
        # Handle each feature column
        columns_to_drop = []
        
        for col in feature_cols:
            try:
                # Check missing value percentage
                missing_pct = df_processed[col].isna().sum() / len(df_processed)
                if missing_pct > 0.7:
                    columns_to_drop.append(col)
                    continue
                
                # Handle different data types
                if pd.api.types.is_numeric_dtype(df_processed[col]):
                    # Numeric column processing
                    df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                    df_processed[col] = df_processed[col].replace([np.inf, -np.inf], np.nan)
                    
                    # Fill missing values
                    if df_processed[col].isna().sum() > 0:
                        fill_value = df_processed[col].median()
                        if pd.isna(fill_value):
                            fill_value = 0
                        df_processed[col] = df_processed[col].fillna(fill_value)
                
                else:
                    # Categorical column processing
                    df_processed[col] = df_processed[col].astype(str).str.strip()
                    
                    # Handle missing values
                    df_processed[col] = df_processed[col].replace(['nan', 'NaN', 'None', 'null', ''], 'Unknown')
                    
                    unique_count = df_processed[col].nunique()
                    
                    if unique_count == 1:
                        # Constant column - remove
                        columns_to_drop.append(col)
                    elif unique_count == 2:
                        # Binary encoding
                        unique_vals = df_processed[col].unique()
                        df_processed[col] = (df_processed[col] == unique_vals[0]).astype(int)
                    elif unique_count <= 10:
                        # One-hot encoding
                        dummies = pd.get_dummies(df_processed[col], prefix=col, drop_first=True)
                        df_processed = pd.concat([df_processed, dummies], axis=1)
                        columns_to_drop.append(col)
                    elif unique_count <= 50:
                        # Label encoding
                        le = LabelEncoder()
                        df_processed[col] = le.fit_transform(df_processed[col])
                    else:
                        # Too many categories - remove
                        columns_to_drop.append(col)
                        
            except Exception as e:
                print(f"   Error processing {col}: {e}")
                columns_to_drop.append(col)
        
        # Drop problematic columns
        if columns_to_drop:
            df_processed = df_processed.drop(columns=columns_to_drop)
            print(f"   Dropped {len(columns_to_drop)} problematic columns")
        
        # Final validation - ensure all features are numeric
        feature_cols = [col for col in df_processed.columns if col != target_column]
        final_drops = []
        
        for col in feature_cols:
            if not pd.api.types.is_numeric_dtype(df_processed[col]):
                try:
                    df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                    df_processed[col] = df_processed[col].fillna(0)
                except:
                    final_drops.append(col)
        
        if final_drops:
            df_processed = df_processed.drop(columns=final_drops)
            print(f"   Final cleanup: dropped {len(final_drops)} non-convertible columns")
        
        # Ensure target is numeric for regression
        if not pd.api.types.is_numeric_dtype(df_processed[target_column]):
            print(f"   Converting target variable '{target_column}' to numeric")
            df_processed[target_column] = pd.to_numeric(df_processed[target_column], errors='coerce')
            df_processed = df_processed.dropna(subset=[target_column])
        
        print(f"✅ Preprocessing complete. Final shape: {df_processed.shape}")
        return df_processed
    
    def preprocess_for_research(self):
        """Preprocess data specifically for the research question"""
        if not self.selected_question:
            return False
        
        print("🔧 Preprocessing data for research analysis...")
        
        target = self.selected_question.target_variable
        
        # Universal preprocessing
        self.data_processed = self.universal_data_preprocessing(self.data, target)
        
        if self.data_processed is None:
            print("❌ Data preprocessing failed")
            return False
        
        # Verify we have enough features
        feature_count = len([col for col in self.data_processed.columns if col != target])
        if feature_count < 1:
            print("❌ No features available for modeling")
            return False
        
        print(f"✅ Preprocessing successful: {len(self.data_processed)} rows, {feature_count} features")
        return True
    
    def run_predictive_modeling(self):
        """Universal predictive modeling that works with any preprocessed dataset"""
        if not self.selected_question or not hasattr(self, 'data_processed'):
            return False
        
        print(f"🤖 Running predictive modeling for research question...")
        
        target = self.selected_question.target_variable
        feature_cols = [col for col in self.data_processed.columns if col != target]
        
        if not feature_cols:
            print("❌ No feature columns available")
            return False
        
        X = self.data_processed[feature_cols].copy()
        y = self.data_processed[target].copy()
        
        print(f"   Features: {len(feature_cols)} columns")
        print(f"   Samples: {len(X)} rows")
        
        # Final data validation and cleaning
        for col in X.columns:
            # Ensure numeric
            if not pd.api.types.is_numeric_dtype(X[col]):
                X[col] = pd.to_numeric(X[col], errors='coerce')
            
            # Handle infinite values
            X[col] = X[col].replace([np.inf, -np.inf], np.nan)
            
            # Fill missing values
            if X[col].isna().sum() > 0:
                fill_value = X[col].median()
                if pd.isna(fill_value):
                    fill_value = 0
                X[col] = X[col].fillna(fill_value)
        
        # Ensure target is clean
        if not pd.api.types.is_numeric_dtype(y):
            y = pd.to_numeric(y, errors='coerce')
        
        y = y.replace([np.inf, -np.inf], np.nan)
        
        # Remove any remaining invalid rows
        valid_mask = X.notna().all(axis=1) & y.notna() & np.isfinite(y)
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(X) < 30:
            print(f"❌ Insufficient clean data: {len(X)} rows")
            return False
        
        print(f"✅ Clean data ready: {X.shape[0]} rows × {X.shape[1]} features")
        
        # Split data
        test_size = min(0.3, max(0.1, 50 / len(X)))  # Adaptive test size
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define models with appropriate complexity for dataset size
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
        }
        
        # Add more complex models for larger datasets
        if len(X_train) > 100:
            models['Random Forest'] = RandomForestRegressor(
                n_estimators=min(50, max(10, len(X_train) // 10)), 
                random_state=42, 
                max_depth=min(8, max(3, len(feature_cols) // 2))
            )
        
        if len(X_train) > 200:
            models['Gradient Boosting'] = GradientBoostingRegressor(
                random_state=42, 
                max_depth=4, 
                n_estimators=min(100, max(50, len(X_train) // 5))
            )
        
        results = {}
        
        for name, model in models.items():
            try:
                print(f"   Training {name}...")
                
                # Use scaled data for linear models
                if 'Linear' in name or 'Ridge' in name:
                    model.fit(X_train_scaled, y_train)
                    y_pred_train = model.predict(X_train_scaled)
                    y_pred_test = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred_train = model.predict(X_train)
                    y_pred_test = model.predict(X_test)
                
                # Calculate metrics
                train_r2 = r2_score(y_train, y_pred_train)
                test_r2 = r2_score(y_test, y_pred_test)
                test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                test_mae = mean_absolute_error(y_test, y_pred_test)
                
                results[name] = {
                    'model': model,
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'test_rmse': test_rmse,
                    'test_mae': test_mae,
                    'y_pred_test': y_pred_test,
                    'y_pred_train': y_pred_train
                }
                
                # Feature importance
                if hasattr(model, 'feature_importances_'):
                    results[name]['feature_importance'] = dict(zip(feature_cols, model.feature_importances_))
                elif hasattr(model, 'coef_'):
                    results[name]['feature_importance'] = dict(zip(feature_cols, np.abs(model.coef_)))
                
                print(f"     ✅ R² = {test_r2:.4f}")
                
            except Exception as e:
                print(f"     ❌ {name} failed: {str(e)}")
                continue
        
        if not results:
            print("❌ All models failed")
            return False
        
        # Select best model
        best_name = max(results.keys(), key=lambda k: results[k]['test_r2'])
        self.best_model_name = best_name
        self.best_model = results[best_name]['model']
        self.best_model_score = results[best_name]['test_r2']
        self.y_pred_best = results[best_name]['y_pred_test']
        self.model_results = results
        
        print(f"🏆 Best Model: {best_name} (R² = {self.best_model_score:.4f})")
        return True
    
    def generate_research_insights(self):
        """Generate comprehensive research insights"""
        if not self.selected_question or not self.model_results:
            return False
        
        print("🧠 Generating research insights...")
        
        insights = []
        target = self.selected_question.target_variable
        
        insights.append(f"🔍 **Research Question**: {self.selected_question.question}")
        insights.append(f"📊 **Analysis Target**: {target}")
        
        # Model Performance
        best_r2 = self.best_model_score
        confidence = "High" if best_r2 > 0.7 else "Moderate" if best_r2 > 0.5 else "Limited"
        
        insights.append(f"🎯 **Predictive Power**: {confidence} (R² = {best_r2:.4f})")
        insights.append(f"   The model explains {best_r2*100:.1f}% of variance in {target}")
        
        # Key Findings
        best_result = self.model_results[self.best_model_name]
        if 'feature_importance' in best_result:
            sorted_features = sorted(
                best_result['feature_importance'].items(),
                key=lambda x: x[1], reverse=True
            )[:5]
            
            insights.append("🔑 **Key Factors Identified**:")
            for i, (feature, importance) in enumerate(sorted_features, 1):
                insights.append(f"   {i}. {feature} (Impact: {importance:.4f})")
        
        # Statistical insights
        if hasattr(self, 'research_findings') and 'eda' in self.research_findings:
            eda = self.research_findings['eda']
            if 'top_correlations' in eda:
                top_corr = list(eda['top_correlations'].items())[0]
                insights.append(f"📈 **Strongest Correlation**: {top_corr[0]} (r = {top_corr[1]:.3f})")
        
        # Business Recommendations
        recommendations = self.generate_business_recommendations()
        if recommendations:
            insights.append("🎯 **Business Recommendations**:")
            for i, rec in enumerate(recommendations, 1):
                insights.append(f"   {i}. {rec}")
        
        # Store insights
        self.selected_question.findings = insights
        self.research_findings['insights'] = insights
        
        return True
    
    def generate_business_recommendations(self):
        """Generate actionable business recommendations"""
        if not self.model_results or not self.selected_question:
            return []
        
        recommendations = []
        target = self.selected_question.target_variable
        target_lower = target.lower()
        
        # Get top predictive factors
        best_result = self.model_results[self.best_model_name]
        if 'feature_importance' in best_result:
            sorted_features = sorted(
                best_result['feature_importance'].items(),
                key=lambda x: x[1], reverse=True
            )[:3]
            
            top_factors = [feature for feature, _ in sorted_features]
            
            # Generate targeted recommendations based on target type
            if any(term in target_lower for term in ['price', 'cost', 'revenue', 'sales', 'salary']):
                recommendations.extend([
                    f"Focus on optimizing {top_factors[0]} to maximize {target}",
                    "Implement data-driven pricing strategies based on key predictive factors",
                    "Monitor and control the top 3 identified drivers regularly"
                ])
            elif any(term in target_lower for term in ['rating', 'score', 'satisfaction', 'quality']):
                recommendations.extend([
                    f"Prioritize improvements in {top_factors[0]} to enhance {target}",
                    "Establish quality monitoring systems for key performance drivers",
                    "Implement feedback loops to track improvement initiatives"
                ])
            elif any(term in target_lower for term in ['time', 'duration', 'length']):
                recommendations.extend([
                    f"Optimize {top_factors[0]} to reduce {target}",
                    "Implement process efficiency improvements targeting key factors",
                    "Establish performance benchmarks for time-related metrics"
                ])
            else:
                recommendations.extend([
                    f"Prioritize {top_factors[0]} optimization for {target} improvement",
                    "Develop monitoring systems for identified key performance indicators",
                    "Create action plans targeting the most influential factors"
                ])
        
        # Model performance-based recommendations
        if self.best_model_score > 0.7:
            recommendations.append("Deploy predictive model for operational decision-making")
        elif self.best_model_score > 0.4:
            recommendations.append("Use model insights for strategic planning with regular validation")
        else:
            recommendations.extend([
                "Focus on collecting additional relevant data to improve predictions",
                "Combine quantitative findings with domain expertise for decision-making"
            ])
        
        return recommendations[:6]  # Return top 6 recommendations
    
    def generate_detailed_data_overview(self):
        """Generate detailed data overview section"""
        overview = []
        
        overview.append("DETAILED DATA OVERVIEW")
        overview.append("=" * 50)
        
        # Original dataset information
        overview.append(f"📊 ORIGINAL DATASET")
        overview.append(f"   Rows: {len(self.original_data):,}")
        overview.append(f"   Columns: {len(self.original_data.columns)}")
        overview.append("")
        
        # Column type breakdown
        numeric_cols = self.original_data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.original_data.select_dtypes(include=['object']).columns.tolist()
        datetime_cols = self.original_data.select_dtypes(include=['datetime']).columns.tolist()
        
        overview.append(f"📈 COLUMN TYPE DISTRIBUTION")
        overview.append(f"   Numeric: {len(numeric_cols)} columns")
        overview.append(f"   Categorical: {len(categorical_cols)} columns")
        overview.append(f"   DateTime: {len(datetime_cols)} columns")
        overview.append("")
        
        # Missing values analysis
        missing_analysis = self.original_data.isnull().sum()
        missing_analysis = missing_analysis[missing_analysis > 0].sort_values(ascending=False)
        
        if len(missing_analysis) > 0:
            overview.append(f"🔍 MISSING VALUES ANALYSIS")
            for col, missing_count in missing_analysis.head(10).items():
                missing_pct = (missing_count / len(self.original_data)) * 100
                overview.append(f"   {col}: {missing_count:,} ({missing_pct:.1f}%)")
            overview.append("")
        
        # Data quality metrics
        overview.append(f"🎯 DATA QUALITY METRICS")
        total_cells = len(self.original_data) * len(self.original_data.columns)
        total_missing = self.original_data.isnull().sum().sum()
        completeness = ((total_cells - total_missing) / total_cells) * 100
        overview.append(f"   Overall Completeness: {completeness:.2f}%")
        
        # Duplicate analysis
        duplicates = self.original_data.duplicated().sum()
        overview.append(f"   Duplicate Rows: {duplicates:,} ({(duplicates/len(self.original_data))*100:.2f}%)")
        overview.append("")
        
        return "\n".join(overview)
    
    def generate_detailed_target_analysis(self):
        """Generate detailed target variable analysis"""
        target_analysis = []
        target = self.selected_question.target_variable
        target_data = self.original_data[target].dropna()
        
        target_analysis.append("DETAILED TARGET VARIABLE ANALYSIS")
        target_analysis.append("=" * 50)
        target_analysis.append(f"🎯 TARGET: {target}")
        target_analysis.append("")
        
        # Basic statistics
        if pd.api.types.is_numeric_dtype(target_data):
            target_analysis.append("📊 DESCRIPTIVE STATISTICS")
            target_analysis.append(f"   Count: {len(target_data):,}")
            target_analysis.append(f"   Mean: {target_data.mean():.4f}")
            target_analysis.append(f"   Median: {target_data.median():.4f}")
            target_analysis.append(f"   Standard Deviation: {target_data.std():.4f}")
            target_analysis.append(f"   Minimum: {target_data.min():.4f}")
            target_analysis.append(f"   Maximum: {target_data.max():.4f}")
            target_analysis.append(f"   Range: {target_data.max() - target_data.min():.4f}")
            target_analysis.append("")
            
            # Percentiles
            target_analysis.append("📈 PERCENTILE DISTRIBUTION")
            percentiles = [5, 10, 25, 50, 75, 90, 95]
            for p in percentiles:
                value = np.percentile(target_data, p)
                target_analysis.append(f"   {p}th percentile: {value:.4f}")
            target_analysis.append("")
            
            # Distribution characteristics
            target_analysis.append("📉 DISTRIBUTION CHARACTERISTICS")
            skewness = target_data.skew()
            kurtosis = target_data.kurtosis()
            cv = target_data.std() / target_data.mean() if target_data.mean() != 0 else 0
            
            target_analysis.append(f"   Skewness: {skewness:.4f} {'(Right-skewed)' if skewness > 0.5 else '(Left-skewed)' if skewness < -0.5 else '(Nearly symmetric)'}")
            target_analysis.append(f"   Kurtosis: {kurtosis:.4f} {'(Heavy-tailed)' if kurtosis > 0 else '(Light-tailed)'}")
            target_analysis.append(f"   Coefficient of Variation: {cv:.4f}")
            target_analysis.append("")
            
            # Outlier analysis
            Q1 = target_data.quantile(0.25)
            Q3 = target_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = target_data[(target_data < lower_bound) | (target_data > upper_bound)]
            
            target_analysis.append("🚨 OUTLIER ANALYSIS")
            target_analysis.append(f"   IQR Range: [{Q1:.4f}, {Q3:.4f}]")
            target_analysis.append(f"   Outlier Bounds: [{lower_bound:.4f}, {upper_bound:.4f}]")
            target_analysis.append(f"   Outliers Detected: {len(outliers)} ({(len(outliers)/len(target_data))*100:.2f}%)")
            target_analysis.append("")
            
        else:
            # Categorical target analysis
            target_analysis.append("📊 CATEGORICAL DISTRIBUTION")
            value_counts = target_data.value_counts()
            target_analysis.append(f"   Unique Categories: {len(value_counts)}")
            target_analysis.append(f"   Most Common: {value_counts.index[0]} ({value_counts.iloc[0]:,} instances)")
            
            target_analysis.append("\n   TOP CATEGORIES:")
            for cat, count in value_counts.head(10).items():
                percentage = (count / len(target_data)) * 100
                target_analysis.append(f"     {cat}: {count:,} ({percentage:.1f}%)")
            target_analysis.append("")
        
        return "\n".join(target_analysis)
    
    def generate_detailed_correlation_analysis(self):
        """Generate detailed correlation analysis"""
        corr_analysis = []
        target = self.selected_question.target_variable
        
        corr_analysis.append("DETAILED CORRELATION ANALYSIS")
        corr_analysis.append("=" * 50)
        
        # Numeric correlations
        numeric_cols = self.original_data.select_dtypes(include=[np.number]).columns.tolist()
        if target in numeric_cols and len(numeric_cols) > 1:
            correlations = self.original_data[numeric_cols].corr()[target].abs().sort_values(ascending=False)
            correlations = correlations[correlations.index != target]
            
            if len(correlations) > 0:
                corr_analysis.append("📈 NUMERIC VARIABLE CORRELATIONS")
                corr_analysis.append(f"   (Sorted by absolute correlation with {target})")
                corr_analysis.append("")
                
                # Strong correlations (>0.5)
                strong_corr = correlations[correlations > 0.5]
                if len(strong_corr) > 0:
                    corr_analysis.append("🔥 STRONG CORRELATIONS (|r| > 0.5)")
                    for var, corr in strong_corr.items():
                        direction = "positive" if self.original_data[numeric_cols].corr().loc[var, target] > 0 else "negative"
                        corr_analysis.append(f"   {var}: {corr:.4f} ({direction})")
                    corr_analysis.append("")
                
                # Moderate correlations (0.3-0.5)
                moderate_corr = correlations[(correlations >= 0.3) & (correlations <= 0.5)]
                if len(moderate_corr) > 0:
                    corr_analysis.append("⚡ MODERATE CORRELATIONS (0.3 ≤ |r| ≤ 0.5)")
                    for var, corr in moderate_corr.items():
                        direction = "positive" if self.original_data[numeric_cols].corr().loc[var, target] > 0 else "negative"
                        corr_analysis.append(f"   {var}: {corr:.4f} ({direction})")
                    corr_analysis.append("")
                
                # Top 10 correlations regardless of strength
                corr_analysis.append("📊 TOP 10 CORRELATIONS")
                for var, corr in correlations.head(10).items():
                    actual_corr = self.original_data[numeric_cols].corr().loc[var, target]
                    direction = "positive" if actual_corr > 0 else "negative"
                    corr_analysis.append(f"   {var}: {corr:.4f} ({direction})")
                corr_analysis.append("")
        
        return "\n".join(corr_analysis)
    
    def generate_detailed_model_analysis(self):
        """Generate detailed model analysis"""
        model_analysis = []
        
        model_analysis.append("DETAILED MODEL ANALYSIS")
        model_analysis.append("=" * 50)
        
        if not self.model_results:
            model_analysis.append("No model results available.")
            return "\n".join(model_analysis)
        
        # Best model details
        best_result = self.model_results[self.best_model_name]
        model_analysis.append(f"🏆 BEST MODEL: {self.best_model_name}")
        model_analysis.append("")
        
        # Performance metrics
        model_analysis.append("📊 PERFORMANCE METRICS")
        model_analysis.append(f"   R² Score (Test): {best_result['test_r2']:.6f}")
        model_analysis.append(f"   R² Score (Train): {best_result['train_r2']:.6f}")
        model_analysis.append(f"   RMSE: {best_result['test_rmse']:.6f}")
        model_analysis.append(f"   MAE: {best_result['test_mae']:.6f}")
        
        # Model interpretation
        r2 = best_result['test_r2']
        if r2 > 0.8:
            interpretation = "Excellent predictive power"
        elif r2 > 0.6:
            interpretation = "Good predictive power"
        elif r2 > 0.4:
            interpretation = "Moderate predictive power"
        elif r2 > 0.2:
            interpretation = "Limited predictive power"
        else:
            interpretation = "Poor predictive power"
        
        model_analysis.append(f"   Interpretation: {interpretation}")
        model_analysis.append(f"   Variance Explained: {r2*100:.2f}%")
        
        # Overfitting check
        train_test_diff = best_result['train_r2'] - best_result['test_r2']
        if train_test_diff > 0.1:
            model_analysis.append(f"   ⚠️  Potential Overfitting Detected (Δ = {train_test_diff:.4f})")
        else:
            model_analysis.append(f"   ✅ Good generalization (Δ = {train_test_diff:.4f})")
        model_analysis.append("")
        
        # Complete model comparison
        model_analysis.append("📈 COMPLETE MODEL COMPARISON")
        model_analysis.append(f"{'Model':<25} {'Train R²':<12} {'Test R²':<12} {'RMSE':<12} {'MAE':<12} {'Status':<15}")
        model_analysis.append("-" * 90)
        
        for name, result in sorted(self.model_results.items(), key=lambda x: x[1]['test_r2'], reverse=True):
            marker = "🏆 BEST" if name == self.best_model_name else ""
            train_r2 = result['train_r2']
            test_r2 = result['test_r2']
            rmse = result['test_rmse']
            mae = result['test_mae']
            
            model_analysis.append(f"{name:<25} {train_r2:<12.6f} {test_r2:<12.6f} {rmse:<12.6f} {mae:<12.6f} {marker:<15}")
        
        model_analysis.append("")
        
        # Feature importance analysis
        if 'feature_importance' in best_result:
            model_analysis.append("🔑 DETAILED FEATURE IMPORTANCE ANALYSIS")
            sorted_features = sorted(best_result['feature_importance'].items(), key=lambda x: x[1], reverse=True)
            
            # Top features
            model_analysis.append("\n   TOP 15 MOST IMPORTANT FEATURES:")
            model_analysis.append(f"   {'Rank':<6} {'Feature':<35} {'Importance':<12} {'Relative %':<12}")
            model_analysis.append("   " + "-" * 70)
            
            total_importance = sum(importance for _, importance in sorted_features)
            for i, (feature, importance) in enumerate(sorted_features[:15], 1):
                relative_pct = (importance / total_importance) * 100 if total_importance > 0 else 0
                model_analysis.append(f"   {i:<6} {feature:<35} {importance:<12.6f} {relative_pct:<12.2f}%")
            
            # Feature importance distribution
            model_analysis.append(f"\n   📊 IMPORTANCE DISTRIBUTION:")
            top5_importance = sum(importance for _, importance in sorted_features[:5])
            top10_importance = sum(importance for _, importance in sorted_features[:10])
            top5_pct = (top5_importance / total_importance) * 100 if total_importance > 0 else 0
            top10_pct = (top10_importance / total_importance) * 100 if total_importance > 0 else 0
            
            model_analysis.append(f"   Top 5 features: {top5_pct:.1f}% of total importance")
            model_analysis.append(f"   Top 10 features: {top10_pct:.1f}% of total importance")
            model_analysis.append("")
        
        return "\n".join(model_analysis)
    
    def generate_detailed_recommendations(self):
        """Generate detailed business recommendations with priorities"""
        recommendations = []
        
        recommendations.append("DETAILED BUSINESS RECOMMENDATIONS")
        recommendations.append("=" * 50)
        
        # Get recommendations
        recs = self.generate_business_recommendations()
        
        # Prioritize recommendations
        recommendations.append("🎯 PRIORITY RECOMMENDATIONS")
        recommendations.append("")
        
        priorities = ["HIGH PRIORITY", "MEDIUM PRIORITY", "LOWER PRIORITY"]
        recs_per_priority = len(recs) // 3 + (1 if len(recs) % 3 > 0 else 0)
        
        for i, priority in enumerate(priorities):
            start_idx = i * recs_per_priority
            end_idx = min((i + 1) * recs_per_priority, len(recs))
            
            if start_idx < len(recs):
                recommendations.append(f"🔥 {priority}")
                for j, rec in enumerate(recs[start_idx:end_idx], 1):
                    recommendations.append(f"   {start_idx + j}. {rec}")
                recommendations.append("")
        
        # Implementation roadmap
        recommendations.append("🛠️  IMPLEMENTATION ROADMAP")
        recommendations.append("")
        
        if self.best_model_score > 0.6:
            recommendations.append("📅 SHORT-TERM (1-3 months):")
            recommendations.append("   • Implement top 3 predictive factors in operations")
            recommendations.append("   • Set up monitoring dashboards for key metrics")
            recommendations.append("   • Begin data collection improvements")
            recommendations.append("")
            
            recommendations.append("📅 MEDIUM-TERM (3-6 months):")
            recommendations.append("   • Deploy predictive model in production environment")
            recommendations.append("   • Train staff on new data-driven processes")
            recommendations.append("   • Establish feedback loops for continuous improvement")
            recommendations.append("")
            
            recommendations.append("📅 LONG-TERM (6+ months):")
            recommendations.append("   • Scale successful interventions across organization")
            recommendations.append("   • Develop advanced analytics capabilities")
            recommendations.append("   • Regular model retraining and validation")
            recommendations.append("")
        else:
            recommendations.append("📅 IMMEDIATE ACTIONS:")
            recommendations.append("   • Improve data quality and collection processes")
            recommendations.append("   • Focus on domain expertise and qualitative analysis")
            recommendations.append("   • Identify additional relevant data sources")
            recommendations.append("")
        
        # Risk assessment
        recommendations.append("⚠️  RISK ASSESSMENT & MITIGATION")
        recommendations.append("")
        
        if self.best_model_score < 0.4:
            recommendations.append("🚨 HIGH RISK: Low model performance")
            recommendations.append("   • Rely heavily on domain expertise")
            recommendations.append("   • Implement gradual, monitored changes")
            recommendations.append("   • Collect more relevant data before major decisions")
        elif self.best_model_score < 0.7:
            recommendations.append("⚡ MEDIUM RISK: Moderate model performance")
            recommendations.append("   • Validate recommendations with pilot programs")
            recommendations.append("   • Monitor outcomes closely")
            recommendations.append("   • Maintain human oversight in decision-making")
        else:
            recommendations.append("✅ LOW RISK: High model performance")
            recommendations.append("   • Proceed with confidence in model recommendations")
            recommendations.append("   • Implement regular model validation")
            recommendations.append("   • Scale successful strategies")
        
        recommendations.append("")
        
        return "\n".join(recommendations)
    
    def generate_research_report(self):
        """Generate comprehensive and detailed research report"""
        if not self.selected_question:
            return "No research question selected."
        
        report = []
        
        # Header
        report.append("=" * 100)
        report.append("UNIVERSAL AUTONOMOUS RESEARCH ANALYTICS REPORT")
        report.append("DETAILED COMPREHENSIVE ANALYSIS WITH EDA VISUALIZATIONS")
        report.append("=" * 100)
        report.append("")
        report.append(f"📁 Dataset: {self.filename}")
        report.append(f"🕐 Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"🔬 Engine Version: Universal Analytics v2.0 Enhanced")
        report.append("")
        
        # Executive summary
        report.append("EXECUTIVE SUMMARY")
        report.append("=" * 50)
        report.append(f"🎯 Research Focus: {self.selected_question.question}")
        report.append(f"📊 Target Variable: {self.selected_question.target_variable}")
        if self.model_results:
            best_r2 = self.best_model_score
            summary_confidence = "High confidence" if best_r2 > 0.7 else "Moderate confidence" if best_r2 > 0.4 else "Limited confidence"
            report.append(f"🏆 Best Model: {self.best_model_name} (R² = {best_r2:.4f})")
            report.append(f"💡 Analysis Confidence: {summary_confidence}")
        
        # NEW: EDA Visualizations Summary
        if self.eda_plots_info:
            report.append(f"📊 EDA Visualizations: {len(self.eda_plots_info)} comprehensive plots created")
        
        report.append("")
        
        # Research question section
        report.append("RESEARCH FRAMEWORK")
        report.append("=" * 50)
        report.append(f"🔍 Primary Question: {self.selected_question.question}")
        report.append(f"🧪 Research Hypothesis: {self.selected_question.hypothesis}")
        report.append(f"📈 Expected Business Impact: {self.selected_question.business_impact}")
        report.append(f"🔬 Methodology: {self.selected_question.methodology}")
        report.append("")
        
        # NEW: EDA Visualizations Section
        if self.eda_plots_info:
            report.append(self.generate_plots_summary_report())
            report.append("")
        
        # Detailed data overview
        report.append(self.generate_detailed_data_overview())
        report.append("")
        
        # Detailed target analysis
        report.append(self.generate_detailed_target_analysis())
        report.append("")
        
        # Correlation analysis
        report.append(self.generate_detailed_correlation_analysis())
        report.append("")
        
        # Model analysis
        report.append(self.generate_detailed_model_analysis())
        report.append("")
        
        # Key findings with detailed insights
        if hasattr(self.selected_question, 'findings') and self.selected_question.findings:
            report.append("KEY RESEARCH FINDINGS & INSIGHTS")
            report.append("=" * 50)
            for finding in self.selected_question.findings:
                report.append(finding)
            report.append("")
        
        # Detailed recommendations
        report.append(self.generate_detailed_recommendations())
        report.append("")
        
        # Alternative research directions
        if len(self.research_questions) > 1:
            report.append("ALTERNATIVE RESEARCH OPPORTUNITIES")
            report.append("=" * 50)
            report.append("🔄 Additional research questions identified during analysis:")
            report.append("")
            for i, q in enumerate(self.research_questions[1:6], 2):
                report.append(f"{i}. {q.question}")
                report.append(f"   🎯 Target: {q.target_variable}")
                report.append(f"   💡 Hypothesis: {q.hypothesis}")
                report.append(f"   🎪 Business Impact: {q.business_impact}")
                report.append("")
        
        # Technical appendix
        report.append("TECHNICAL APPENDIX")
        report.append("=" * 50)
        
        if hasattr(self, 'data_processed'):
            report.append("🔧 DATA PROCESSING PIPELINE:")
            report.append(f"   • Original dataset: {len(self.original_data)} rows × {len(self.original_data.columns)} columns")
            report.append(f"   • Final processed data: {len(self.data_processed)} rows × {len(self.data_processed.columns)} columns")
            data_retention = (len(self.data_processed) / len(self.original_data)) * 100
            report.append(f"   • Data retention rate: {data_retention:.1f}%")
            feature_count = len([col for col in self.data_processed.columns if col != self.selected_question.target_variable])
            report.append(f"   • Features used for modeling: {feature_count}")
            report.append("")
        
        if hasattr(self, 'X_train') and self.X_train is not None:
            report.append("🎲 MODEL TRAINING SETUP:")
            report.append(f"   • Training samples: {len(self.X_train):,}")
            report.append(f"   • Testing samples: {len(self.X_test):,}")
            test_ratio = len(self.X_test) / (len(self.X_train) + len(self.X_test))
            report.append(f"   • Test split ratio: {test_ratio:.2f}")
            report.append("")
        
        # NEW: EDA Technical Details
        if self.eda_plots_info:
            report.append("📊 EDA VISUALIZATION PIPELINE:")
            report.append(f"   • Total visualizations created: {len(self.eda_plots_info)}")
            report.append(f"   • Plot formats: High-resolution PNG (300 DPI)")
            report.append(f"   • Visualization types: Distribution, correlation, scatter, outlier, categorical analysis")
            report.append(f"   • Output directory: eda_plots/")
            report.append("")
        
        report.append("⚖️  MODEL VALIDATION:")
        report.append("   • Cross-validation approach: Train/Test split")
        report.append("   • Evaluation metrics: R², RMSE, MAE")
        report.append("   • Feature scaling: StandardScaler for linear models")
        report.append("   • Overfitting prevention: Regularization and complexity control")
        report.append("")
        
        # Disclaimers and limitations
        report.append("LIMITATIONS & DISCLAIMERS")
        report.append("=" * 50)
        report.append("⚠️  Please consider the following limitations:")
        report.append("")
        report.append("📊 DATA LIMITATIONS:")
        if hasattr(self, 'data_processed'):
            missing_info = self.original_data.isnull().sum().sum()
            if missing_info > 0:
                report.append("   • Missing data may impact model accuracy")
            data_size = len(self.data_processed)
            if data_size < 1000:
                report.append("   • Limited sample size may affect generalizability")
        
        report.append("")
        report.append("🤖 MODEL LIMITATIONS:")
        if self.best_model_score < 0.5:
            report.append("   • Low predictive performance suggests complex underlying relationships")
        report.append("   • Correlation does not imply causation")
        report.append("   • Model assumes relationships remain stable over time")
        report.append("   • External factors not in dataset may influence outcomes")
        report.append("")
        
        report.append("💼 BUSINESS CONSIDERATIONS:")
        report.append("   • Recommendations should be validated with domain expertise")
        report.append("   • Implement changes gradually and monitor outcomes")
        report.append("   • Regular model retraining recommended as new data becomes available")
        report.append("   • Consider regulatory, ethical, and operational constraints")
        report.append("")
        
        # Footer
        report.append("=" * 100)
        report.append("END OF ENHANCED RESEARCH REPORT WITH EDA VISUALIZATIONS")
        report.append(f"Generated by Universal Autonomous Research Analytics Engine v2.0")
        report.append(f"Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 100)
        
        return "\n".join(report)
    
    def run_universal_autonomous_research(self):
        """Run the complete universal autonomous research analysis pipeline"""
        print("🚀 Starting Universal Autonomous Research Analytics...")
        print("=" * 80)
        
        try:
            # Step 1: Formulate research questions
            if not self.formulate_universal_research_questions():
                return {"success": False, "error": "Failed to formulate research questions"}
            
            # Step 2: Select primary research question
            if not self.select_primary_research_question():
                return {"success": False, "error": "Failed to select research question"}
            
            # Step 3: Conduct exploratory data analysis (now includes plotting)
            if not self.conduct_exploratory_data_analysis():
                return {"success": False, "error": "Failed to conduct EDA"}
            
            # Step 4: Preprocess data for research
            if not self.preprocess_for_research():
                return {"success": False, "error": "Failed to preprocess data"}
            
            # Step 5: Run predictive modeling
            if not self.run_predictive_modeling():
                return {"success": False, "error": "Failed to run predictive models"}
            
            # Step 6: Generate research insights
            if not self.generate_research_insights():
                return {"success": False, "error": "Failed to generate insights"}
            
            # Step 7: Generate report
            report_content = self.generate_research_report()
            
            return {
                "success": True,
                "research_question": self.selected_question.question,
                "target_variable": self.selected_question.target_variable,
                "hypothesis": self.selected_question.hypothesis,
                "best_model": self.best_model_name,
                "model_performance": self.best_model_score,
                "key_findings": self.selected_question.findings if hasattr(self.selected_question, 'findings') else [],
                "recommendations": self.generate_business_recommendations(),
                "report_content": report_content,
                "all_questions": [q.question for q in self.research_questions],
                "eda_plots": self.eda_plots_info  # NEW: Include plot information
            }
        
        except Exception as e:
            print(f"❌ Analysis failed with error: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}

    def generate_executive_dashboard(self):
        """Generate an executive dashboard summary"""
        dashboard = []
        
        dashboard.append("EXECUTIVE DASHBOARD")
        dashboard.append("=" * 50)
        dashboard.append("")
        
        # Key metrics at a glance
        dashboard.append("📊 KEY METRICS AT A GLANCE")
        dashboard.append("┌" + "─" * 48 + "┐")
        
        if self.model_results:
            r2_score = self.best_model_score
            confidence_level = "🟢 HIGH" if r2_score > 0.7 else "🟡 MEDIUM" if r2_score > 0.4 else "🔴 LIMITED"
            dashboard.append(f"│ Model Confidence: {confidence_level:<28} │")
            dashboard.append(f"│ Predictive Power: {r2_score:.2%} variance explained{' '*8} │")
            dashboard.append(f"│ Best Algorithm: {self.best_model_name:<31} │")
        
        if hasattr(self, 'data_processed'):
            data_quality = (len(self.data_processed) / len(self.original_data)) * 100
            dashboard.append(f"│ Data Quality: {data_quality:.1f}% usable data{' '*16} │")
        
        # NEW: Add EDA visualization info
        if self.eda_plots_info:
            dashboard.append(f"│ EDA Plots: {len(self.eda_plots_info)} visualizations created{' '*11} │")
            
        dashboard.append("└" + "─" * 48 + "┘")
        dashboard.append("")
        
        # Risk assessment
        if self.model_results:
            dashboard.append("⚠️  RISK ASSESSMENT")
            if self.best_model_score > 0.7:
                dashboard.append("🟢 LOW RISK - High confidence in findings")
                dashboard.append("   → Proceed with implementing recommendations")
            elif self.best_model_score > 0.4:
                dashboard.append("🟡 MEDIUM RISK - Moderate confidence in findings") 
                dashboard.append("   → Validate with pilot programs before full implementation")
            else:
                dashboard.append("🔴 HIGH RISK - Limited predictive power")
                dashboard.append("   → Use insights for exploration, validate with domain expertise")
            dashboard.append("")
        
        # Top 3 actionable insights
        dashboard.append("🎯 TOP 3 ACTIONABLE INSIGHTS")
        recs = self.generate_business_recommendations()
        for i, rec in enumerate(recs[:3], 1):
            dashboard.append(f"{i}. {rec}")
        dashboard.append("")
        
        # NEW: EDA Insights Summary
        if self.eda_plots_info:
            dashboard.append("📊 VISUAL INSIGHTS AVAILABLE")
            dashboard.append("   • Target variable distribution analysis")
            dashboard.append("   • Correlation patterns and relationships")  
            dashboard.append("   • Data quality and missing value assessment")
            dashboard.append("   • Outlier detection and analysis")
            dashboard.append("   • Categorical variable distributions")
            dashboard.append(f"   📁 View {len(self.eda_plots_info)} plots in: eda_plots/")
            dashboard.append("")
        
        return "\n".join(dashboard)

def analyze_with_universal_engine(filepath, output_dir="research_output"):
    """Analyze a dataset with the universal autonomous research engine"""
    
    print("🧠 Universal Autonomous Research Analytics Engine v2.0 Enhanced")
    print("=" * 70)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize universal engine
    engine = UniversalAutonomousResearchAnalytics()
    
    # NEW: Set output directory for plotting
    engine.output_dir = output_dir
    
    # Load data
    if not engine.load_data(filepath):
        return False
    
    # Run universal autonomous research analysis
    results = engine.run_universal_autonomous_research()
    
    if not results["success"]:
        print(f"❌ Research analysis failed: {results['error']}")
        return False
    
    # Save results
    print("\n💾 Saving Comprehensive Research Results...")
    
    # Save detailed comprehensive report
    report_file = os.path.join(output_dir, "detailed_research_report.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(results["report_content"])
    print(f"📄 Detailed research report saved: {report_file}")
    
    # Generate and save executive dashboard
    executive_dashboard = engine.generate_executive_dashboard()
    dashboard_file = os.path.join(output_dir, "executive_dashboard.txt")
    with open(dashboard_file, 'w', encoding='utf-8') as f:
        f.write(executive_dashboard)
    print(f"📊 Executive dashboard saved: {dashboard_file}")
    
    # Save comprehensive results as JSON
    json_file = os.path.join(output_dir, "comprehensive_results.json")
    json_data = {
        "metadata": {
            "filename": engine.filename,
            "analysis_date": datetime.now().isoformat(),
            "engine_version": "Universal Analytics v2.0 Enhanced with EDA"
        },
        "research_framework": {
            "primary_question": results["research_question"],
            "target_variable": results["target_variable"],
            "hypothesis": results["hypothesis"],
            "methodology": results.get("methodology", "Predictive modeling with feature importance analysis"),
            "business_impact": results.get("business_impact", "")
        },
        "model_performance": {
            "best_model": results["best_model"],
            "r2_score": results["model_performance"],
            "confidence_level": "High" if results["model_performance"] > 0.7 else "Moderate" if results["model_performance"] > 0.4 else "Limited"
        },
        "key_findings": results["key_findings"],
        "business_recommendations": {
            "high_priority": results["recommendations"][:2] if len(results["recommendations"]) >= 2 else results["recommendations"],
            "medium_priority": results["recommendations"][2:4] if len(results["recommendations"]) > 2 else [],
            "all_recommendations": results["recommendations"]
        },
        "alternative_research_opportunities": results["all_questions"][1:] if len(results["all_questions"]) > 1 else [],
        "data_quality": {
            "original_rows": len(engine.original_data) if hasattr(engine, 'original_data') else 0,
            "processed_rows": len(engine.data_processed) if hasattr(engine, 'data_processed') else 0,
            "retention_rate": (len(engine.data_processed) / len(engine.original_data)) * 100 if hasattr(engine, 'data_processed') and hasattr(engine, 'original_data') else 0
        },
        # NEW: EDA visualization information
        "eda_visualizations": {
            "total_plots": len(results.get("eda_plots", [])),
            "plots_directory": "eda_plots/",
            "plot_details": results.get("eda_plots", [])
        }
    }
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2)
    print(f"📈 Comprehensive JSON results saved: {json_file}")
    
    # Generate summary CSV for quick analysis
    if engine.model_results and 'feature_importance' in engine.model_results[engine.best_model_name]:
        feature_importance_df = pd.DataFrame([
            {'feature': feature, 'importance': importance, 'rank': i+1}
            for i, (feature, importance) in enumerate(
                sorted(engine.model_results[engine.best_model_name]['feature_importance'].items(),
                      key=lambda x: x[1], reverse=True)
            )
        ])
        
        csv_file = os.path.join(output_dir, "feature_importance.csv")
        feature_importance_df.to_csv(csv_file, index=False)
        print(f"📋 Feature importance CSV saved: {csv_file}")
    
    # NEW: Save EDA plots summary CSV
    if results.get("eda_plots"):
        eda_df = pd.DataFrame(results["eda_plots"])
        eda_csv_file = os.path.join(output_dir, "eda_plots_summary.csv")
        eda_df.to_csv(eda_csv_file, index=False)
        print(f"📊 EDA plots summary saved: {eda_csv_file}")
    
    # Display enhanced summary
    print("\n" + "="*80)
    print("✅ COMPREHENSIVE AUTONOMOUS RESEARCH ANALYSIS COMPLETE!")
    print("="*80)
    
    # Executive summary
    confidence_emoji = "🟢" if results["model_performance"] > 0.7 else "🟡" if results["model_performance"] > 0.4 else "🔴"
    print(f"\n📊 EXECUTIVE SUMMARY:")
    print(f"   {confidence_emoji} Analysis Confidence: {results['model_performance']:.2%} predictive accuracy")
    print(f"   🎯 Research Focus: {results['research_question']}")
    print(f"   📈 Target Variable: {results['target_variable']}")
    print(f"   🏆 Best Model: {results['best_model']}")
    
    print(f"\n🔍 DETAILED ANALYSIS:")
    print(f"   📄 Generated {len(results['key_findings'])} key findings")
    print(f"   💼 Provided {len(results['recommendations'])} business recommendations")
    print(f"   🔄 Identified {len(results['all_questions']) - 1} alternative research opportunities")
    
    # NEW: EDA visualization summary
    if results.get("eda_plots"):
        print(f"   📊 Created {len(results['eda_plots'])} EDA visualizations")
        print(f"   🎨 Visual analysis includes: distribution, correlation, outlier, and categorical analysis")
    
    if hasattr(engine, 'data_processed') and hasattr(engine, 'original_data'):
        data_retention = (len(engine.data_processed) / len(engine.original_data)) * 100
        print(f"   📊 Data quality: {data_retention:.1f}% retention rate")
    
    print(f"\n💾 OUTPUT FILES:")
    print(f"   📋 Detailed Report: detailed_research_report.txt")
    print(f"   📊 Executive Dashboard: executive_dashboard.txt") 
    print(f"   📈 Comprehensive Data: comprehensive_results.json")
    if engine.model_results:
        print(f"   📋 Feature Analysis: feature_importance.csv")
    
    # NEW: EDA files summary
    if results.get("eda_plots"):
        print(f"   🎨 EDA Visualizations: {len(results['eda_plots'])} plots in eda_plots/")
        print(f"   📊 EDA Summary: eda_plots_summary.csv")
    
    print(f"   📁 All files saved to: {output_dir}/")
    
    # Quick recommendations preview
    print(f"\n🎯 TOP RECOMMENDATIONS PREVIEW:")
    for i, rec in enumerate(results['recommendations'][:3], 1):
        print(f"   {i}. {rec}")
    
    # NEW: EDA insights preview
    if results.get("eda_plots"):
        print(f"\n📊 VISUAL INSIGHTS CREATED:")
        plot_types = set()
        for plot in results['eda_plots']:
            if 'distribution' in plot['filename'].lower():
                plot_types.add("📈 Target distribution analysis")
            elif 'correlation' in plot['filename'].lower():
                plot_types.add("🔗 Correlation analysis")
            elif 'scatter' in plot['filename'].lower():
                plot_types.add("📊 Relationship scatter plots")
            elif 'outlier' in plot['filename'].lower():
                plot_types.add("🎯 Outlier detection")
            elif 'categorical' in plot['filename'].lower():
                plot_types.add("📂 Categorical analysis")
            elif 'quality' in plot['filename'].lower():
                plot_types.add("🔍 Data quality overview")
        
        for plot_type in sorted(plot_types):
            print(f"   {plot_type}")
    
    print("\n" + "="*80)
    
    return True


def main():
    """Main function for the universal script"""
    parser = argparse.ArgumentParser(
        description='Universal Autonomous Research Analytics Engine with Enhanced EDA Visualizations',
        epilog="""
This enhanced universal version works with ANY dataset by:
  • Intelligently identifying target variables
  • Automatically handling any data type
  • Generating relevant research questions
  • Creating comprehensive EDA visualizations
  • Providing actionable business insights
        """
    )
    
    parser.add_argument('filepath', help='Path to CSV or Excel file')
    parser.add_argument('--output', '-o', default='research_output', 
                       help='Output directory (default: research_output)')
    
    args = parser.parse_args()
    
    # Validate file
    if not os.path.exists(args.filepath):
        print(f"❌ File not found: {args.filepath}")
        return 1
    
    # Run universal analysis
    success = analyze_with_universal_engine(args.filepath, args.output)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
