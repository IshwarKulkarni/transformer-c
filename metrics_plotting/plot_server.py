#!/usr/bin/env python3
"""
Interactive Web Server for Accuracy CSV Files
Reads all accuracy*.csv files and creates zoomable, interactive plots with hover tooltips
Written entirely by Cursor AI
"""

import os
import glob
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from flask import Flask, render_template_string
import json
import re
from datetime import datetime
from typing import cast

app = Flask(__name__)

file_patterns = ["runs/*/training_metrics*.csv"]

def find_accuracy_files():
    """Find all accuracy*.c1sv files in the current directory and subdirectories"""
    files = []
    for pattern in file_patterns:
        files.extend(glob.glob(pattern))
    
    if len(files) == 0:
        print("No training_metrics.csv files found")
        print("Looked in the following directories:")
        for pattern in file_patterns: print(f"   - {pattern}")
        return []
    

    files = [f for f in set(files) if "/latest/" not in f]
    print(f"Found {len(files)} training_metrics.csv files")
    for file in files:
        print(f"   - {file}")
    # files = list(set(files))
    # sort by creation time
    files.sort(key=lambda x: extract_datetime_from_filename(x), reverse=True)
    return files

def extract_datetime_from_filename(filename):
    """Extract creation time from file and format it nicely"""
    try:
        ctime = os.path.getctime(filename)
        dt = datetime.fromtimestamp(ctime)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return "Unknown Date"

def create_interactive_plots(csv_files):
    """Create interactive plots for all CSV files"""
    if not csv_files:
        return []
    
    # Read all CSV files first
    all_dataframes = []
    for csv_file in csv_files:
        try:
            df = cast(pd.DataFrame, pd.read_csv(csv_file))
            all_dataframes.append({
                'filename': csv_file,
                'df': df,
                'datetime': extract_datetime_from_filename(csv_file)
            })
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
            continue
    
    if not all_dataframes:
        return []
    
    plots_data = []
    
    # Create main comparison plot with latest vs selected
    if len(all_dataframes) >= 1:
        main_figures = {}
        
        # Store all dataframes for comparison
        comparison_data = {
            'latest': all_dataframes[0],
            'others': all_dataframes[1:] if len(all_dataframes) > 1 else []
        }
        
        # Figure 1: Accuracy comparison
        accuracy_exists = any('train_acc' in data['df'].columns or 'valdn_acc' in data['df'].columns 
                             for data in all_dataframes[:2])
        
        if accuracy_exists:
            fig_acc = go.Figure()
            
            # Add latest data (always first)
            latest_df = all_dataframes[0]['df']
            x_col = 'batches' if 'batches' in latest_df.columns else 'epoch'
            x_data = latest_df[x_col].tolist()
            
            if 'train_acc' in latest_df.columns:
                fig_acc.add_trace(
                    go.Scatter(
                        x=x_data, y=latest_df['train_acc'].tolist(),
                        mode='lines+markers',
                        name='Train Accuracy (Latest)',
                        line=dict(color='blue', width=2, dash='solid'),
                        marker=dict(size=4),
                        opacity=1.0,
                        hovertemplate=f'<b>Train Accuracy (Latest)</b><br>' +
                                    f'{x_col.capitalize()}: %{{x}}<br>' +
                                    'Accuracy: %{y:.2f}%<br>' +
                                    '<extra></extra>'
                    )
                )
            
            if 'valdn_acc' in latest_df.columns:
                fig_acc.add_trace(
                    go.Scatter(
                        x=x_data, y=latest_df['valdn_acc'].tolist(),
                        mode='lines+markers',
                        name='Validation Accuracy (Latest)',
                        line=dict(color='red', width=2, dash='solid'),
                        marker=dict(size=4),
                        opacity=1.0,
                        hovertemplate=f'<b>Validation Accuracy (Latest)</b><br>' +
                                    f'{x_col.capitalize()}: %{{x}}<br>' +
                                    'Accuracy: %{y:.2f}%<br>' +
                                    '<extra></extra>'
                    )
                )
            
            # Add comparison data (initially the second file, but will be changeable)
            if len(all_dataframes) > 1:
                comp_df = all_dataframes[1]['df']
                comp_x_data = comp_df[x_col].tolist()
                
                if 'train_acc' in comp_df.columns:
                    fig_acc.add_trace(
                        go.Scatter(
                            x=comp_x_data, y=comp_df['train_acc'].tolist(),
                            mode='lines+markers',
                            name='Train Accuracy (Comparison)',
                            line=dict(color='blue', width=1, dash='dot'),
                            marker=dict(size=2),
                            opacity=0.5,
                            hovertemplate=f'<b>Train Accuracy (Comparison)</b><br>' +
                                        f'{x_col.capitalize()}: %{{x}}<br>' +
                                        'Accuracy: %{y:.2f}%<br>' +
                                        '<extra></extra>'
                        )
                    )
                
                if 'valdn_acc' in comp_df.columns:
                    fig_acc.add_trace(
                        go.Scatter(
                            x=comp_x_data, y=comp_df['valdn_acc'].tolist(),
                            mode='lines+markers',
                            name='Validation Accuracy (Comparison)',
                            line=dict(color='red', width=1, dash='dot'),
                            marker=dict(size=2),
                            opacity=0.5,
                            hovertemplate=f'<b>Validation Accuracy (Comparison)</b><br>' +
                                        f'{x_col.capitalize()}: %{{x}}<br>' +
                                        'Accuracy: %{y:.2f}%<br>' +
                                        '<extra></extra>'
                        )
                    )
            
            fig_acc.update_layout(
                title='Training & Validation Accuracy Comparison',
                xaxis_title=x_col.capitalize(),
                yaxis_title='Accuracy (%)',
                height=400,
                showlegend=True,
                hovermode='x unified',
                template='plotly_white'
            )
            
            # Add grid to axes
            fig_acc.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
            fig_acc.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
            
            main_figures['accuracy'] = fig_acc
        
        # Figure 2: Loss comparison
        loss_exists = any('train_loss' in data['df'].columns or 'valdn_loss' in data['df'].columns 
                         for data in all_dataframes[:2])
        
        if loss_exists:
            fig_loss = go.Figure()
            
            # Add latest data
            latest_df = all_dataframes[0]['df']
            x_col = 'batches' if 'batches' in latest_df.columns else 'epoch'
            x_data = latest_df[x_col].tolist()
            
            if 'train_loss' in latest_df.columns:
                fig_loss.add_trace(
                    go.Scatter(
                        x=x_data, y=latest_df['train_loss'].tolist(),
                        mode='lines+markers',
                        name='Train Loss (Latest)',
                        line=dict(color='green', width=2, dash='solid'),
                        marker=dict(size=4),
                        opacity=1.0,
                        hovertemplate=f'<b>Train Loss (Latest)</b><br>' +
                                    f'{x_col.capitalize()}: %{{x}}<br>' +
                                    'Loss: %{y:.4f}<br>' +
                                    '<extra></extra>'
                    )
                )
            
            if 'valdn_loss' in latest_df.columns:
                fig_loss.add_trace(
                    go.Scatter(
                        x=x_data, y=latest_df['valdn_loss'].tolist(),
                        mode='lines+markers',
                        name='Validation Loss (Latest)',
                        line=dict(color='orange', width=2, dash='solid'),
                        marker=dict(size=4),
                        opacity=1.0,
                        hovertemplate=f'<b>Validation Loss (Latest)</b><br>' +
                                    f'{x_col.capitalize()}: %{{x}}<br>' +
                                    'Loss: %{y:.4f}<br>' +
                                    '<extra></extra>'
                    )
                )
            
            # Add comparison data
            if len(all_dataframes) > 1:
                comp_df = all_dataframes[1]['df']
                comp_x_data = comp_df[x_col].tolist()
                
                if 'train_loss' in comp_df.columns:
                    fig_loss.add_trace(
                        go.Scatter(
                            x=comp_x_data, y=comp_df['train_loss'].tolist(),
                            mode='lines+markers',
                            name='Train Loss (Comparison)',
                            line=dict(color='green', width=1, dash='dot'),
                            marker=dict(size=2),
                            opacity=0.5,
                            hovertemplate=f'<b>Train Loss (Comparison)</b><br>' +
                                        f'{x_col.capitalize()}: %{{x}}<br>' +
                                        'Loss: %{y:.4f}<br>' +
                                        '<extra></extra>'
                        )
                    )
                
                if 'valdn_loss' in comp_df.columns:
                    fig_loss.add_trace(
                        go.Scatter(
                            x=comp_x_data, y=comp_df['valdn_loss'].tolist(),
                            mode='lines+markers',
                            name='Validation Loss (Comparison)',
                            line=dict(color='orange', width=1, dash='dot'),
                            marker=dict(size=2),
                            opacity=0.5,
                            hovertemplate=f'<b>Validation Loss (Comparison)</b><br>' +
                                        f'{x_col.capitalize()}: %{{x}}<br>' +
                                        'Loss: %{y:.4f}<br>' +
                                        '<extra></extra>'
                        )
                    )
            
            fig_loss.update_layout(
                title='Training & Validation Loss Comparison',
                xaxis_title=x_col.capitalize(),
                yaxis_title='Loss',
                height=400,
                showlegend=True,
                hovermode='x unified',
                template='plotly_white'
            )
            
            # Add grid to axes
            fig_loss.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
            fig_loss.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
            
            main_figures['loss'] = fig_loss
        
        # Figure 3: Learning Rate comparison
        lr_exists = any('lr' in data['df'].columns for data in all_dataframes[:2])
        
        if lr_exists:
            fig_lr = go.Figure()
            
            # Add latest data
            latest_df = all_dataframes[0]['df']
            x_col = 'batches' if 'batches' in latest_df.columns else 'epoch'
            x_data = latest_df[x_col].tolist()
            
            if 'lr' in latest_df.columns:
                fig_lr.add_trace(
                    go.Scatter(
                        x=x_data, y=latest_df['lr'].tolist(),
                        mode='lines+markers',
                        name='Learning Rate (Latest)',
                        line=dict(color='purple', width=2, dash='solid'),
                        marker=dict(size=4),
                        opacity=1.0,
                        hovertemplate=f'<b>Learning Rate (Latest)</b><br>' +
                                    f'{x_col.capitalize()}: %{{x}}<br>' +
                                    'LR: %{y:.6f}<br>' +
                                    '<extra></extra>'
                    )
                )
            
            # Add comparison data
            if len(all_dataframes) > 1:
                comp_df = all_dataframes[1]['df']
                comp_x_data = comp_df[x_col].tolist()
                
                if 'lr' in comp_df.columns:
                    fig_lr.add_trace(
                        go.Scatter(
                            x=comp_x_data, y=comp_df['lr'].tolist(),
                            mode='lines+markers',
                            name='Learning Rate (Comparison)',
                            line=dict(color='purple', width=1, dash='dot'),
                            marker=dict(size=2),
                            opacity=0.5,
                            hovertemplate=f'<b>Learning Rate (Comparison)</b><br>' +
                                        f'{x_col.capitalize()}: %{{x}}<br>' +
                                        'LR: %{y:.6f}<br>' +
                                        '<extra></extra>'
                        )
                    )
            
            fig_lr.update_layout(
                title='Learning Rate Comparison',
                xaxis_title=x_col.capitalize(),
                yaxis_title='Learning Rate',
                yaxis_type='log',
                height=400,
                showlegend=True,
                hovermode='x unified',
                template='plotly_white'
            )
            
            # Add grid to axes
            fig_lr.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
            fig_lr.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
            
            main_figures['learning_rate'] = fig_lr
        
        # Add main comparison tab with all data for dynamic comparison
        if main_figures:
            plots_data.append({
                'filename': f"Latest vs Comparison",
                'figures': {name: json.dumps(fig.to_dict()) for name, fig in main_figures.items()},
                'datetime': "Dynamic Comparison",
                'available_plots': list(main_figures.keys()),
                'is_comparison': True,
                'all_dataframes': [
                    {
                        'filename': data['filename'],
                        'datetime': data['datetime'],
                        'index': i
                    } for i, data in enumerate(all_dataframes)
                ]
            })
    
    # Create individual tabs for all runs except the first (index 1+)
    for i, data in enumerate(all_dataframes[1:], start=1):
        df = data['df']
        filename = data['filename']
        datetime_str = data['datetime']
        
        individual_figures = {}
        
        # Get x-axis data (batches or epoch)
        x_col = 'batches' if 'batches' in df.columns else 'epoch'
        x_data = df[x_col].tolist()
        
        # Individual Accuracy plot
        if 'train_acc' in df.columns or 'valdn_acc' in df.columns:
            fig_acc = go.Figure()
            
            if 'train_acc' in df.columns:
                fig_acc.add_trace(
                    go.Scatter(
                        x=x_data, y=df['train_acc'].tolist(),
                        mode='lines+markers',
                        name='Train Accuracy',
                        line=dict(color='blue', width=2),
                        marker=dict(size=4),
                        hovertemplate='<b>Train Accuracy</b><br>' +
                                    f'{x_col.capitalize()}: %{{x}}<br>' +
                                    'Accuracy: %{y:.2f}%<br>' +
                                    '<extra></extra>'
                    )
                )
            
            if 'valdn_acc' in df.columns:
                fig_acc.add_trace(
                    go.Scatter(
                        x=x_data, y=df['valdn_acc'].tolist(),
                        mode='lines+markers',
                        name='Validation Accuracy',
                        line=dict(color='red', width=2),
                        marker=dict(size=4),
                        hovertemplate='<b>Validation Accuracy</b><br>' +
                                    f'{x_col.capitalize()}: %{{x}}<br>' +
                                    'Accuracy: %{y:.2f}%<br>' +
                                    '<extra></extra>'
                    )
                )
            
            fig_acc.update_layout(
                title='Training & Validation Accuracy',
                xaxis_title=x_col.capitalize(),
                yaxis_title='Accuracy (%)',
                height=400,
                showlegend=True,
                hovermode='x unified',
                template='plotly_white'
            )
            
            fig_acc.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
            fig_acc.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
            
            individual_figures['accuracy'] = fig_acc
        
        # Individual Loss plot
        if 'train_loss' in df.columns or 'valdn_loss' in df.columns:
            fig_loss = go.Figure()
            
            if 'train_loss' in df.columns:
                fig_loss.add_trace(
                    go.Scatter(
                        x=x_data, y=df['train_loss'].tolist(),
                        mode='lines+markers',
                        name='Train Loss',
                        line=dict(color='green', width=2),
                        marker=dict(size=4),
                        hovertemplate='<b>Train Loss</b><br>' +
                                    f'{x_col.capitalize()}: %{{x}}<br>' +
                                    'Loss: %{y:.4f}<br>' +
                                    '<extra></extra>'
                    )
                )
            
            if 'valdn_loss' in df.columns:
                fig_loss.add_trace(
                    go.Scatter(
                        x=x_data, y=df['valdn_loss'].tolist(),
                        mode='lines+markers',
                        name='Validation Loss',
                        line=dict(color='orange', width=2),
                        marker=dict(size=4),
                        hovertemplate='<b>Validation Loss</b><br>' +
                                    f'{x_col.capitalize()}: %{{x}}<br>' +
                                    'Loss: %{y:.4f}<br>' +
                                    '<extra></extra>'
                    )
                )
            
            fig_loss.update_layout(
                title='Training & Validation Loss',
                xaxis_title=x_col.capitalize(),
                yaxis_title='Loss',
                height=400,
                showlegend=True,
                hovermode='x unified',
                template='plotly_white'
            )
            
            fig_loss.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
            fig_loss.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
            
            individual_figures['loss'] = fig_loss
        
        # Individual Learning Rate plot
        if 'lr' in df.columns:
            fig_lr = go.Figure()
            
            fig_lr.add_trace(
                go.Scatter(
                    x=x_data, y=df['lr'].tolist(),
                    mode='lines+markers',
                    name='Learning Rate',
                    line=dict(color='purple', width=2),
                    marker=dict(size=4),
                    hovertemplate='<b>Learning Rate</b><br>' +
                                f'{x_col.capitalize()}: %{{x}}<br>' +
                                'LR: %{y:.6f}<br>' +
                                '<extra></extra>'
                )
            )
            
            fig_lr.update_layout(
                title='Learning Rate',
                xaxis_title=x_col.capitalize(),
                yaxis_title='Learning Rate',
                yaxis_type='log',
                height=400,
                showlegend=True,
                hovermode='x unified',
                template='plotly_white'
            )
            
            fig_lr.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
            fig_lr.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
            
            individual_figures['learning_rate'] = fig_lr
        
        # Add individual tab for this older run
        if individual_figures:
            plots_data.append({
                'filename': filename,
                'figures': {name: json.dumps(fig.to_dict()) for name, fig in individual_figures.items()},
                'datetime': datetime_str,
                'available_plots': list(individual_figures.keys())
            })
    
    return plots_data

# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Interactive Accuracy Plots</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 30px;
            border-bottom: 2px solid #007bff;
            padding-bottom: 10px;
        }
        .plot-section {
            margin-bottom: 20px;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            overflow: hidden;
            background-color: white;
        }
        .plot-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 20px;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: background-color 0.3s ease;
        }
        .plot-header:hover {
            background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%);
        }
        .plot-header h3 {
            margin: 0;
            font-size: 16px;
            font-weight: 600;
        }
        .header-right {
            display: flex;
            align-items: center;
            gap: 15px;
        }
        .datetime {
            font-size: 14px;
            opacity: 0.9;
            font-weight: 500;
        }
        .toggle-icon {
            font-size: 18px;
            transition: transform 0.3s ease;
        }
        .plot-content {
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.3s ease-out;
        }
        .plot-content.expanded {
            max-height: 1200px;
            transition: max-height 0.3s ease-in;
        }
        .file-info {
            background-color: #f8f9fa;
            padding: 15px 20px;
            border-bottom: 1px solid #dee2e6;
        }
        .file-info p {
            margin: 5px 0;
            color: #6c757d;
            font-size: 14px;
        }
        .plot-container {
            padding: 20px;
        }
        .plot-controls {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            flex-wrap: wrap;
            justify-content: center;
        }
        .plot-btn {
            padding: 10px 20px;
            border: 2px solid #007bff;
            background-color: white;
            color: #007bff;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.3s ease;
            font-weight: 500;
        }
        .plot-btn:hover {
            background-color: #007bff;
            color: white;
        }
        .plot-btn.active {
            background-color: #007bff;
            color: white;
        }
        .main-plot {
            width: 100%;
            margin-bottom: 20px;
        }
        .secondary-plots {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-top: 20px;
        }
        .secondary-plot {
            height: 300px;
        }
        .no-files {
            text-align: center;
            color: #6c757d;
            font-size: 18px;
            padding: 40px;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        .stat-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }
        .stat-card h4 {
            margin: 0 0 5px 0;
            font-size: 14px;
            opacity: 0.9;
        }
        .stat-card p {
            margin: 0;
            font-size: 24px;
            font-weight: bold;
        }
        .controls {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        .btn {
            padding: 8px 16px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.3s ease;
        }
        .btn-primary {
            background-color: #007bff;
            color: white;
        }
        .btn-primary:hover {
            background-color: #0056b3;
        }
        .btn-success {
            background-color: #28a745;
            color: white;
        }
        .btn-success:hover {
            background-color: #1e7e34;
        }
        .btn-warning {
            background-color: #ffc107;
            color: #212529;
        }
        .btn-warning:hover {
            background-color: #e0a800;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Interactive Training Metrics Dashboard</h1>
        
        {% if plots_data %}
        <div class="stats">
            <div class="stat-card">
                <h4>Total Files</h4>
                <p>{{ plots_data|length }}</p>
            </div>
            <div class="stat-card">
                <h4>Files Processed</h4>
                <p>{{ plots_data|length }}</p>
            </div>
        </div>
        
        <div class="controls">
            <button class="btn btn-primary" onclick="expandAll()">📖 Expand All</button>
            <button class="btn btn-warning" onclick="collapseAll()">📕 Collapse All</button>
            <button class="btn btn-success" onclick="expandFirst()">📖 Expand First</button>
        </div>
        
        {% for plot in plots_data %}
        {% set file_index = loop.index %}
        <div class="plot-section">
            <div class="plot-header" onclick="togglePlot({{ file_index }})">
                <h3>📁 {{ plot.filename }}</h3>
                <div class="header-right">
                    <span class="datetime">{{ plot.datetime }}</span>
                    <span class="toggle-icon" id="toggle-{{ file_index }}">▼</span>
                </div>
            </div>
            <div class="plot-content" id="content-{{ file_index }}">
                <div class="file-info">
                    <p><strong>Instructions:</strong> Click on metric buttons to switch between full-width views</p>
                </div>
                <div class="plot-container">
                    <div class="plot-controls">
                        {% for plot_type in plot.available_plots %}
                        <button class="plot-btn {% if loop.first %}active{% endif %}" 
                                onclick="switchPlot({{ file_index }}, '{{ plot_type }}')"
                                id="btn-{{ file_index }}-{{ plot_type }}">
                            {% if plot_type == 'accuracy' %}📈 Accuracy
                            {% elif plot_type == 'loss' %}📉 Loss
                            {% elif plot_type == 'learning_rate' %}⚡ Learning Rate
                            {% else %}{{ plot_type | title }}{% endif %}
                        </button>
                        {% endfor %}
                    </div>
                    
                    <div class="main-plot">
                        <div id="main-plot-{{ file_index }}" style="width:100%; height:500px;"></div>
                    </div>
                    
                    {% if plot.available_plots|length > 1 %}
                    <div class="secondary-plots">
                        {% for plot_type in plot.available_plots %}
                        {% if not loop.first %}
                        <div class="secondary-plot">
                            <div id="secondary-{{ file_index }}-{{ plot_type }}" style="width:100%; height:100%;"></div>
                        </div>
                        {% endif %}
                        {% endfor %}
                    </div>
                    {% endif %}
                </div>
            </div>
        </div>
        {% endfor %}
        
        {% else %}
        <div class="no-files">
            <h2>📂 No training_metrics.csv files found</h2>
            <p> Looked in the following directories: </p>
            <p>{% for pattern in file_patterns %} {{ pattern }} {% endfor %}</p>
        </div>
        {% endif %}
    </div>

    <script>
        // Store plot data for each file
        const plotData = {};
        
        {% for plot in plots_data %}
        plotData[{{ loop.index }}] = {
            figures: {{ plot.figures | tojson | safe }},
            available_plots: {{ plot.available_plots | tojson | safe }},
            current_plot: '{{ plot.available_plots[0] }}'
        };
        {% endfor %}
        
        // Load all plots
        {% for plot in plots_data %}
        {% set file_index = loop.index %}
        // Load main plot
        Plotly.newPlot('main-plot-{{ file_index }}', 
            JSON.parse(plotData[{{ file_index }}].figures['{{ plot.available_plots[0] }}']).data, 
            JSON.parse(plotData[{{ file_index }}].figures['{{ plot.available_plots[0] }}']).layout, {
            responsive: true,
            displayModeBar: true,
            modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d'],
            displaylogo: false
        });
        
        // Load secondary plots
        {% for plot_type in plot.available_plots %}
        {% if not loop.first %}
        Plotly.newPlot('secondary-{{ file_index }}-{{ plot_type }}', 
            JSON.parse(plotData[{{ file_index }}].figures['{{ plot_type }}']).data, 
            JSON.parse(plotData[{{ file_index }}].figures['{{ plot_type }}']).layout, {
            responsive: true,
            displayModeBar: true,
            modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d'],
            displaylogo: false
        });
        {% endif %}
        {% endfor %}
        {% endfor %}
        
        // Function to switch between plots
        function switchPlot(fileIndex, plotType) {
            const data = plotData[fileIndex];
            if (!data || !data.figures[plotType]) return;
            
            // Update button states
            data.available_plots.forEach(type => {
                const btn = document.getElementById(`btn-${fileIndex}-${type}`);
                if (btn) {
                    btn.classList.remove('active');
                }
            });
            document.getElementById(`btn-${fileIndex}-${plotType}`).classList.add('active');
            
            // Update main plot
            const plotJson = JSON.parse(data.figures[plotType]);
            Plotly.react('main-plot-' + fileIndex, plotJson.data, plotJson.layout, {
                responsive: true,
                displayModeBar: true,
                modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d'],
                displaylogo: false
            });
            
            data.current_plot = plotType;
        }
        
        // Collapsible functionality
        function togglePlot(index) {
            const content = document.getElementById(`content-${index}`);
            const icon = document.getElementById(`toggle-${index}`);
            
            if (content.classList.contains('expanded')) {
                content.classList.remove('expanded');
                icon.textContent = '▶';
            } else {
                content.classList.add('expanded');
                icon.textContent = '▼';
            }
        }
        
        function expandAll() {
            {% for plot in plots_data %}
            const content{{ loop.index }} = document.getElementById('content-{{ loop.index }}');
            const icon{{ loop.index }} = document.getElementById('toggle-{{ loop.index }}');
            content{{ loop.index }}.classList.add('expanded');
            icon{{ loop.index }}.textContent = '▼';
            {% endfor %}
        }
        
        function collapseAll() {
            {% for plot in plots_data %}
            const content{{ loop.index }} = document.getElementById('content-{{ loop.index }}');
            const icon{{ loop.index }} = document.getElementById('toggle-{{ loop.index }}');
            content{{ loop.index }}.classList.remove('expanded');
            icon{{ loop.index }}.textContent = '▶';
            {% endfor %}
        }
        
        function expandFirst() {
            // Expand only the first plot
            {% if plots_data %}
            const content1 = document.getElementById('content-1');
            const icon1 = document.getElementById('toggle-1');
            content1.classList.add('expanded');
            icon1.textContent = '▼';
            
            // Collapse all others
            {% for plot in plots_data %}
            {% if not loop.first %}
            const content{{ loop.index }} = document.getElementById('content-{{ loop.index }}');
            const icon{{ loop.index }} = document.getElementById('toggle-{{ loop.index }}');
            content{{ loop.index }}.classList.remove('expanded');
            icon{{ loop.index }}.textContent = '▶';
            {% endif %}
            {% endfor %}
            {% endif %}
        }
        
        // Auto-expand first plot on page load
        window.addEventListener('load', function() {
            {% if plots_data %}
            expandFirst();
            {% endif %}
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Main route that displays all interactive plots"""
    csv_files = find_accuracy_files()
    plots_data = create_interactive_plots(csv_files)
    return render_template_string(HTML_TEMPLATE, plots_data=plots_data)

@app.route('/api/files')
def api_files():
    """API endpoint to get list of accuracy files"""
    csv_files = find_accuracy_files()
    return {'files': csv_files}

@app.route('/api/plot/<filename>')
def api_plot(filename):
    """API endpoint to get plot data for a specific file"""
    if os.path.exists(filename):
        plots_data = create_interactive_plots([filename])
        if plots_data:
            return plots_data[0]['figures'][plots_data[0]['available_plots'][0]]
    return {'error': 'File not found'}, 404

if __name__ == '__main__':
    print("🔍 Searching for accuracy*.csv files...")
    csv_files = find_accuracy_files()
    print(f"📊 Found {len(csv_files)} accuracy files:")
    for file in csv_files:
        print(f"   - {file}")
    
    print("\n🚀 Starting web server...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("🛑 Press Ctrl+C to stop the server")
    
    app.run(debug=False, host='0.0.0.0', port=5001) 