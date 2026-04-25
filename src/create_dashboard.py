import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from datetime import datetime

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

def load_training_history():
    """Load training history if available"""
    if os.path.exists("training_history.json"):
        with open("training_history.json", "r") as f:
            return json.load(f)
    return None

def load_evaluation_metrics():
    """Load evaluation metrics if available"""
    if os.path.exists("evaluation_metrics.json"):
        with open("evaluation_metrics.json", "r") as f:
            return json.load(f)
    return None

def create_performance_dashboard():
    """Create comprehensive performance dashboard"""
    
    print("Creating performance dashboard...")
    
    # Load data
    training_history = load_training_history()
    eval_metrics = load_evaluation_metrics()
    
    # Check if visualization files exist
    scatter_exists = os.path.exists("scatter_actual_pred.png")
    residuals_exists = os.path.exists("residuals.png")
    time_series_files = [f"time_series_{city}.png" for city in ["Chennai", "Coimbatore", "Kanyakumari"]]
    time_series_exist = [os.path.exists(f) for f in time_series_files]
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    # Title
    fig.suptitle('Rainfall Prediction Model - Performance Dashboard', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # ===== ROW 1: Training History =====
    if training_history:
        ax1 = fig.add_subplot(gs[0, :2])
        epochs = training_history["epochs"]
        losses = training_history["loss"]
        best_losses = training_history["best_loss"]
        
        ax1.plot(epochs, losses, 'b-', alpha=0.6, label='Training Loss', linewidth=2)
        ax1.plot(epochs, best_losses, 'r--', label='Best Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=11)
        ax1.set_ylabel('Loss (MSE)', fontsize=11)
        ax1.set_title('Training History', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add final best loss annotation
        final_best = training_history.get("final_best_loss", best_losses[-1])
        ax1.axhline(y=final_best, color='g', linestyle=':', alpha=0.7, label=f'Final Best: {final_best:.4f}')
        ax1.legend()
    else:
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.text(0.5, 0.5, 'Training history not available.\nRun train.py first.', 
                ha='center', va='center', fontsize=12, style='italic')
        ax1.set_title('Training History', fontsize=13, fontweight='bold')
        ax1.axis('off')
    
    # ===== ROW 1: Metrics Summary =====
    ax2 = fig.add_subplot(gs[0, 2:])
    ax2.axis('off')
    
    if eval_metrics:
        metrics = eval_metrics["metrics"]
        targets = eval_metrics["performance_targets"]
        targets_met = eval_metrics["targets_met"]
        
        metrics_text = "PERFORMANCE METRICS SUMMARY\n" + "="*40 + "\n\n"
        metrics_text += f"RMSE:        {metrics['RMSE']:.4f} mm "
        metrics_text += f"{'✓' if targets_met['RMSE'] else '✗'} (Target: < {targets['RMSE_target']:.1f} mm)\n\n"
        metrics_text += f"MAE:         {metrics['MAE']:.4f} mm "
        metrics_text += f"{'✓' if targets_met['MAE'] else '✗'} (Target: < {targets['MAE_target']:.1f} mm)\n\n"
        metrics_text += f"R² Score:    {metrics['R2_score']:.4f} "
        metrics_text += f"{'✓' if targets_met['R2'] else '✗'} (Target: > {targets['R2_target']:.1f})\n\n"
        metrics_text += f"Correlation: {metrics['Correlation']:.4f} "
        metrics_text += f"{'✓' if targets_met['Correlation'] else '✗'} (Target: > {targets['Correlation_target']:.1f})\n\n"
        metrics_text += f"\nTest Samples: {eval_metrics['test_samples']}\n"
        metrics_text += f"Evaluation Date: {eval_metrics['evaluation_date']}"
        
        ax2.text(0.1, 0.9, metrics_text, fontsize=11, verticalalignment='top',
                family='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    else:
        ax2.text(0.5, 0.5, 'Evaluation metrics not available.\nRun evaluate.py first.', 
                ha='center', va='center', fontsize=12, style='italic')
    
    # ===== ROW 2: Metrics Comparison Bar Chart =====
    ax3 = fig.add_subplot(gs[1, :2])
    if eval_metrics:
        metrics = eval_metrics["metrics"]
        targets = eval_metrics["performance_targets"]
        targets_met = eval_metrics["targets_met"]
        
        # Normalize metrics for comparison (inverse for RMSE and MAE)
        metric_names = ['RMSE', 'MAE', 'R²', 'Correlation']
        metric_values = [
            metrics['RMSE'] / targets['RMSE_target'] * 100,  # Percentage of target
            metrics['MAE'] / targets['MAE_target'] * 100,
            metrics['R2_score'] / targets['R2_target'] * 100,
            metrics['Correlation'] / targets['Correlation_target'] * 100
        ]
        target_line = [100] * 4  # 100% = target met
        
        x_pos = np.arange(len(metric_names))
        colors = ['green' if met else 'red' for met in [
            targets_met['RMSE'], targets_met['MAE'], 
            targets_met['R2'], targets_met['Correlation']
        ]]
        
        bars = ax3.bar(x_pos, metric_values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax3.axhline(y=100, color='blue', linestyle='--', linewidth=2, label='Target (100%)')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(metric_names)
        ax3.set_ylabel('Performance (% of Target)', fontsize=11)
        ax3.set_title('Metrics vs Targets', fontsize=13, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, metric_values)):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
    else:
        ax3.text(0.5, 0.5, 'No metrics available', ha='center', va='center', fontsize=12)
        ax3.set_title('Metrics vs Targets', fontsize=13, fontweight='bold')
    
    # ===== ROW 2: Statistics Summary =====
    ax4 = fig.add_subplot(gs[1, 2:])
    ax4.axis('off')
    
    if eval_metrics:
        stats = eval_metrics["additional_stats"]
        stats_text = "DATA STATISTICS\n" + "="*40 + "\n\n"
        stats_text += "Actual Rainfall:\n"
        stats_text += f"  Mean: {stats['actual_mean']:.2f} mm\n"
        stats_text += f"  Std:  {stats['actual_std']:.2f} mm\n"
        stats_text += f"  Range: [{stats['actual_min']:.2f}, {stats['actual_max']:.2f}] mm\n\n"
        stats_text += "Predicted Rainfall:\n"
        stats_text += f"  Mean: {stats['predicted_mean']:.2f} mm\n"
        stats_text += f"  Std:  {stats['predicted_std']:.2f} mm\n"
        stats_text += f"  Range: [{stats['predicted_min']:.2f}, {stats['predicted_max']:.2f}] mm\n"
        
        ax4.text(0.1, 0.9, stats_text, fontsize=10, verticalalignment='top',
                family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    else:
        ax4.text(0.5, 0.5, 'No statistics available', ha='center', va='center', fontsize=12)
    
    # ===== ROW 3: Scatter Plot (if exists) =====
    ax5 = fig.add_subplot(gs[2, :2])
    if scatter_exists:
        # Try to load and display the scatter plot
        try:
            img = plt.imread("scatter_actual_pred.png")
            ax5.imshow(img)
            ax5.axis('off')
            ax5.set_title('Predicted vs Actual', fontsize=13, fontweight='bold', pad=10)
        except:
            ax5.text(0.5, 0.5, 'Could not load scatter plot', ha='center', va='center', fontsize=12)
    else:
        ax5.text(0.5, 0.5, 'Scatter plot not available.\nRun evaluate.py first.', 
                ha='center', va='center', fontsize=12, style='italic')
        ax5.set_title('Predicted vs Actual', fontsize=13, fontweight='bold')
    
    # ===== ROW 3: Residual Plot (if exists) =====
    ax6 = fig.add_subplot(gs[2, 2:])
    if residuals_exists:
        try:
            img = plt.imread("residuals.png")
            ax6.imshow(img)
            ax6.axis('off')
            ax6.set_title('Residual Error Distribution', fontsize=13, fontweight='bold', pad=10)
        except:
            ax6.text(0.5, 0.5, 'Could not load residual plot', ha='center', va='center', fontsize=12)
    else:
        ax6.text(0.5, 0.5, 'Residual plot not available.\nRun evaluate.py first.', 
                ha='center', va='center', fontsize=12, style='italic')
        ax6.set_title('Residual Error Distribution', fontsize=13, fontweight='bold')
    
    # ===== ROW 4: Time Series Plots =====
    cities = ["Chennai", "Coimbatore", "Kanyakumari"]
    for idx, city in enumerate(cities):
        ax = fig.add_subplot(gs[3, idx])
        file_path = f"time_series_{city}.png"
        if os.path.exists(file_path):
            try:
                img = plt.imread(file_path)
                ax.imshow(img)
                ax.axis('off')
                ax.set_title(f'{city} Forecast', fontsize=12, fontweight='bold', pad=5)
            except:
                ax.text(0.5, 0.5, f'{city}\nnot available', ha='center', va='center', fontsize=10)
                ax.axis('off')
        else:
            ax.text(0.5, 0.5, f'{city}\nnot available', ha='center', va='center', fontsize=10, style='italic')
            ax.axis('off')
    
    # Save dashboard
    plt.savefig('performance_dashboard.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Performance dashboard saved: performance_dashboard.png")
    
    # Also create a summary text file
    if eval_metrics or training_history:
        summary_text = "="*70 + "\n"
        summary_text += "RAINFALL PREDICTION MODEL - PERFORMANCE SUMMARY\n"
        summary_text += "="*70 + "\n\n"
        summary_text += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        if training_history:
            summary_text += "TRAINING INFORMATION\n"
            summary_text += "-"*70 + "\n"
            summary_text += f"Total Epochs: {training_history['total_epochs']}\n"
            summary_text += f"Final Best Loss: {training_history['final_best_loss']:.6f}\n"
            summary_text += f"Initial Loss: {training_history['loss'][0]:.6f}\n"
            improvement = ((training_history['loss'][0] - training_history['final_best_loss']) / training_history['loss'][0]) * 100
            summary_text += f"Improvement: {improvement:.2f}%\n\n"
        
        if eval_metrics:
            summary_text += "EVALUATION METRICS\n"
            summary_text += "-"*70 + "\n"
            metrics = eval_metrics["metrics"]
            targets_met = eval_metrics["targets_met"]
            summary_text += f"RMSE:        {metrics['RMSE']:.4f} mm {'✓ PASS' if targets_met['RMSE'] else '✗ FAIL'}\n"
            summary_text += f"MAE:         {metrics['MAE']:.4f} mm {'✓ PASS' if targets_met['MAE'] else '✗ FAIL'}\n"
            summary_text += f"R² Score:    {metrics['R2_score']:.4f} {'✓ PASS' if targets_met['R2'] else '✗ FAIL'}\n"
            summary_text += f"Correlation:  {metrics['Correlation']:.4f} {'✓ PASS' if targets_met['Correlation'] else '✗ FAIL'}\n\n"
            
            # Overall status
            all_passed = all(targets_met.values())
            summary_text += "OVERALL STATUS\n"
            summary_text += "-"*70 + "\n"
            if all_passed:
                summary_text += "✓ ALL PERFORMANCE TARGETS MET\n"
            else:
                failed = [k for k, v in targets_met.items() if not v]
                summary_text += f"✗ SOME TARGETS NOT MET: {', '.join(failed)}\n"
        
        summary_text += "\n" + "="*70 + "\n"
        
        with open("performance_summary.txt", "w") as f:
            f.write(summary_text)
        
        print("✓ Performance summary saved: performance_summary.txt")
    
    plt.close()
    print("\n✔ Performance dashboard creation complete!\n")

if __name__ == "__main__":
    create_performance_dashboard()
