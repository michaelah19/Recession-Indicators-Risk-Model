"""
Static Plot Generation for Recession Prediction Analysis

Creates publication-ready matplotlib/seaborn plots for:
- Model comparison (ROC curves, PR curves, confusion matrices)
- Feature importance
- Model performance metrics
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (10, 6)


def plot_model_comparison_bar(
    comparison_df: pd.DataFrame,
    save_path: Path,
    metrics: List[str] = ['test_roc_auc', 'test_recall', 'test_specificity']
):
    """
    Create bar chart comparing models across key metrics.

    Args:
        comparison_df: DataFrame with model comparison results
        save_path: Path to save figure
        metrics: List of metrics to plot
    """
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))

    for idx, metric in enumerate(metrics):
        ax = axes[idx] if len(metrics) > 1 else axes

        # Get data
        models = comparison_df['model']
        values = comparison_df[metric]

        # Create bar plot
        bars = ax.bar(range(len(models)), values, color=['#2ecc71', '#3498db', '#e74c3c'])

        # Customize
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=15, ha='right')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()}')
        ax.set_ylim(0, 1.0)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3, label='Baseline')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    logger.info(f"✓ Model comparison bar chart saved to {save_path}")


def plot_confusion_matrices(
    models_predictions: Dict[str, Dict],
    save_path: Path
):
    """
    Create grid of confusion matrices for all models.

    Args:
        models_predictions: Dict of {model_name: {'y_true': ..., 'y_pred': ...}}
        save_path: Path to save figure
    """
    from sklearn.metrics import confusion_matrix

    n_models = len(models_predictions)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))

    for idx, (model_name, preds) in enumerate(models_predictions.items()):
        ax = axes[idx] if n_models > 1 else axes

        # Compute confusion matrix
        cm = confusion_matrix(preds['y_true'], preds['y_pred'], labels=[0, 1])

        # Plot heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                   xticklabels=['No Recession', 'Recession'],
                   yticklabels=['No Recession', 'Recession'])

        ax.set_title(f'{model_name}\nConfusion Matrix')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    logger.info(f"✓ Confusion matrices saved to {save_path}")


def plot_feature_importance(
    importance_df: pd.DataFrame,
    save_path: Path,
    top_n: int = 20
):
    """
    Create horizontal bar chart of feature importances.

    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns
        save_path: Path to save figure
        top_n: Number of top features to show
    """
    # Get top N features
    top_features = importance_df.head(top_n).copy()
    top_features = top_features.sort_values('importance', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create horizontal bar chart
    bars = ax.barh(range(len(top_features)), top_features['importance'], color='#3498db')

    # Customize
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'], fontsize=8)
    ax.set_xlabel('Importance Score')
    ax.set_title(f'Top {top_n} Most Important Features for Recession Prediction')
    ax.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
               f'{width:.4f}',
               ha='left', va='center', fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    logger.info(f"✓ Feature importance plot saved to {save_path}")


def plot_metrics_heatmap(
    comparison_df: pd.DataFrame,
    save_path: Path
):
    """
    Create heatmap of all metrics across models.

    Args:
        comparison_df: DataFrame with model comparison results
        save_path: Path to save figure
    """
    # Select metric columns
    metric_cols = [col for col in comparison_df.columns if col.startswith('test_')]

    # Prepare data
    heatmap_data = comparison_df[['model'] + metric_cols].set_index('model')
    heatmap_data.columns = [col.replace('test_', '').replace('_', ' ').title() for col in heatmap_data.columns]

    fig, ax = plt.subplots(figsize=(12, 4))

    # Create heatmap
    sns.heatmap(heatmap_data.T, annot=True, fmt='.3f', cmap='RdYlGn',
               ax=ax, cbar_kws={'label': 'Score'}, vmin=0, vmax=1)

    ax.set_title('Model Performance Metrics Heatmap (Test Set)')
    ax.set_xlabel('Model')
    ax.set_ylabel('Metric')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    logger.info(f"✓ Metrics heatmap saved to {save_path}")


def plot_training_vs_test_comparison(
    comparison_df: pd.DataFrame,
    save_path: Path
):
    """
    Compare training vs test performance to check for overfitting.

    Args:
        comparison_df: DataFrame with model comparison results
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    models = comparison_df['model']
    x = np.arange(len(models))
    width = 0.35

    # Plot train and test ROC-AUC
    train_bars = ax.bar(x - width/2, comparison_df['train_roc_auc'], width,
                       label='Train ROC-AUC', color='#3498db', alpha=0.8)
    test_bars = ax.bar(x + width/2, comparison_df['test_roc_auc'], width,
                      label='Test ROC-AUC', color='#e74c3c', alpha=0.8)

    # Customize
    ax.set_xlabel('Model')
    ax.set_ylabel('ROC-AUC Score')
    ax.set_title('Training vs Test Performance (ROC-AUC)')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3, label='Random Baseline')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bars in [train_bars, test_bars]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    logger.info(f"✓ Train vs test comparison saved to {save_path}")


def create_model_selection_summary(
    comparison_df: pd.DataFrame,
    save_path: Path
):
    """
    Create a summary figure showing why a model was selected.

    Args:
        comparison_df: DataFrame with model comparison results (sorted by test_roc_auc)
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    models = comparison_df['model']
    colors = ['#2ecc71', '#3498db', '#e74c3c']

    # 1. ROC-AUC Comparison
    ax = axes[0, 0]
    bars = ax.bar(range(len(models)), comparison_df['test_roc_auc'], color=colors)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.set_ylabel('ROC-AUC')
    ax.set_title('Test ROC-AUC (Primary Metric)')
    ax.set_ylim(0, 1.0)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.3f}',
               ha='center', va='bottom')

    # 2. Recall vs Specificity Trade-off
    ax = axes[0, 1]
    for i, (model, color) in enumerate(zip(models, colors)):
        recall = comparison_df.iloc[i]['test_recall']
        spec = comparison_df.iloc[i]['test_specificity']
        ax.scatter(spec, recall, s=200, c=color, label=model, alpha=0.7, edgecolors='black')
        ax.annotate(model, (spec, recall), xytext=(5, 5), textcoords='offset points', fontsize=8)

    ax.set_xlabel('Specificity (True Negative Rate)')
    ax.set_ylabel('Recall (True Positive Rate)')
    ax.set_title('Recall vs Specificity Trade-off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Equal Trade-off')
    ax.legend()
    ax.grid(alpha=0.3)

    # 3. Multiple Metrics Radar
    ax = axes[1, 0]
    metrics_to_plot = ['test_roc_auc', 'test_recall', 'test_specificity', 'test_precision', 'test_f1']
    metric_labels = ['ROC-AUC', 'Recall', 'Specificity', 'Precision', 'F1']

    # Just show a bar chart of all metrics for best model
    best_model_metrics = comparison_df.iloc[0][metrics_to_plot].values
    bars = ax.bar(range(len(metric_labels)), best_model_metrics, color='#2ecc71', alpha=0.7)
    ax.set_xticks(range(len(metric_labels)))
    ax.set_xticklabels(metric_labels, rotation=45, ha='right')
    ax.set_ylabel('Score')
    ax.set_title(f'Best Model ({comparison_df.iloc[0]["model"]}) - All Metrics')
    ax.set_ylim(0, 1.0)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}',
               ha='center', va='bottom', fontsize=8)

    # 4. Model Selection Rationale Text
    ax = axes[1, 1]
    ax.axis('off')

    best_model = comparison_df.iloc[0]
    second_model = comparison_df.iloc[1] if len(comparison_df) > 1 else None

    rationale_text = f"""
MODEL SELECTION SUMMARY

✓ SELECTED: {best_model['model']}

Key Performance Metrics:
• ROC-AUC: {best_model['test_roc_auc']:.3f} (discrimination ability)
• Recall: {best_model['test_recall']:.1%} (catches {int(best_model['test_recall']*3)}/3 recessions)
• Specificity: {best_model['test_specificity']:.1%} (low false alarm rate)
• F1 Score: {best_model['test_f1']:.3f} (balanced performance)

Selection Rationale:
• Highest test set ROC-AUC
• Good balance of recall and specificity
• Minimal overfitting (train ROC-AUC: {best_model['train_roc_auc']:.3f})

Trade-offs Considered:
"""

    if second_model is not None:
        rationale_text += f"""• {second_model['model']} has {'higher' if second_model['test_roc_auc'] > best_model['test_roc_auc'] else 'lower'} ROC-AUC ({second_model['test_roc_auc']:.3f})
  but {'better' if second_model['test_recall'] > best_model['test_recall'] else 'worse'} recall ({second_model['test_recall']:.1%})
"""

    rationale_text += f"""
Conclusion: {best_model['model']} provides the best
overall discrimination (ROC-AUC) while maintaining
acceptable recall for recession detection.
    """

    ax.text(0.05, 0.95, rationale_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle('Model Selection Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    logger.info(f"✓ Model selection summary saved to {save_path}")


def generate_all_plots(
    comparison_df: pd.DataFrame,
    importance_df: Optional[pd.DataFrame] = None,
    output_dir: Path = Path("reports/figures")
):
    """
    Generate all static plots for model analysis.

    Args:
        comparison_df: Model comparison results
        importance_df: Feature importance data (optional)
        output_dir: Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating plots in {output_dir}...")

    # 1. Model comparison bar chart
    plot_model_comparison_bar(
        comparison_df,
        output_dir / "model_comparison_metrics.png"
    )

    # 2. Metrics heatmap
    plot_metrics_heatmap(
        comparison_df,
        output_dir / "model_metrics_heatmap.png"
    )

    # 3. Train vs test comparison
    plot_training_vs_test_comparison(
        comparison_df,
        output_dir / "train_vs_test_performance.png"
    )

    # 4. Model selection summary
    create_model_selection_summary(
        comparison_df,
        output_dir / "model_selection_summary.png"
    )

    # 5. Feature importance (if provided)
    if importance_df is not None:
        plot_feature_importance(
            importance_df,
            output_dir / "feature_importance_top20.png",
            top_n=20
        )

    logger.info(f"✓ All plots generated successfully in {output_dir}")


def plot_full_timeline_with_predictions(
    train_dates: pd.Series,
    train_y_true: np.ndarray,
    test_dates: pd.Series,
    test_y_true: np.ndarray,
    test_y_pred_proba: np.ndarray,
    test_y_pred: np.ndarray,
    save_path: Path,
    threshold: float = 0.5
):
    """
    Plot full timeline showing training data context and test set predictions.

    Args:
        train_dates: Series of dates for training samples
        train_y_true: Actual recession labels for training (0/1)
        test_dates: Series of dates for test samples
        test_y_true: Actual recession labels for test (0/1)
        test_y_pred_proba: Predicted recession probabilities (test only)
        test_y_pred: Predicted recession labels (test only, 0/1)
        save_path: Path to save figure
        threshold: Classification threshold
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 6))

    # Combine all dates for x-axis (convert to Series if needed)
    if isinstance(train_dates, pd.DatetimeIndex):
        train_dates = pd.Series(train_dates.values, index=range(len(train_dates)))
    if isinstance(test_dates, pd.DatetimeIndex):
        test_dates = pd.Series(test_dates.values, index=range(len(train_dates), len(train_dates) + len(test_dates)))

    all_dates = pd.concat([train_dates, test_dates])
    all_y_true = np.concatenate([train_y_true, test_y_true])

    # Plot actual recession periods as background shading (full timeline)
    for i, (date, is_recession) in enumerate(zip(all_dates, all_y_true)):
        if is_recession == 1:
            ax.axvspan(date, date, alpha=0.15, color='red', linewidth=0)

    # Add vertical line to mark train/test split
    split_date = test_dates.iloc[0]
    ax.axvline(x=split_date, color='black', linestyle='--', linewidth=2,
               label='Train/Test Split', alpha=0.7)

    # Plot actual values as line (full timeline)
    ax.plot(all_dates, all_y_true, 'o-', color='red', linewidth=1.5,
            markersize=3, alpha=0.6, label='Actual Recession Status')

    # Plot predictions as line (test set only)
    ax.plot(test_dates, test_y_pred, 's-', color='blue', linewidth=2,
            markersize=5, alpha=0.8, label='Predicted Recession Status (Test)', zorder=5)

    # Plot prediction probabilities on secondary axis
    ax2 = ax.twinx()
    ax2.plot(test_dates, test_y_pred_proba, 'o--', color='green', linewidth=1.5,
             markersize=4, alpha=0.7, label='Predicted Probability (Test)')
    ax2.axhline(y=threshold, color='green', linestyle=':', linewidth=1.5,
                alpha=0.5, label=f'Threshold ({threshold})')
    ax2.set_ylabel('Recession Probability', fontsize=11, color='green')
    ax2.set_ylim(-0.05, 1.05)
    ax2.tick_params(axis='y', labelcolor='green')

    # Formatting
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Recession Status (0=No, 1=Yes)', fontsize=11)
    ax.set_title('Recession Predictions: Full Timeline - Quarterly Data (Training Context + Test Predictions)',
                 fontsize=14, fontweight='bold')
    ax.set_ylim(-0.15, 1.3)
    ax.grid(axis='y', alpha=0.3)

    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)

    # Add text box with metrics
    accuracy = (test_y_true == test_y_pred).mean()
    n_correct = (test_y_true == test_y_pred).sum()
    n_total = len(test_y_true)
    metrics_text = f'Test Set Performance:\nAccuracy: {accuracy:.1%} ({n_correct}/{n_total} quarters)\nRecession quarters: {test_y_true.sum()}/{n_total}'
    ax.text(0.98, 0.05, metrics_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

    logger.info(f"✓ Full timeline with predictions saved to {save_path}")


def plot_indicator_predictions_vs_actual(
    indicator_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    recession_true: np.ndarray,
    dates: Optional[pd.Series] = None,
    save_path: Optional[Path] = None,
    unit: str = '%'
):
    """
    Plot predicted vs actual indicator changes with recession periods shaded.

    Args:
        indicator_name: Name of the indicator
        y_true: Actual indicator changes (all test samples)
        y_pred: Predicted indicator changes (all test samples)
        recession_true: Actual recession status (0/1) for shading
        dates: Optional dates for timeline plot
        save_path: Path to save figure
        unit: Unit of measurement (%, pp, etc.)
    """
    # Filter scatter plot to only recession periods (where predictions matter most)
    recession_mask = recession_true == 1
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred)) & recession_mask

    y_true_recession = y_true[valid_mask]
    y_pred_recession = y_pred[valid_mask]

    if len(y_true_recession) == 0:
        logger.warning(f"No valid recession data for {indicator_name}, skipping plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Scatter plot (ONLY recession periods)
    ax1 = axes[0]
    ax1.scatter(y_true_recession, y_pred_recession, s=100, alpha=0.6, edgecolors='black', linewidth=1.5)

    # 1:1 line (perfect agreement)
    min_val = min(y_true_recession.min(), y_pred_recession.min())
    max_val = max(y_true_recession.max(), y_pred_recession.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='y=x (Prediction == Actual)')

    # Calculate metrics (only for recession periods)
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    mae = mean_absolute_error(y_true_recession, y_pred_recession)
    rmse = np.sqrt(mean_squared_error(y_true_recession, y_pred_recession))
    r2 = r2_score(y_true_recession, y_pred_recession)

    ax1.set_xlabel(f'Actual Change ({unit})', fontsize=11)
    ax1.set_ylabel(f'Predicted Change ({unit})', fontsize=11)
    ax1.set_title(f'{indicator_name}: Predicted vs Actual (Recession Quarters Only)', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Add metrics text box
    metrics_text = f'MAE: {mae:.2f}{unit}\nRMSE: {rmse:.2f}{unit}\nR²: {r2:.3f}\nRecession quarters: {len(y_true_recession)}'
    ax1.text(0.05, 0.95, metrics_text, transform=ax1.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 2: Timeline comparison with recession shading
    ax2 = axes[1]
    if dates is not None and len(dates) == len(y_true):
        # Convert DatetimeIndex to array if needed
        if isinstance(dates, pd.DatetimeIndex):
            dates_array = dates.values
        elif isinstance(dates, pd.Series):
            dates_array = dates.values
        else:
            dates_array = np.array(dates)

        # Shade recession periods
        for i, is_recession in enumerate(recession_true):
            if is_recession == 1:
                ax2.axvspan(i-0.5, i+0.5, alpha=0.2, color='red', linewidth=0)

        # Plot lines
        x_pos = np.arange(len(dates))
        ax2.plot(x_pos, y_true, 'o-', color='red', linewidth=2, markersize=5,
                 label='Actual', alpha=0.8, zorder=3)
        ax2.plot(x_pos, y_pred, 's--', color='blue', linewidth=2, markersize=5,
                 label='Predicted', alpha=0.8, zorder=3)

        # Show ~10 year labels
        step = max(1, len(dates)//10)
        ax2.set_xticks(x_pos[::step])
        ax2.set_xticklabels([pd.Timestamp(dates_array[i]).strftime('%Y')
                              for i in range(0, len(dates), step)],
                            rotation=45, ha='right', fontsize=9)
        ax2.set_xlabel('Year', fontsize=11)
    else:
        # Simple timeline by sample number
        x_pos = np.arange(len(y_true))

        # Shade recession periods
        for i, is_recession in enumerate(recession_true):
            if is_recession == 1:
                ax2.axvspan(i-0.5, i+0.5, alpha=0.2, color='red', linewidth=0)

        ax2.plot(x_pos, y_true, 'o-', color='red', linewidth=2, markersize=5,
                 label='Actual', alpha=0.8)
        ax2.plot(x_pos, y_pred, 's--', color='blue', linewidth=2, markersize=5,
                 label='Predicted', alpha=0.8)
        ax2.set_xlabel('Test Sample', fontsize=11)

    ax2.set_ylabel(f'Change ({unit})', fontsize=11)
    ax2.set_title('Timeline: Test Set - Quarterly Data (Red shading = Recession)', fontsize=12, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(axis='y', alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        logger.info(f"✓ Indicator prediction plot saved to {save_path}")
    else:
        plt.show()


def plot_all_indicators_comparison(
    indicators_dict: Dict[str, Dict[str, np.ndarray]],
    save_path: Path
):
    """
    Create grid of small multiples comparing all indicators.

    Args:
        indicators_dict: Dict of {indicator_name: {'y_true': ..., 'y_pred': ...}}
        save_path: Path to save figure
    """
    n_indicators = len(indicators_dict)
    n_cols = 2
    n_rows = (n_indicators + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
    axes = axes.flatten() if n_indicators > 1 else [axes]

    for idx, (indicator_name, data) in enumerate(indicators_dict.items()):
        ax = axes[idx]
        y_true = data['y_true']
        y_pred = data['y_pred']

        # Scatter plot
        ax.scatter(y_true, y_pred, s=80, alpha=0.6, edgecolors='black')

        # 1:1 line (perfect agreement)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5, alpha=0.7, label='y=x')

        # Calculate R²
        from sklearn.metrics import r2_score, mean_absolute_error
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)

        ax.set_xlabel('Actual', fontsize=9)
        ax.set_ylabel('Predicted', fontsize=9)
        ax.set_title(f'{indicator_name}\nR²={r2:.3f}, MAE={mae:.2f}', fontsize=10)
        ax.grid(alpha=0.3)

    # Hide extra subplots
    for idx in range(n_indicators, len(axes)):
        axes[idx].axis('off')

    plt.suptitle('All Indicators: Predicted vs Actual (Recession Quarters Only)', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

    logger.info(f"✓ All indicators comparison saved to {save_path}")
