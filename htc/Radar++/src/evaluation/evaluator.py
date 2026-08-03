"""Evaluation Manager for Multi-class Classifcation"""
import pickle
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

import core
import evaluation


class EvaluationManager(core.BaseManager):
    """Manages evaluation of multi-class classification models."""

    def __init__(self, config, rank, world_size, taxonomy_manager):
        """Initialize the evaluation manager."""
        super().__init__(config, rank, world_size)

        # Initialize evaluation metrics
        self.f1_metric = evaluation.metrics.MulticlassF1()
        self.precision_metric = evaluation.metrics.MulticlassPrecision(k_values=[
                                                                       1, 3, 5])

        # Store taxonomy information
        self.taxonomy_manager = taxonomy_manager
        self.num_labels = len(taxonomy_manager.true_labels)

        # Configure matplotlib
        plt.style.use('default')

        # Configure file paths for results and plots
        results_dir = Path(f'../saved_results/{self.config.dataset}')
        self.path_results = results_dir / f'{self.config.results_id}_all.pkl'

        plot_filename = 'plot-synthetic.pdf' if self.config.expansion == 'expand' else 'plot.pdf'
        self.path_plot = results_dir / plot_filename

    def eval(self):
        """Main evaluation function the processes results and generates metrics."""
        self._load_and_process_predictions()

        # p@k and R-precision
        precision_at_k, r_precision = self.precision_metric.compute()
        for k, pk_score in precision_at_k.items():
            self.logger.info('Precision@%d: %.2f%%', k, pk_score * 100)
        self.logger.info('R-precision: %.2f%%', r_precision * 100)

        # Micro-F1 and Macro-F1 + class-wise F1-scores
        f1_micro, f1_macro, f1_scores = self.f1_metric.compute()
        self.logger.info('Micro F1: %.2f%%', f1_micro * 100)
        self.logger.info('Macro F1: %.2f%%', f1_macro * 100)

        self._plot(f1_scores)

    def _load_and_process_predictions(self):
        """Load precitions from pickle fiel and update metrics."""
        try:
            with open(self.path_results, 'rb') as f:
                last_position = f.tell()
                pbar = tqdm(desc='Loading precitions', unit='batches')

                while True:
                    try:
                        predictions, targets = pickle.load(f)

                        # Move to CPU and ensure correct dimensions
                        predictions = self._prepare_tensor(predictions)
                        targets = self._prepare_tensor(targets)

                        # Update metrics
                        self.f1_metric.update(predictions, targets)
                        self.precision_metric.update(predictions, targets)
                        pbar.update(1)

                        # Check for inifinte loop (file pointer not moving)
                        new_position = f.tell()
                        if new_position == last_position:
                            self.logger.error(
                                'File pointer stuck, stopping evaluation.')
                            break
                        last_position = new_position

                    except EOFError:
                        self.logger.info('Reached end of results file.')
                        break

                pbar.close()

        except FileNotFoundError:
            self.logger.error('Results file not found: %s', self.path_results)
            raise
        except Exception as e:
            self.logger.error('Error loading predictions: %s', str(e))
            raise

    def _prepare_tensor(self, tensor):
        """Prepare tensor for evaluation by moving to CPU and ensuring correct shape."""
        tensor = tensor.cpu()
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    def _plot(self, f1_scores):
        """Generate visualization plot of label distribution and F1 scores."""
        self.logger.info('Generating visualization plot...')

        # Prepare data for plotting
        class_data = self._prepare_class_data()

        # Create the plot
        fig = self._create_plot(class_data, f1_scores)

        # Save plot
        self._save_plot(fig)

    def _prepare_class_data(self):
        """Prepare class frequency data for plotting."""
        class_count_items = self.taxonomy_manager.true_class_count.most_common()
        ranked_class_count = [(i, item[1])
                              for i, item in enumerate(class_count_items)]

        label_ids, counts = zip(*ranked_class_count)
        return list(label_ids), list(counts)

    def _create_plot(self, class_data, f1_scores):
        """Create the main visualization plot."""
        label_ids, counts = class_data

        # Create figure and primary axis
        fig, ax1 = plt.subplots(figsize=(14, 7))

        # Plot frequency bars
        self._plot_frequency_bars(ax1, label_ids, counts)

        # Add threshold lines
        self._add_threshold_lines(ax1, label_ids, counts)

        # Plot F1 scores on secondary axis
        self._plot_f1_scores(ax1, label_ids, f1_scores)

        # Configure layout and legend
        self._configure_plot_layout(ax1, fig)

        return fig

    def _plot_frequency_bars(self, ax1, label_ids, counts):
        """Plot frequency bars on primary axis."""
        ax1.bar(label_ids, counts, alpha=0.7,
                color='blue', label='Label Frequency')
        ax1.set_xlabel('Label ID (Frequency Rank)')
        ax1.set_ylabel('Count (Occurrences)', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.set_title(
            f'Label Distribution and F1-Scores ({self.num_labels} labels)')

        # Set x-axis ticks (show every 20th label to avoid crowding)
        step = max(1, len(label_ids) // 20)
        ax1.set_xticks(label_ids[::step])
        ax1.tick_params(axis='x', rotation=45)

    def _add_threshold_lines(self, ax1, label_ids, counts):
        """Add threshold lines showing coverage percentages."""
        threshold_percentages = [0.5, 0.8]
        colors = ['red', 'purple']

        total_count = sum(counts)
        cumulative_counts = np.cumsum(counts)

        for i, percentage in enumerate(threshold_percentages):
            threshold_value = total_count * percentage
            threshold_index = np.searchsorted(
                cumulative_counts, threshold_value)

            if threshold_index < len(label_ids):
                num_labels_at_threshold = threshold_index + 1
                #label_id_at_threshold = label_ids[threshold_index]
                x_position = threshold_index + 0.5

                ax1.axvline(
                    x=x_position,
                    color=colors[i],
                    linestyle='--',
                    linewidth=2,
                    label=f'{percentage*100:.0f}% Coverage with ({num_labels_at_threshold} labels)'
                )

    def _plot_f1_scores(self, ax1, label_ids, f1_scores):
        """Plot F1 scores on secondary y-axis."""
        ax2 = ax1.twinx()
        ax2.plot(label_ids, f1_scores, color='green', marker='o', linestyle='-',
                 linewidth=2, markersize=4, label='F1 Score')
        ax2.set_ylabel('F1 Score', color='green')
        ax2.tick_params(axis='y', labelcolor='green')
        ax2.set_ylim(0, 1)

    def _configure_plot_layout(self, ax1, fig):
        """Configure plot layout and legend."""
        # Get legend handles and labels from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        ax2 = ax1.get_shared_x_axes().get_siblings(
            ax1)[0] if ax1.get_shared_x_axes().get_siblings(ax1) else ax1.twinx()
        lines2, labels2 = ax2.get_legend_handles_labels()

        # Create combined legend
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        # Apply tight layout
        fig.tight_layout()

    def _save_plot(self, fig):
        """Save the plot to file."""
        try:
            # Ensure directory exists
            self.path_plot.parent.mkdir(parents=True, exist_ok=True)

            # Save plot
            fig.savefig(self.path_plot, bbox_inches='tight', dpi=300)
            self.logger.info('Plot saved to: %s', self.path_plot)

        except Exception as e:
            self.logger.error('Error saving plot: %s', str(e))
            raise
        finally:
            plt.close(fig)  # Clean up memory
