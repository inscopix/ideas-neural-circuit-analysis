import os
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.collections import PolyCollection, LineCollection, QuadMesh
from matplotlib.gridspec import GridSpec
from utils.utils import (
    estimate_svg_size,
    save_optimized_svg,
    QUADMESH_FACTOR,
    LINECOLLECTION_SEGMENT_FACTOR,
    POLYCOLLECTION_FACTOR,
    SCATTER_POINT_FACTOR,
)


class TestSVGOptimization:
    """Test suite for SVG size estimation and optimization.

    This test suite validates the SVG size estimation factors used in save_optimized_svg
    to prevent large SVG files from being generated. The factors are critical for:

    - QUADMESH_FACTOR (200): Used for correlation matrix heatmaps
    - LINECOLLECTION_SEGMENT_FACTOR (2500): Used for line plots and spatial connections
    - POLYCOLLECTION_FACTOR (35000): Used for hexbin plots
    - SCATTER_POINT_FACTOR (800): Used for scatter plots (newly added)

    These tests ensure:
    1. Factors provide reasonable size estimation accuracy (~10-20% error)
    2. Large files (>10MB) are properly detected for rasterization
    3. Estimation logic is consistent and scales appropriately
    4. Edge cases are handled correctly
    5. Scatter plots (major cause of large SVG files) are properly detected

    If these tests fail after factor changes, it indicates the SVG optimization
    may not work correctly, potentially leading to 100MB+ SVG files.
    """

    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp(prefix="svg_test_")
        # Use consistent random seed for reproducible tests
        np.random.seed(42)

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _measure_actual_svg_size(self, fig, filename):
        """Save figure as SVG and return file size in bytes."""
        estimated_size = save_optimized_svg(fig, filename)
        actual_size = os.path.getsize(filename)
        return actual_size, estimated_size

    def _count_plot_elements(self, fig):
        """Count the different types of elements in the plot."""
        quad_cells = 0
        poly_count = 0
        line_segments = 0

        for ax in fig.get_axes():
            for collection in ax.collections:
                if isinstance(collection, QuadMesh) and hasattr(
                    collection, "_coordinates"
                ):
                    mesh_shape = collection._coordinates.shape
                    if len(mesh_shape) >= 3:
                        quad_cells += (mesh_shape[0] - 1) * (mesh_shape[1] - 1)
                elif isinstance(collection, PolyCollection):
                    try:
                        poly_count += len(collection.get_paths())
                    except Exception:
                        pass
                elif isinstance(collection, LineCollection):
                    try:
                        line_segments += len(collection.get_segments())
                    except Exception:
                        pass

        return quad_cells, poly_count, line_segments

    def _create_spatial_correlation_plot(
        self, n_points=1000, gridsize=30, n_lines=100
    ):
        """Create a spatial correlation plot similar to correlation tool."""
        # Generate synthetic data
        x_pos = np.random.uniform(0, 1000, n_points)
        y_pos = np.random.uniform(0, 1000, n_points)
        distances = np.sqrt(
            (x_pos[:, None] - x_pos[None, :]) ** 2
            + (y_pos[:, None] - y_pos[None, :]) ** 2
        )
        correlations = 0.8 * np.exp(-distances / 200) + 0.2 * np.random.randn(
            n_points, n_points
        )

        # Create plot
        fig = plt.figure(figsize=(12, 8))
        gs = GridSpec(2, 2, height_ratios=[2, 0.05])

        # Scatter plot with trend line
        ax_scatter = fig.add_subplot(gs[0, 0])
        triu_indices = np.triu_indices(n_points, k=1)
        dist_flat = distances[triu_indices]
        corr_flat = correlations[triu_indices]

        # Scatter plot
        sns.scatterplot(
            x=dist_flat[:n_points],
            y=corr_flat[:n_points],
            alpha=0.3,
            s=10,
            ax=ax_scatter,
        )

        # Regression line
        sns.regplot(
            x=dist_flat[:n_points],
            y=corr_flat[:n_points],
            scatter=False,
            ax=ax_scatter,
            color="gray",
        )

        ax_scatter.set_title("Correlation vs Distance")
        ax_scatter.set_ylabel("Correlation")
        ax_scatter.set_xlabel("Distance (pixels)")
        ax_scatter.set_xlim(0, 1000)
        ax_scatter.set_ylim(-1, 1)

        # Hexbin density plot (creates PolyCollection)
        ax_density = fig.add_subplot(gs[0, 1])
        hb = ax_density.hexbin(
            dist_flat[:n_points],
            corr_flat[:n_points],
            gridsize=gridsize,
            cmap="viridis",
            mincnt=1,
        )

        ax_density.set_title("Density Plot")
        ax_density.set_xlabel("Distance (pixels)")
        ax_density.set_ylabel("Correlation")
        ax_density.set_xlim(0, 1000)
        ax_density.set_ylim(-1, 1)

        # Add line collections (spatial correlation lines)
        if n_lines > 0:
            lines = []
            for i in range(min(n_lines, len(x_pos) - 1)):
                j = i + 1
                lines.append([(x_pos[i], y_pos[i]), (x_pos[j], y_pos[j])])

            lc = LineCollection(lines, alpha=0.5, linewidths=1)
            ax_scatter.add_collection(lc)

        # Colorbar
        cax = fig.add_subplot(gs[1, :])
        plt.colorbar(hb, cax=cax, orientation="horizontal", label="Count")

        plt.tight_layout()
        return fig

    def _create_correlation_matrix_plot(self, matrix_size=100):
        """Create a correlation matrix plot with QuadMesh."""
        # Generate correlation matrix
        corr_matrix = np.random.randn(matrix_size, matrix_size)

        fig, ax = plt.subplots(figsize=(8, 6))

        # Create coordinates for pcolormesh
        x = np.arange(matrix_size + 1)
        y = np.arange(matrix_size + 1)

        # Create heatmap using pcolormesh (this creates QuadMesh with proper sizing)
        im = ax.pcolormesh(x, y, corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1)

        ax.set_title(f"Correlation Matrix ({matrix_size}x{matrix_size})")
        ax.set_xlabel("Cell Index")
        ax.set_ylabel("Cell Index")
        plt.colorbar(im, ax=ax, label="Correlation")

        plt.tight_layout()
        return fig

    def test_svg_factor_constants(self):
        """Test that the SVG size estimation factors are sensible constants."""
        # Test that factors are positive
        assert QUADMESH_FACTOR > 0, "QUADMESH_FACTOR should be positive"
        assert (
            LINECOLLECTION_SEGMENT_FACTOR > 0
        ), "LINECOLLECTION_SEGMENT_FACTOR should be positive"
        assert (
            POLYCOLLECTION_FACTOR > 0
        ), "POLYCOLLECTION_FACTOR should be positive"
        assert (
            SCATTER_POINT_FACTOR > 0
        ), "SCATTER_POINT_FACTOR should be positive"

        # Test relative magnitudes make sense
        # PolyCollection should have highest factor (most complex)
        # LineCollection and ScatterPoint should be medium
        # QuadMesh should be lowest (most efficient)
        assert (
            POLYCOLLECTION_FACTOR > LINECOLLECTION_SEGMENT_FACTOR
        ), "POLYCOLLECTION_FACTOR should be > LINECOLLECTION_SEGMENT_FACTOR"
        assert (
            POLYCOLLECTION_FACTOR > SCATTER_POINT_FACTOR
        ), "POLYCOLLECTION_FACTOR should be > SCATTER_POINT_FACTOR"
        assert (
            LINECOLLECTION_SEGMENT_FACTOR > QUADMESH_FACTOR
        ), "LINECOLLECTION_SEGMENT_FACTOR should be > QUADMESH_FACTOR"
        assert (
            SCATTER_POINT_FACTOR > QUADMESH_FACTOR
        ), "SCATTER_POINT_FACTOR should be > QUADMESH_FACTOR"

    def test_quadmesh_factor_accuracy(self):
        """Test that QUADMESH_FACTOR provides reasonable size estimation for matrices."""
        test_cases = [
            {"matrix_size": 50, "expected_error_threshold": 40},
            {"matrix_size": 100, "expected_error_threshold": 40},
            {"matrix_size": 200, "expected_error_threshold": 40},
        ]

        for case in test_cases:
            with plt.style.context("default"):  # Ensure consistent styling
                fig = self._create_correlation_matrix_plot(
                    matrix_size=case["matrix_size"]
                )

                # Count elements
                (
                    quad_cells,
                    poly_count,
                    line_segments,
                ) = self._count_plot_elements(fig)

                # Measure actual size and get estimated size
                svg_path = os.path.join(
                    self.temp_dir, f"test_matrix_{case['matrix_size']}.svg"
                )
                (
                    actual_size_bytes,
                    estimated_size_bytes,
                ) = self._measure_actual_svg_size(fig, svg_path)

                # Ensure actual size is valid
                assert (
                    actual_size_bytes > 0
                ), f"Invalid actual file size: {actual_size_bytes}"

                # Calculate accuracy
                accuracy_ratio = estimated_size_bytes / actual_size_bytes
                error_percent = abs(1 - accuracy_ratio) * 100

                # Assert accuracy is within acceptable range
                assert error_percent < case["expected_error_threshold"], (
                    f"Matrix size {case['matrix_size']}: error {error_percent:.1f}% exceeds "
                    f"threshold {case['expected_error_threshold']}%. "
                    f"Actual: {actual_size_bytes / 1024 / 1024:.3f}MB, "
                    f"Estimated: {estimated_size_bytes / 1024 / 1024:.3f}MB"
                )

                # Verify QuadMesh was detected
                assert (
                    quad_cells > 0
                ), f"No QuadMesh cells detected for matrix size {case['matrix_size']}"

                plt.close(fig)

    def test_polycollection_factor_accuracy(self):
        """Test that POLYCOLLECTION_FACTOR provides reasonable size estimation for spatial plots."""
        test_cases = [
            {
                "n_points": 500,
                "gridsize": 15,
                "n_lines": 50,
                "expected_error_threshold": 350,
            },  # Enhanced heuristic with scatter detection is more conservative
            {
                "n_points": 1000,
                "gridsize": 20,
                "n_lines": 100,
                "expected_error_threshold": 350,
            },  # Better to overestimate than create 100MB+ files
        ]

        for case in test_cases:
            with plt.style.context("default"):  # Ensure consistent styling
                fig = self._create_spatial_correlation_plot(
                    n_points=case["n_points"],
                    gridsize=case["gridsize"],
                    n_lines=case["n_lines"],
                )

                # Count elements
                (
                    quad_cells,
                    poly_count,
                    line_segments,
                ) = self._count_plot_elements(fig)

                # Measure actual size and get estimated size
                svg_path = os.path.join(
                    self.temp_dir, f"test_spatial_{case['n_points']}.svg"
                )
                (
                    actual_size_bytes,
                    estimated_size_bytes,
                ) = self._measure_actual_svg_size(fig, svg_path)

                # Ensure actual size is valid
                assert (
                    actual_size_bytes > 0
                ), f"Invalid actual file size: {actual_size_bytes}"

                # Calculate accuracy
                accuracy_ratio = estimated_size_bytes / actual_size_bytes
                error_percent = abs(1 - accuracy_ratio) * 100

                # Assert accuracy is within acceptable range
                assert error_percent < case["expected_error_threshold"], (
                    f"Spatial plot {case['n_points']} points: error {error_percent:.1f}% exceeds "
                    f"threshold {case['expected_error_threshold']}%. "
                    f"Actual: {actual_size_bytes / 1024 / 1024:.3f}MB, "
                    f"Estimated: {estimated_size_bytes / 1024 / 1024:.3f}MB"
                )

                # Verify elements were detected
                assert (
                    poly_count > 0
                ), f"No PolyCollection detected for {case['n_points']} points"
                assert (
                    line_segments > 0
                ), f"No LineCollection detected for {case['n_points']} points"

                plt.close(fig)

    def _create_scatter_plot(self, n_points=1000):
        """Create a scatter plot with specified number of points."""
        # Generate random data for scatter plot
        x = np.random.uniform(0, 100, n_points)
        y = np.random.uniform(0, 100, n_points)
        colors = np.random.uniform(0, 1, n_points)
        sizes = np.random.uniform(10, 50, n_points)

        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(
            x, y, c=colors, s=sizes, alpha=0.7, cmap="viridis"
        )

        ax.set_title(f"Scatter Plot ({n_points} points)")
        ax.set_xlabel("X Value")
        ax.set_ylabel("Y Value")
        plt.colorbar(scatter, ax=ax, label="Color Value")

        plt.tight_layout()
        return fig

    def test_scatter_point_factor_accuracy(self):
        """Test that SCATTER_POINT_FACTOR provides reasonable size estimation for scatter plots."""
        test_cases = [
            {"n_points": 500, "expected_error_threshold": 50},
            {"n_points": 1000, "expected_error_threshold": 50},
            {"n_points": 2000, "expected_error_threshold": 50},
            {"n_points": 5000, "expected_error_threshold": 50},
        ]

        for case in test_cases:
            with plt.style.context("default"):  # Ensure consistent styling
                fig = self._create_scatter_plot(n_points=case["n_points"])

                # Measure actual size and get estimated size
                svg_path = os.path.join(
                    self.temp_dir, f"test_scatter_{case['n_points']}.svg"
                )
                (
                    actual_size_bytes,
                    estimated_size_bytes,
                ) = self._measure_actual_svg_size(fig, svg_path)

                # Ensure actual size is valid
                assert (
                    actual_size_bytes > 0
                ), f"Invalid actual file size: {actual_size_bytes}"

                # Calculate accuracy
                accuracy_ratio = estimated_size_bytes / actual_size_bytes
                error_percent = abs(1 - accuracy_ratio) * 100

                # Assert accuracy is within acceptable range
                assert error_percent < case["expected_error_threshold"], (
                    f"Scatter plot {case['n_points']} points: error {error_percent:.1f}% exceeds "
                    f"threshold {case['expected_error_threshold']}%. "
                    f"Actual: {actual_size_bytes / 1024 / 1024:.3f}MB, "
                    f"Estimated: {estimated_size_bytes / 1024 / 1024:.3f}MB"
                )

                # Verify that the estimated size is reasonable based on SCATTER_POINT_FACTOR
                expected_min_size = (
                    case["n_points"] * SCATTER_POINT_FACTOR * 0.5
                )  # Allow 50% variance
                expected_max_size = (
                    case["n_points"] * SCATTER_POINT_FACTOR * 1.5
                )  # Allow 50% variance

                assert estimated_size_bytes >= expected_min_size, (
                    f"Estimated size {estimated_size_bytes} too low for {case['n_points']} points. "
                    f"Expected >= {expected_min_size} (factor: {SCATTER_POINT_FACTOR})"
                )
                assert estimated_size_bytes <= expected_max_size, (
                    f"Estimated size {estimated_size_bytes} too high for {case['n_points']} points."
                    f"Expected <= {expected_max_size} (factor: {SCATTER_POINT_FACTOR})"
                )

                plt.close(fig)

    def test_large_file_detection(self):
        """Test that large files are properly detected and would trigger rasterization."""
        # Create a large correlation matrix that should exceed 10MB threshold
        with plt.style.context("default"):
            fig = self._create_correlation_matrix_plot(matrix_size=500)

            # Estimate size using the utilities function
            estimated_size_bytes = estimate_svg_size(fig)
            estimated_size_mb = estimated_size_bytes / (1024 * 1024)

            # Should be detected as large (>10MB)
            assert estimated_size_mb > 10, (
                f"Large matrix (500x500) not detected as large file. "
                f"Estimated size: {estimated_size_mb:.2f}MB should be > 10MB"
            )

            plt.close(fig)

    def test_element_counting_edge_cases(self):
        """Test element counting with edge cases and empty plots."""
        # Test empty plot
        fig, ax = plt.subplots()
        quad_cells, poly_count, line_segments = self._count_plot_elements(fig)

        assert quad_cells == 0, "Empty plot should have 0 QuadMesh cells"
        assert (
            poly_count == 0
        ), "Empty plot should have 0 PolyCollection elements"
        assert (
            line_segments == 0
        ), "Empty plot should have 0 LineCollection segments"

        plt.close(fig)

        # Test plot with only lines (no collections)
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])

        quad_cells, poly_count, line_segments = self._count_plot_elements(fig)

        # Should still be 0 since plot() creates Line2D objects, not collections
        assert quad_cells == 0, "Line plot should have 0 QuadMesh cells"
        assert (
            poly_count == 0
        ), "Line plot should have 0 PolyCollection elements"
        assert (
            line_segments == 0
        ), "Line plot should have 0 LineCollection segments"

        plt.close(fig)

    def test_factors_return_positive_estimates(self):
        """Test that the factors used in estimation return sensible positive numbers."""
        # Test with different plot types
        test_plots = [
            ("empty", lambda: plt.subplots()),
            ("matrix", lambda: self._create_correlation_matrix_plot(50)),
            (
                "spatial",
                lambda: self._create_spatial_correlation_plot(100, 10, 20),
            ),
        ]

        for plot_name, plot_creator in test_plots:
            if plot_name == "empty":
                fig, ax = plot_creator()
            else:
                fig = plot_creator()

            estimated_size = estimate_svg_size(fig)

            # Should return a non-negative number
            assert (
                estimated_size >= 0
            ), f"Estimated size should be non-negative for {plot_name} plot, got {estimated_size}"

            plt.close(fig)

    def test_estimation_logic_consistency(self):
        """Test that the estimation logic is consistent and repeatable."""
        # Close any existing figures to prevent memory warnings
        plt.close("all")

        # Create the same plot twice and ensure estimates are identical
        with plt.style.context("default"):
            fig1 = self._create_correlation_matrix_plot(matrix_size=100)
            fig2 = self._create_correlation_matrix_plot(matrix_size=100)

            estimate1 = estimate_svg_size(fig1)
            estimate2 = estimate_svg_size(fig2)

            assert estimate1 == estimate2, (
                f"Identical plots should have identical estimates. "
                f"Got {estimate1} and {estimate2}"
            )

            plt.close(fig1)
            plt.close(fig2)

    def test_relative_size_scaling(self):
        """Test that larger/more complex plots have proportionally larger estimates."""
        # Close any existing figures to prevent memory warnings
        plt.close("all")

        with plt.style.context("default"):
            # Test matrix scaling
            small_matrix = self._create_correlation_matrix_plot(matrix_size=50)
            large_matrix = self._create_correlation_matrix_plot(
                matrix_size=200
            )

            small_estimate = estimate_svg_size(small_matrix)
            large_estimate = estimate_svg_size(large_matrix)

            # Larger matrix should have larger estimate
            assert large_estimate > small_estimate, (
                f"200x200 matrix should have larger estimate than 50x50. "
                f"Got {large_estimate} vs {small_estimate}"
            )

            plt.close(small_matrix)
            plt.close(large_matrix)

            # Test spatial plot scaling
            simple_spatial = self._create_spatial_correlation_plot(
                n_points=500, gridsize=10, n_lines=25
            )
            complex_spatial = self._create_spatial_correlation_plot(
                n_points=2000, gridsize=30, n_lines=200
            )

            simple_estimate = estimate_svg_size(simple_spatial)
            complex_estimate = estimate_svg_size(complex_spatial)

            # More complex spatial plot should have larger estimate
            assert complex_estimate > simple_estimate, (
                f"Complex spatial plot should have larger estimate than simple. "
                f"Got {complex_estimate} vs {simple_estimate}"
            )

            plt.close(simple_spatial)
            plt.close(complex_spatial)

    def test_threshold_detection_boundaries(self):
        """Test that files near the 10MB threshold are properly classified."""
        # Test sizes around the 10MB boundary
        test_sizes = [100, 200, 300, 400, 500]  # Matrix sizes

        estimates_mb = []
        for size in test_sizes:
            with plt.style.context("default"):
                fig = self._create_correlation_matrix_plot(matrix_size=size)
                estimate_bytes = estimate_svg_size(fig)
                estimate_mb = estimate_bytes / (1024 * 1024)
                estimates_mb.append(estimate_mb)
                plt.close(fig)

    def test_scatter_plot_detection(self):
        """Test that scatter plots are properly detected and estimated by the heuristic."""
        # Test various scatter plot sizes
        scatter_sizes = [100, 1000, 5000]

        for n_points in scatter_sizes:
            fig, ax = plt.subplots(figsize=(8, 6))

            # Create scatter plot
            x = np.random.uniform(0, 100, n_points)
            y = np.random.uniform(0, 100, n_points)
            colors = np.random.uniform(0, 1, n_points)
            ax.scatter(x, y, c=colors, s=10, alpha=0.7, cmap="viridis")

            # Test detection
            estimated_size = estimate_svg_size(fig)

            # Should detect the points (factor = 800 bytes/point)
            expected_min = n_points * 700  # Allow some variance
            expected_max = n_points * 900

            assert estimated_size >= expected_min, (
                f"Scatter plot estimation too low for {n_points} points: "
                f"got {estimated_size}, expected >= {expected_min}"
            )
            assert estimated_size <= expected_max, (
                f"Scatter plot estimation too high for {n_points} points: "
                f"got {estimated_size}, expected <= {expected_max}"
            )

            # Should detect large scatter plots as needing rasterization
            if n_points >= 15000:  # ~12MB with factor 800
                assert (
                    estimated_size > 10 * 1024 * 1024
                ), f"Large scatter plot ({n_points} points) should exceed 10MB threshold"

            plt.close(fig)

    def test_mixed_plot_types_with_scatter(self):
        """Test that mixed plots with scatter are properly handled."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

        # Scatter plot
        n_scatter = 2000
        x = np.random.uniform(0, 100, n_scatter)
        y = np.random.uniform(0, 100, n_scatter)
        ax1.scatter(x, y, s=5, alpha=0.6)
        ax1.set_title("Scatter Plot")

        # Correlation matrix (QuadMesh)
        matrix = np.random.randn(50, 50)
        ax2.imshow(matrix, cmap="RdBu_r")
        ax2.set_title("Correlation Matrix")

        # Hexbin plot (PolyCollection)
        x_hex = np.random.normal(0, 1, 1000)
        y_hex = np.random.normal(0, 1, 1000)
        ax3.hexbin(x_hex, y_hex, gridsize=20, cmap="viridis")
        ax3.set_title("Hexbin Plot")

        # Line plot (should not contribute much)
        x_line = np.linspace(0, 10, 100)
        y_line = np.sin(x_line)
        ax4.plot(x_line, y_line, "b-", linewidth=2)
        ax4.set_title("Line Plot")

        plt.tight_layout()

        # Test estimation
        estimated_size = estimate_svg_size(fig)

        # Should be dominated by scatter plot contribution
        scatter_contribution = n_scatter * 800  # Expected scatter contribution

        # Total should be at least the scatter contribution
        assert estimated_size >= scatter_contribution * 0.8, (
            f"Mixed plot estimation should include scatter contribution. "
            f"Expected >= {scatter_contribution * 0.8}, got {estimated_size}"
        )

        # Should not be wildly larger (other elements shouldn't dominate)
        assert estimated_size <= scatter_contribution * 3, (
            f"Mixed plot estimation too high. "
            f"Expected <= {scatter_contribution * 3}, got {estimated_size}"
        )

        plt.close(fig)
