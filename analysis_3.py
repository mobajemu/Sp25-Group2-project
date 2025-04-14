from datetime import datetime
from data_loader import DataLoader
import numpy as np
from sklearn.linear_model import LinearRegression, RANSACRegressor
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

class Contributor:
    def __init__(self, name):
        ALL_EVENT_TYPES = [
            "closed", "comment_deleted", "reopened", "labeled", "converted_to_discussion",
            "renamed", "unsubscribed", "commented", "unassigned", "unlabeled",
            "cross-referenced", "locked", "referenced", "subscribed", "milestoned",
            "demilestoned", "marked_as_duplicate", "mentioned", "connected", "assigned",
            "transferred", "pinned"
        ]
        self.name = name
        self.frequency_activity = [
            {"event_type": event, "count": 0} for event in ALL_EVENT_TYPES
        ]
        self.first_date = None
        self.last_date = None
        self.total_contributions = 0  # Track total contributions for each contributor

    def record_event(self, event_type, event_date):
        for entry in self.frequency_activity:
            if entry["event_type"] == event_type:
                entry["count"] += 1
                break

        # Update total contributions
        self.total_contributions = sum(entry["count"] for entry in self.frequency_activity)

        # Use event_date directly since it's already a datetime object
        if self.first_date is None or event_date < self.first_date:
            self.first_date = event_date
        if self.last_date is None or event_date > self.last_date:
            self.last_date = event_date

    def get_first_date_str(self):
        return self.first_date.strftime("%Y-%m-%d") if self.first_date else "N/A"

    def get_last_date_str(self):
        return self.last_date.strftime("%Y-%m-%d") if self.last_date else "N/A"

    def __repr__(self):
        return f"<Contributor {self.name}: {self.frequency_activity}>"

class Analysis3:
    def __init__(self):
        self.issues_data = DataLoader().get_issues()
        self.contributors = {}
        
        for issue in self.issues_data:
            for event in issue.events:
                author = event.author
                if not author or "[bot]" in author or not event.event_type:
                    continue  # skip bots or malformed events

                if author not in self.contributors:
                    self.contributors[author] = Contributor(author)

                self.contributors[author].record_event(event.event_type, event.event_date)

        # Prepare data for regression
        self.X = []  # Active duration (in days)
        self.y = []  # Total contributions

        for contributor in self.contributors.values():
            if contributor.first_date and contributor.last_date:
                duration_in_days = (contributor.last_date - contributor.first_date).days
                self.X.append([duration_in_days])  # Active duration as a feature
                self.y.append(contributor.total_contributions)  # Total contributions as the target variable

        # Convert to numpy arrays
        self.X = np.array(self.X)
        self.y = np.array(self.y)
   
    def apply_linear_regression(self):
        # --- ORIGINAL MODEL ---
        self.original_model = LinearRegression()
        self.original_model.fit(self.X, self.y)
        self.predictions_original = self.original_model.predict(self.X)
        coef_orig = self.original_model.coef_[0]
        intercept_orig = self.original_model.intercept_
        print(f"[Original] y = {coef_orig} * X + {intercept_orig}")
        # Calculate minimum days for non-negative prediction
        self.min_days_original = max(0, -intercept_orig / coef_orig) if coef_orig != 0 else float('inf')
        print(f"[Original] Minimum active days for y ≥ 0: {self.min_days_original:.2f}")

        # --- FILTERED MODEL (±30%) ---
        relative_error = np.abs((self.y - self.predictions_original) / self.y)
        mask = relative_error <= 0.3
        self.X_filtered = self.X[mask]
        self.y_filtered = self.y[mask]
        self.model_filtered = LinearRegression()
        self.model_filtered.fit(self.X_filtered, self.y_filtered)
        self.predictions_filtered =self.model_filtered.predict(self.X_filtered)
        coef_filtered = self.model_filtered.coef_[0]
        intercept_filtered = self.model_filtered.intercept_
        print(f"[Filtered ±30%] y = {coef_filtered} * X + {intercept_filtered}")
        # Calculate minimum days for non-negative prediction
        self.min_days_filtered = max(0, -intercept_filtered / coef_filtered) if coef_filtered != 0 else float('inf')
        print(f"[Filtered ±30%] Minimum active days for y ≥ 0: {self.min_days_filtered:.2f}")
    
    def plot_top_contributors(self, ax, top_n=10):
        # Sort contributors by total contributions
        top = sorted(self.contributors.values(), key=lambda c: c.total_contributions, reverse=True)[:top_n]

        names = []
        totals = []
        ratios = []

        for c in top:
            duration_days = (c.last_date - c.first_date).days if c.first_date and c.last_date else None
            ratio = c.total_contributions / duration_days if duration_days and duration_days > 0 else 0
            names.append(c.name)
            totals.append(c.total_contributions)
            ratios.append(ratio)

        # Create horizontal bar chart
        bars = ax.barh(names[::-1], totals[::-1], color='skyblue')  # Reverse for highest at top
        ax.set_title(f'Top {top_n} Contributors By Total Interactions')
        ax.set_xlabel('Total Contributions')
        ax.set_ylabel('Contributor')
        ax.grid(True, axis='x', linestyle='--', alpha=0.7)

        # Add contribution/day ratio next to bars
        for i, bar in enumerate(bars):
            ax.text(
                bar.get_width() + 1,  # x position just to the right of the bar
                bar.get_y() + bar.get_height() / 2,
                f"{ratios[::-1][i]:.2f}/day",
                va='center',
                fontsize=9,
                color='gray'
            )

        return ax       

    def plot_original(self, ax):
        ax.scatter(self.X, self.y, color='blue', label='Actual')
        ax.plot(self.X, self.predictions_original, color='red', label='Regression')
        ax.set_title('Original')
        ax.set_xlabel('Days Active')
        ax.set_ylabel('Contributions')
        ax.legend()
        ax.grid(True)
        return ax
    
    def plot_filtered(self, ax):
        ax.scatter(self.X_filtered, self.y_filtered, color='blue', label='Filtered')
        ax.plot(self.X_filtered, self.predictions_filtered, color='red', label='Regression')
        ax.set_title('Filtered (±30%)')
        ax.set_xlabel('Days Active')
        ax.set_ylabel('Contributions')
        ax.legend()
        ax.grid(True)
        return ax

    def plot_top_contributors_by_ratio(self, ax, top_n=10, use_filtered=True):
        model = self.model_filtered if use_filtered else self.original_model

        contributor_data = []
        for name, c in self.contributors.items():
            if c.first_date and c.last_date:
                days_active = (c.last_date - c.first_date).days
                if days_active > self.min_days_original and c.total_contributions > 0:
                    contributor_data.append((name, days_active, c.total_contributions))

        if not contributor_data:
            print("No valid contributor data to plot.")
            return ax

        # Prepare inputs
        names = [x[0] for x in contributor_data]
        X_input = np.array([x[1] for x in contributor_data]).reshape(-1, 1)
        actuals = np.array([x[2] for x in contributor_data])

        # Predict contributions
        predictions = model.predict(X_input)

        # Calculate ratio = predicted / actual
        ratios = predictions / actuals

        # Combine and sort by lowest ratio
        combined = list(zip(names, predictions, actuals, ratios))
        top = sorted(combined, key=lambda x: x[3])[:top_n]

        # Plotting
        top_names = [x[0] for x in top]
        top_ratios = [x[3] for x in top]

        bars = ax.barh(top_names[::-1], top_ratios[::-1], color='purple')
        title_suffix = "(Filtered)" if use_filtered else "(Original)"
        ax.set_title(f'Top Contributors by Predicted/Actual Ratio {title_suffix}')
        ax.set_xlabel('Predicted / Actual')
        ax.set_ylabel('Contributor')
        ax.grid(True, axis='x', linestyle='--', alpha=0.7)

        for i, bar in enumerate(bars):
            ax.text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{top_ratios[::-1][i]:.2f}",
                va='center',
                fontsize=9,
                color='gray'
            )

        return ax

    def plot_all_regression_variants(self):
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 2, figure=fig)

        # First row
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])

        # Second row
        ax3 = fig.add_subplot(gs[1, 1])
        ax4 = fig.add_subplot(gs[1, 0])

        # Third row (spans both columns)
        ax5 = fig.add_subplot(gs[2, :])

        # Plotting
        self.plot_original(ax1)
        self.plot_filtered(ax2)
        self.plot_top_contributors_by_ratio(ax3, use_filtered=True)
        self.plot_top_contributors_by_ratio(ax4, use_filtered=False)
        self.plot_top_contributors(ax5, 10)

        plt.tight_layout()
        plt.show()

    def show_top_contributors(self, top_n=10):
        sorted_contributors = sorted(
            self.contributors.values(),
            key=lambda c: c.total_contributions,
            reverse=True
        )

        print(f"\nTop {top_n} most active contributors:")
        for i, c in enumerate(sorted_contributors[:top_n], start=1):
            first_date_str = c.get_first_date_str()
            last_date_str = c.get_last_date_str()

            if c.first_date and c.last_date:
                duration_days = (c.last_date - c.first_date).days
            else:
                duration_days = "N/A"

            print(f"\n{i}. {c.name}")
            print(f"   Total Contributions: {c.total_contributions}")
            print(f"   First Contribution:  {first_date_str}")
            print(f"   Last Contribution:   {last_date_str}")
            print(f"   Active Duration:     {duration_days} days")
            print("   Breakdown by event type:")
            for entry in c.frequency_activity:
                if entry["count"] > 0:
                    print(f"     - {entry['event_type']}: {entry['count']}")

    def show_contributor_by_name(self, name):
        contributor = self.contributors.get(name)

        if not contributor:
            print(f"\nContributor '{name}' not found.")
            return

        first_date_str = contributor.get_first_date_str()
        last_date_str = contributor.get_last_date_str()

        if contributor.first_date and contributor.last_date:
            duration_days = (contributor.last_date - contributor.first_date).days
        else:
            duration_days = "N/A"

        print(f"\nContributor: {contributor.name}")
        print(f"   Total Contributions: {contributor.total_contributions}")
        print(f"   First Contribution:  {first_date_str}")
        print(f"   Last Contribution:   {last_date_str}")
        print(f"   Active Duration:     {duration_days} days")
        print("   Breakdown by event type:")
        for entry in contributor.frequency_activity:
            if entry["count"] > 0:
                print(f"     - {entry['event_type']}: {entry['count']}")

    def run_all_analysis(self, user=None):
        # Show ranking of contributors by number of interactions
        self.show_top_contributors()

        # Search for a contributor by name if provided
        if user:
            self.show_contributor_by_name(user)

        # Apply linear regression and plot results
        self.apply_linear_regression()
        self.plot_all_regression_variants()

