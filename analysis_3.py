from datetime import datetime
from data_loader import DataLoader
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

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

        self.unique_authors = set()
        for issue in self.issues_data:
            for event in issue.events:
                if event.author and "[bot]" not in event.author:
                    self.unique_authors.add(event.author)

        self.contributors = {
            author: Contributor(author)
            for author in self.unique_authors
        }

        for issue in self.issues_data:
            for event in issue.events:
                if not event.author or "[bot]" in event.author or not event.event_type:
                    continue  # skip bot or malformed events
                self.contributors[event.author].record_event(event.event_type, event.event_date)

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
        # Fit a linear regression model
        self.model = LinearRegression()
        self.model.fit(self.X, self.y)

        # Get predictions
        predictions = self.model.predict(self.X)

        # Print out the regression line coefficients
        print(f"Linear Regression Model: y = {self.model.coef_[0]} * X + {self.model.intercept_}")

        # Plot the data and the regression line
        plt.figure(figsize=(8, 6))
        plt.scatter(self.X, self.y, color='blue', label='Actual data')
        plt.plot(self.X, predictions, color='red', linewidth=2, label='Regression line')
        plt.title('Linear Regression Fit')
        plt.xlabel('Contributions')
        plt.ylabel('Number of Days')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def predict_contribution_for_active_days(self, day_active):
        predicted_contributions = self.model.predict([[day_active]])  
        return predicted_contributions[0]

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

    def find_most_contributor_ratio(self, num_of_contributors): 
        contributors_with_ratios = []

        for contributor in self.contributors.values():
            total_contributions = sum(entry["count"] for entry in contributor.frequency_activity)
            
            # Check if first and last dates exist
            if contributor.first_date and contributor.last_date:
                duration_in_days = (contributor.last_date - contributor.first_date).days

                # Skip contributors with zero active duration
                if duration_in_days == 0:
                    continue

                # Predict contributions based on active days
                predicted_contributions = self.predict_contribution_for_active_days(duration_in_days)

                # Avoid division by zero
                if total_contributions > 0:
                    ratio = total_contributions / predicted_contributions
                    # Append name, total contributions, and the ratio to the list
                    contributors_with_ratios.append((contributor.name, total_contributions, ratio, duration_in_days))

        # Sort contributors based on the ratio
        sorted_contributors = sorted(contributors_with_ratios, key=lambda x: x[2], reverse=True)

        # Print top contributors based on ratio
        print("\nTop 3 Contributors by Predicted-to-Actual Contributions Ratio:")
        for i, (name, total_contributions, ratio, duration_in_days) in enumerate(sorted_contributors[:num_of_contributors], start=1):
            print(f"{i}. {name} - total_contributions: {total_contributions} - Duration: {duration_in_days} - Ratio: {ratio:.2f}")




# Run the analysis
analysis = Analysis3()
analysis.show_top_contributors()
analysis.apply_linear_regression()
analysis.find_most_contributor_ratio(3)
