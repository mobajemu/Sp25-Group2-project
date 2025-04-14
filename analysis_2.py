# Import modules
from typing import List
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from data_loader import DataLoader
from model import Issue

class IssueReopenRate:

    # Function to extract the kind value from the labels list
    @staticmethod
    def _extract_kind(labels):
        if isinstance(labels, list):
            # Look for a label that starts with kind/
            for lab in labels:
                if lab.startswith('kind/'):
                    # Return everything after kind/
                    return lab.split('kind/')[1]
        return 'unknown'

    # Function to check if the reopened event label is present in each isuee
    @staticmethod
    def _check_reopened(issue:Issue):
        # Access the events from the row if present, else default to an empty list
        events = issue.events
        if isinstance(events, list):
            for event in events:
                if event.event_type == 'reopened':
                    return 1
        return 0

    # Function to plot reopen rate
    @staticmethod
    def _plot_reopen_rate(model, data, legend, X_train, X_test, y_train, y_test):
        title = "RandomForest_" if isinstance(model, RandomForestClassifier) else "LogisticRegression_"

        # Predictions, evaluation, and score
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nAccuracy: {accuracy*100:.2f}")

        data = data.copy()
        # Compute the predicted probability of reopening for each sample in the full dataset
        data['predicted_prob'] = model.predict_proba(data[['kind_encoded']])[:, 1]

        # Group the data by the encoded 'kind' and calculate the average predicted probability, map encoded values to their original kind names
        grouped_probs = data.groupby('kind_encoded')['predicted_prob'].mean().reset_index()
        grouped_probs['kind'] = grouped_probs['kind_encoded'].apply(lambda x: legend.get(x, 'unknown'))

        # Plot the bar chart
        plt.figure(figsize=(8, 6))
        plt.bar(grouped_probs['kind'], grouped_probs['predicted_prob'])
        plt.xlabel('Issue Kind')
        plt.ylabel('Predicted Probability of Reopen')
        plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f'{x*100:.0f}%'))
        plt.title(f'Predicted reopen probability by issue kind using {title}')
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save the figures
        os.makedirs('./out', exist_ok=True)
        plt.savefig(f'./out/{title}_reopenrate.png', dpi=300)
        print(f"Plots and score's saved successfully in ./out/")
        plt.show()

    def run(self):

        # Load data in
        issues:List[Issue] = DataLoader().get_issues()
        rows = []

        for issue in issues:
            rows.append({"kind": IssueReopenRate._extract_kind(issue.labels), "reopened": IssueReopenRate._check_reopened(issue)})

        # Load data into dataframe
        df = pd.DataFrame.from_records(rows)

        # Encode Kind label
        df['kind'] = df['kind'].astype(str)
        label_encoder = LabelEncoder()
        df['kind_encoded'] = label_encoder.fit_transform(df['kind'])

        # Print Kind legend
        kind_legend = dict(enumerate(label_encoder.classes_))
        print("Kind Legend:")
        for key, value in kind_legend.items():
            print(f"{key}: {value}")

        # Create new data frame of desired columns
        new_df = df[['kind_encoded', 'reopened']]
        print(new_df.head())

        # Set predictors and targets
        predictor = new_df[['kind_encoded']]
        target = new_df['reopened']

        # Train test split
        X_train, X_test, y_train, y_test = train_test_split(
            predictor, target,
            test_size=1/3, random_state=42
        )

        # Build model for Random forest and Logistic regression
        RFmodel = RandomForestClassifier(random_state=42)
        self._plot_reopen_rate(RFmodel, new_df, kind_legend, X_train, X_test, y_train, y_test)

        LRmodel = LogisticRegression(random_state=42)
        self._plot_reopen_rate(LRmodel, new_df, kind_legend, X_train, X_test, y_train, y_test)

if __name__ == '__main__':
    # Invoke run method when running this module directly
    IssueReopenRate().run()