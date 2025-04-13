# Import modules
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from data_loader import DataLoader
from model import Issue,Event,State
import config

class IssueReopenRate:
    # Function to extract the kind value from the 'labels' list.
    def _extract_kind(labels):
        if isinstance(labels, list):
            # Look for a label that starts with 'kind/'
            for lab in labels:
                if lab.startswith('kind/'):
                    # Return everything after "kind/"
                    return lab.split('kind/')[1]
        return 'unknown'

    # Function to determine if the 'reopened' event label is present.
    def _check_reopened(row):
        # Access the events from the row if present, else default to an empty list.
        events = row.get('events', [])
        if isinstance(events, list):
            for event in events:
                if event.get('event_type') == 'reopened':
                    return 1
        return 0

    # Function to plot reopen rate
    def _plot_reopen_rate(model, data, legend, X_train, X_test, y_train, y_test):

        title = "Random Forest Model" if model == RFmodel else "Logsitic Regression"

        # Predictions, evaluation, and score
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nAccuracy: {accuracy*100:.2f}")

        # We compute the predicted probability of reopening for each sample in the full dataset
        data['predicted_prob'] = model.predict_proba(data[['kind_encoded']])[:, 1]

        # Group the data by the encoded 'kind' and calculate the average predicted probability
        grouped_probs = data.groupby('kind_encoded')['predicted_prob'].mean().reset_index()

        # Map encoded values to their original kind names
        grouped_probs['kind'] = grouped_probs['kind_encoded'].apply(lambda x: legend.get(x, 'unknown'))

        # Plot the bar chart using the actual kind names for the x-axis labels
        plt.figure(figsize=(8, 6))
        plt.bar(grouped_probs['kind'], grouped_probs['predicted_prob'])
        plt.xlabel('Issue Kind')
        plt.ylabel('Predicted Probability of Reopen')
        plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f'{x*100:.0f}%'))
        plt.title('Predicted reopen probability by issue kind')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        # Save the figures
        os.makedirs('./out', exist_ok=True)
        plt.savefig(f'./out/{title}_reopenrate.png', dpi=300)

        print(f"Plots and score's saved successfully in ./out/")
        return plt

    # Load the JSON data into a DataFrame
    df = DataLoader().get_issues()

    # Create new data frame for the 'kind' and reopened.
    df['kind'] = df['labels'].apply(_extract_kind)
    df['reopened'] = df.apply(_check_reopened, axis=1)

    # Ensure 'kind' columns are strings
    df['kind'] = df['kind'].astype(str)

    # Apply LabelEncoder to 'kind' and 'state' columns
    label_encoder = LabelEncoder()
    df['kind_encoded'] = label_encoder.fit_transform(df['kind'])

    # Create a legend for 'kind' labels
    kind_legend = dict(enumerate(label_encoder.classes_))
    print("Kind Legend:")
    for key, value in kind_legend.items():
        print(f"{key}: {value}")

    # Build the  DataFrame with the selected columns: kind, state, text_length, time_diff_min, and reopened.
    new_df = df[['kind_encoded', 'reopened']]

    # Display the first few rows of the  DataFrame
    print( new_df.head())

    # Update predictors to use the encoded columns
    predictor = new_df[['kind_encoded']]
    target = new_df['reopened']

    # Split and test the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        predictor, target,
        test_size=1/3, random_state=42
    )

    # Build and plot models
    RFmodel = RandomForestClassifier(random_state=42)
    RF_fig = _plot_reopen_rate(RFmodel, new_df, kind_legend, X_train, X_test, y_train, y_test)
    plt.show()

    LRmodel = LogisticRegression(random_state=42)
    LR_fig = _plot_reopen_rate(LRmodel, new_df, kind_legend, X_train, X_test, y_train, y_test)
    plt.show()
    
if __name__ == '__main__':
    # Invoke run method when running this module directly
    IssueReopenRate().run()