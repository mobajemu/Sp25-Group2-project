
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime

from data_loader import DataLoader
from model import Issue,Event,State
import config

class EngagementResolutionAnalysis:
    def __init__(self):
        self.LABEL:str = config.get_parameter('label')

    def run(self):
        issues:List[Issue] = DataLoader().get_issues()
        rows = []

        for issue in issues:
            if issue.state == State.closed:
                resolution_time = self._get_resolution_time(issue)
                num_comments = self._get_comment_count(issue)

                if resolution_time is None:
                    continue

                rows.append({"resolution_time": resolution_time, "num_comments": num_comments, "labels": issue.labels})

        df = pd.DataFrame.from_records(rows)

        if self.LABEL:
            df = df[df["labels"].apply(lambda labels: self.LABEL in labels)]

        self._show_plot(df)

    def _get_comment_count(self, issue: Issue) -> int:
        comment_count = 0

        for event in issue.events:
            if event.event_type == "commented":
                comment_count += 1

        return comment_count

    def _get_resolution_time(self, issue: Issue) -> int:
        created = issue.created_date

        # Find when the issue was closed
        # We want to understand the time to the final resolution of the issue,
        # which is why we iterate through the events in reverse chronological order
        for event in reversed(issue.events):
            if event.event_type == "closed" and event.event_date is not None: # if the event_date is null, skip the issue
                closed = event.event_date
                resolution_time = (closed - created).days
                return resolution_time

        return None

    def _show_plot(self, df):
        if self.LABEL:
            title = f"Resolution Time vs. Number of Comments for '{self.LABEL}' Issues"
        else:
            title = "Resolution Time vs. Number of Comments"

        grouped = df.groupby("num_comments")["resolution_time"].mean().reset_index()
        plt.plot(grouped["num_comments"], grouped["resolution_time"], marker="o")
        plt.xlabel("Number of Comments")
        plt.ylabel("Average Resolution Time (days)")
        plt.title(title)
        plt.show()

if __name__ == '__main__':
    # Invoke run method when running this module directly
    EngagementResolutionAnalysis().run()