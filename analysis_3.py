from data_loader import DataLoader


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

    def record_event(self, event_type):
        for entry in self.frequency_activity:
            if entry["event_type"] == event_type:
                entry["count"] += 1
                return

    def __repr__(self):
        return f"<Contributor {self.name}: {self.frequency_activity}>"





class Analysis3:
    def __init__(self):
        self.issues_data = DataLoader().get_issues()

        # Create a set of all unique authors from events
        self.unique_authors = set()
        for issue in self.issues_data:
            for event in issue.events:
                if event.author:
                    self.unique_authors.add(event.author)

        # Initialize contributors hash table
        self.contributors = {
            author: Contributor(author)
            for author in self.unique_authors
        }

        # Record events into each contributor
        for issue in self.issues_data:
            for event in issue.events:
                if event.author and event.event_type:
                    self.contributors[event.author].record_event(event.event_type)

    def show_top_contributors(self, top_n=10):
        sorted_contributors = sorted(
            self.contributors.values(),
            key=lambda c: sum(entry["count"] for entry in c.frequency_activity),
            reverse=True
        )

        print(f"\nTop {top_n} most active contributors:")
        for i, c in enumerate(sorted_contributors[:top_n], start=1):
            total = sum(entry["count"] for entry in c.frequency_activity)
            print(f"\n{i}. {c.name} — Total Events: {total}")
            print("   Breakdown by event type:")
            for entry in c.frequency_activity:
                if entry["count"] > 0:
                    print(f"     - {entry['event_type']}: {entry['count']}")







analysis = Analysis3()
analysis.show_top_contributors()
