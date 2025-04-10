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

    def get_unique_items(self):
        unique_creators = set()
        unique_author = set()
        unique_events = set()
        for issue in self.issues_data:
            if issue.creator:
                unique_creators.add(issue.creator)
            
            for event in issue.events:
                unique_events.add(event.event_type)
                unique_author.add(event.author)


        
        print(f"Total unique creators: {len(unique_creators)}")
        print(f"Total unique events: {len(unique_events)}")
        print(f"Total unique authors: {len(unique_author)}")

        if unique_creators.issubset(unique_author):
            print("✅ All creators are also authors.")
        else:
            missing = unique_creators - unique_author
            print("❌ Some creators are not in authors:")
            print(missing)

    




analysis = Analysis3()
analysis.get_unique_items()


