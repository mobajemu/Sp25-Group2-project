# ENPM 611 Class Project - Group 2

Group Members:

- Kevin Zong
- Michael Obajemu
- Anh Vu

## Task 1: Domain Model

#### Entity Relationship Diagram

<img src="diagrams/erd.svg" alt="ERD" width="200">

## Task 2: Application Design

#### Class Diagram

<img src="diagrams/class_diagram.svg" alt="Class Diagram">

## Task 3: Implementation

### Analyses

#### Analyses 1: Average Resolution Time vs Number of comments

**Insight:** Do issues with more discussion get resolved faster or slower?
**Input:** Label (optional)
**Output:** Line graph

- X axis - Engagement (number of comments)
- Y axis - Average resolution time (days)
- Each dot represents an issue.
  - If no label is inputted, all issues will be plotted.
  - If a label is inputted, only issues with that label will be plotted.

Run with command:

```python
python run.py --feature 1
```

#### Analyses 2: Issue Reopen Rate

**Insight:** Which types of issues are most likely to be reopened?
**Input:** None
**Output:** Bar chart

- X axis - Labels
- Y axis - % of issues that get reopened

Run with command:

```python
python run.py --feature 2
```

#### Analyses 3: Top contributors' specialties

**Insight:** Who are the most active contributors and what types of issues do they spend their time on
**Input:** None required. Optionally, use --user to inspect data for a specific contributor.
**Output:**

- A ranked list of the top contributors based on total interactions (comments, labels, assignments, etc.).

- For each contributor:

  - Total number of contributions.

  - First and last contribution dates.

  - Duration of activity.

  - Breakdown of contributions by type.

  - (Graphically) Average contributions per day and performance compared to the regression model.

Run with command:

```python
python run.py --feature 3
```
