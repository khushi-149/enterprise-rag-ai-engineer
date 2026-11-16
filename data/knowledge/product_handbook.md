# Acme Analytics Product Handbook

## Data refresh
Standard analytics dashboards refresh every four hours. Enterprise customers can configure hourly refresh for supported connectors. Failed refreshes are retried up to three times with exponential backoff.

## Export limits
CSV exports support up to 100,000 rows per file. Larger exports use an asynchronous export job. Download links expire after 24 hours.

## Workspace
A workspace is the top-level container for teams, users and projects. Each project belongs to exactly one workspace.
