# JIRA Resource Management App with AI PM Buddy

An advanced AI-powered JIRA Resource Management application designed to transform project management through intelligent insights, dynamic simulation, and interactive reporting.

## Features

- **Interactive Dashboards**: Visualize project metrics, resource allocation, and workload.
- **AI PM Buddy**: AI-powered assistant for project management guidance and strategic insights.
- **Technical Debt & Risk Management**: Track and manage technical debt and project risks.
- **Sprint Planning Assistant**: Intelligent sprint planning with capacity analysis and composition recommendations.
- **Advanced Leave Conflict Detection**: Identify and resolve conflicts between leaves and project deadlines.
- **Task Redistribution**: AI-based recommendations for optimal task assignments.

## Deployment on Streamlit Cloud

1. Fork this repository
2. Add your `OPENAI_API_KEY` to the Streamlit Cloud secrets
3. Deploy the app on Streamlit Cloud

## Setting up Secrets

Rename `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and add your OpenAI API key:

```toml
OPENAI_API_KEY = "your-actual-openai-api-key"
```

In Streamlit Cloud, add this as a secret in the app settings.

## Local Development

1. Install dependencies: `pip install -r requirements-deploy.txt`
2. Set up the secrets as described above
3. Run the app: `streamlit run app.py`

## Sample Data

The app includes sample data files for demonstration purposes:
- `enriched_jira_project_data.xlsx`: Basic JIRA project data
- `enriched_jira_data_with_simulated.xlsx`: Extended data with technical debt and other worksheets

## Project Structure

- `app.py`: Main application file
- `utils.py`: Utility functions for data loading and processing
- `*.py`: Feature-specific modules (AI task redistribution, leave conflict detection, etc.)
- `.streamlit/`: Configuration files for Streamlit
