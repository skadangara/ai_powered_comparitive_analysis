
# Project Title

Automated Comparative Analysis of Github READMEs.

# Prerequisites
Install necessary libraries from the requirements.txt file.

# Setup
- Extract the source code
- Install from the requirements file.
- RUN agent_workflow.py file.

# Overall Architecture
- Step1:  Load data from the source and filter out the unlabelled data
- Step2: Identify zkp and non-zkp projects from the readme content using LLM.
- Step3: Extract zkp projects identified in step2, use LLM to identify dimensions from the readme content.
- Step4: Combine all zkp projects dimensions from step4, and use LLM to identify N common dimensions which can be used for the project comparison.
- Step5: Extract N common dimension's details from all the zkp projects using LLM.

# Tools & Technologies
- Programming Language : Python
- LLM : GPT-4o (OpenAI)
- Framework : LangGraph

# Comparable Dimensions identified
- Programming Language
- Licensing
- Community and Support
- Installation Process
- Deployment Process
- Testing Framework

# Author
- Sajana Kadangara

# License
[Sajana]


