
# Project Title

AI-Powered Automated Comparative Analysis of GitHub READMEs.

# Summary
This project automates the comparative analysis of GitHub README files using AI, with a particular focus on identifying and analyzing Zero-Knowledge Proof (ZKP) related projects. Leveraging the GPT-4o language model and built on the LangGraph framework, the system follows a structured workflow: it filters raw data, classifies projects, extracts relevant information, and determines common comparison dimensions such as licensing, programming language, community support, and testing frameworks. Developed in Python, this AI-driven tool streamlines the extraction of standardized insights from diverse README formats, enabling efficient project evaluation and comparison.

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


