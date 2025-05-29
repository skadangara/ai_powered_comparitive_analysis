import glob
import pandas as pd
import time
import json
import pyarrow.parquet as pq
from langgraph.graph import StateGraph
import llm_lib as llm_lib
import cleaning_lib as cleaning_lib
import warnings
import os
warnings.filterwarnings("ignore")


class WorkflowState:
    def __init__(self, df=None, comparison_dimensions=None, batch_size=None, dim_cnt=None):
        self.df = df
        self.batch_size = 1000
        self.dim_cnt = 6
        self.comparison_dimensions = comparison_dimensions


# Agent:  Load dataset
def load_data(state: WorkflowState):
    try:
        df = pq.read_table("case_study_data.parquet").to_pandas()
        state.df = df
    except Exception as e:
        print(f"Error loading data: {e}")
    return state


#  Filter unlabelled projects
def filter_zkp_projects(state: WorkflowState):
    if state.df is not None:
        unlabelled_df = state.df[state.df['label'].isna()]
        unlabelled_df.to_parquet("unlabelled_data.parquet", index=False, engine='pyarrow')
    return state


# Identify zkp and non-zkp projects using llm agent
def identify_zkp_projects(state: WorkflowState):
    if state.df is not None:
        file_path = 'unlabelled.parquet'
        if os.path.exists(file_path):
            try:
                parquet_file = pq.ParquetFile("unlabelled.parquet")
                i = 1
                for batch in parquet_file.iter_batches(state.batch_size):
                    chunk = batch.to_pandas()
                    chunk['raw_label'] = chunk['readme'].apply(lambda x: llm_lib.call_llm_for_zkp(x))
                    chunk.to_parquet("data/zkp_projects_pred_" + str(i) + ".parquet", index=False, engine='pyarrow')
                    i = i + 1
                    time.sleep(3)
                parquet_files = glob.glob("data/*.parquet")
                batch_df = pd.concat([pd.read_parquet(file) for file in parquet_files], ignore_index=True)
                batch_df.to_parquet("final_zkp_predictions.parquet", index=False, engine='pyarrow')
            except Exception as e:
                print(f"Error in identifying zkp project {e}")
    return state


# Extract zkp and non-zkp labels from llm predictions.
def curate_zkp_projects_predictions(state: WorkflowState):
    if state.df is not None:
        file_path = 'final_zkp_predictions.parquet'
        if os.path.exists(file_path):
            try:
                parquet_file = pq.ParquetFile("final_zkp_predictions.parquet")
                i = 1
                for batch in parquet_file.iter_batches(state.batch_size):
                    chunk = batch.to_pandas()
                    chunk['label'] = chunk['raw_label'].apply(lambda x: cleaning_lib.clean_zkp_predicted_label(x))
                    chunk.to_parquet("curated_data/zkp_projects_pred_curated_" + str(i) + ".parquet", index=False,
                                     engine='pyarrow')
                    i = i + 1
                    time.sleep(2)
                parquet_files = glob.glob("curated_data/*.parquet")
                batch_df = pd.concat([pd.read_parquet(file) for file in parquet_files], ignore_index=True)
                batch_df.to_parquet("curated_zkp_predictions.parquet", index=False, engine='pyarrow')
                state.zkp_projects = batch_df
            except Exception as e:
                print(f"Error in curating zkp project {e}")
    return state


# Identify dimensions from verified zkp projects using llm
def identify_zkp_dimensions(state: WorkflowState):
    if state.df is not None:
        dim_len = input("Please enter the number of dimensions to be identified: ")
        if dim_len.isdigit():
            dim_len = int(dim_len)
            state.dim_cnt = dim_len
        else:
            dim_len = state.dim_cnt
        file_path = 'curated_zkp_predictions.parquet'
        if os.path.exists(file_path):
            try:
                parquet_file = pq.ParquetFile("curated_zkp_predictions.parquet")
                i = 1
                for batch in parquet_file.iter_batches(state.batch_size):
                    chunk = batch.to_pandas()
                    chunk = chunk[chunk['label'] == "yes"]
                    chunk['dimensions_raw'] = chunk['readme'].apply(
                        lambda x: llm_lib.call_llm_for_dimension(x, dim_len))
                    chunk.to_parquet("dimensions/zkp_projects_dim_" + str(i) + ".parquet", index=False,
                                     engine='pyarrow')
                    i = i + 1
                    time.sleep(2)
                parquet_files = glob.glob("dimensions/*.parquet")
                batch_df = pd.concat([pd.read_parquet(file) for file in parquet_files], ignore_index=True)
                batch_df.to_parquet("zkp_projects_dimensions.parquet", index=False, engine='pyarrow')
                state.zkp_projects_dimensions = batch_df
            except Exception as e:
                print(f"Error in identifying dimensions {e}")
    return state


# Identify comparable N dimensions from all zkp project dimensions identified.
def identify_common_dimensions(state: WorkflowState):
    file_path = 'zkp_projects_dimensions.parquet'
    if os.path.exists(file_path):
        try:
            parquet_file = pq.ParquetFile("zkp_projects_dimensions.parquet")
            dim_df = parquet_file.read().to_pandas()
            combined_dimensions = ''.join(dim_df['dimensions_raw'])
            common_dimensions = llm_lib.call_llm_for_common_dimensions(combined_dimensions, state.dim_cnt)
            state.comparison_dimensions = common_dimensions
        except Exception as e:
            print(f"Error in identifying common dimensions {e}")
    return state


# Extract comparable N dimensions details from the zkp projects
def extract_common_dimensions(state: WorkflowState):
    file_path = 'zkp_projects_dimensions.parquet'
    if os.path.exists(file_path):
        try:
            parquet_file = pq.ParquetFile("zkp_projects_dimensions.parquet")
            i = 1
            for batch in parquet_file.iter_batches(state.batch_size):
                chunk = batch.to_pandas()
                chunk['extracted_dimensions'] = chunk['readme'].apply(
                    lambda x: llm_lib.call_llm_for_extract_dimensions(x, state.dim_cnt, state.comparison_dimensions))
                chunk.to_parquet("extract_dimensions/zkp_projects_extracted_dim_" + str(i) + ".parquet", index=False,
                                 engine='pyarrow')
                i = i + 1
                time.sleep(2)
            parquet_files = glob.glob("extract_dimensions/*.parquet")
            batch_df = pd.concat([pd.read_parquet(file) for file in parquet_files], ignore_index=True)
            batch_df.to_parquet("zkp_predictions_extracted_dimensions.parquet", index=False, engine='pyarrow')
        except Exception as e:
            print(f"Error in extracting common dimensions {e}")
    return state


# Generate comparison table in json format
def generate_comparison_table(state: WorkflowState):
    file_path = 'zkp_predictions_extracted_dimensions.parquet'
    if os.path.exists(file_path):
        parquet_file = pq.ParquetFile("zkp_predictions_extracted_dimensions.parquet")
        try:
            dim_df = parquet_file.read().to_pandas()
            data_json_comm_dim = json.loads(state.comparison_dimensions)
            dim_len = state.dim_cnt
            dim_list = []
            for i in range(dim_len):
                dim_list.append(data_json_comm_dim['dimensions'][i]['dimension'])
            dim_df['comparison_dim'] = dim_df.apply(lambda x:
                                                    cleaning_lib.curate_extracted_dimensions(dim_list, x['full_name'],
                                                                                             x['extracted_dimensions']),
                                                    axis=1)
            dim_df.to_parquet("project_comparison_dimensions.parquet", index=False, engine='pyarrow')
            dim_df.to_excel("project_comparison_dimensions.xlsx", index=False)

            filtered_df = dim_df[dim_df['comparison_dim'].apply(lambda x: isinstance(x, dict) and len(x) > 1)]
            filtered_df.to_parquet("project_comparison_dimensions_filtered.parquet", index=False, engine='pyarrow')
            filtered_df.to_excel("project_comparison_dimensions_filtered.xlsx", index=False)
            filtered_df['comparison_dim'] = filtered_df['comparison_dim'].astype(str)

            combined_data = ','.join(filtered_df['comparison_dim'])
            new_data = "[" + combined_data + "]"
            correct_json = new_data.replace("'", '"')
            combined_data_json = json.dumps(correct_json)
            with open("zkp_comparison.json", "w") as f:
                json.dump(correct_json, f, indent=4)
            print("Comparison table saved to zkp_comparison.json")
        except Exception as e:
            print(f"Error in generating comparison table {e}")
    return state


# Define LangGraph Workflow
graph = StateGraph(WorkflowState)
graph.add_node("load_data", load_data)
graph.add_node("filter_zkp_projects", filter_zkp_projects)
graph.add_node("identify_zkp_projects", identify_zkp_projects)
graph.add_node("curate_zkp_projects_predictions", curate_zkp_projects_predictions)
graph.add_node("identify_zkp_dimensions", identify_zkp_dimensions)
graph.add_node("identify_common_dimensions", identify_common_dimensions)
graph.add_node("extract_common_dimensions", extract_common_dimensions)  #
graph.add_node("generate_comparison_table", generate_comparison_table)

# Define edges
graph.add_edge("load_data", "filter_zkp_projects")
graph.add_edge("filter_zkp_projects", "identify_zkp_projects")
graph.add_edge("identify_zkp_projects", "curate_zkp_projects_predictions")

graph.add_edge("curate_zkp_projects_predictions", "identify_zkp_dimensions")
graph.add_edge("identify_zkp_dimensions", "identify_common_dimensions")
graph.add_edge("identify_common_dimensions", "extract_common_dimensions")
graph.add_edge("extract_common_dimensions", "generate_comparison_table")
graph.set_entry_point("load_data")
executor = graph.compile()

# Run the workflow
if __name__ == "__main__":
    executor.invoke(WorkflowState())
