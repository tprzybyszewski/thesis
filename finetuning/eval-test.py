import matplotlib.pyplot as plt
import matplotlib
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import pandas as pd
import numpy as np
import json
from mistralai import Mistral
import time

client = Mistral(api_key="fr")
def remove_hashes_and_asterisks(text):
    cleaned_text = text.replace('###', '').replace('**', '')
    return cleaned_text

def calculate_rouge(reference, prediction):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return {
        'ROUGE-1': scores['rouge1'].fmeasure,
        'ROUGE-2': scores['rouge2'].fmeasure,
        'ROUGE-L': scores['rougeL'].fmeasure
    }

def calculate_bertscore(reference_list, prediction_list):
    _, _, F1 = bert_score(prediction_list, reference_list, lang="en")
    return np.mean(F1.tolist()) 

def evaluate_models(prompts, correct_answers, fine_tuned_responses, standard_model_responses):
    results = {
        "Model": [],
        "Prompt": [],
        "ROUGE-1": [],
        "ROUGE-2": [],
        "ROUGE-L": [],
        "BERTScore": []
    }

    for idx in range(len(prompts)):
        rouge_scores_fine = calculate_rouge(correct_answers[idx], fine_tuned_responses[idx])
        bertscore_fine = calculate_bertscore([correct_answers[idx]], [fine_tuned_responses[idx]])

        results["Model"].append("Fine-Tuned Model")
        results["Prompt"].append(f"Prompt {idx + 1}")
        results["ROUGE-1"].append(rouge_scores_fine['ROUGE-1'])
        results["ROUGE-2"].append(rouge_scores_fine['ROUGE-2'])
        results["ROUGE-L"].append(rouge_scores_fine['ROUGE-L'])
        results["BERTScore"].append(bertscore_fine)

        rouge_scores_standard = calculate_rouge(correct_answers[idx], standard_model_responses[idx])
        bertscore_standard = calculate_bertscore([correct_answers[idx]], [standard_model_responses[idx]])

        results["Model"].append("Standard Model")
        results["Prompt"].append(f"Prompt {idx + 1}")
        results["ROUGE-1"].append(rouge_scores_standard['ROUGE-1'])
        results["ROUGE-2"].append(rouge_scores_standard['ROUGE-2'])
        results["ROUGE-L"].append(rouge_scores_standard['ROUGE-L'])
        results["BERTScore"].append(bertscore_standard)

    return pd.DataFrame(results)

def plot_results(df):
    metrics = ["ROUGE-1", "ROUGE-2", "ROUGE-L", "BERTScore"]

    for metric in metrics:
        plt.figure(figsize=(10, 6))
        for model in df['Model'].unique():
            model_data = df[df["Model"] == model]
            plt.bar(model_data['Prompt'], model_data[metric], label=model, alpha=0.6)
        
        plt.title(f"Comparison of {metric} between Fine-Tuned and Standard Models")
        plt.xlabel("Prompts")
        plt.ylabel(metric)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

def load_prompts_and_answers(jsonl_path):
    prompts = []
    correct_answers = []

    with open(jsonl_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            user_message = next((msg["content"] for msg in data["messages"] if msg["role"] == "user"), None)
            assistant_message = next((msg["content"] for msg in data["messages"] if msg["role"] == "assistant"), None)

            if user_message and assistant_message:
                prompts.append(user_message)
                correct_answers.append(assistant_message)

    return prompts, correct_answers

def print_average_scores(df):
    avg_scores = df.groupby("Model")[["ROUGE-1", "ROUGE-2", "ROUGE-L", "BERTScore"]].mean()
    print("Average Scores Comparison:")
    print(avg_scores)
    print(df.to_string(index=False))

def summarize_log(log_text, id):
    response = client.agents.complete(
        agent_id=id,
        messages=[
            {"role": "user", "content": f"{log_text}"},
        ],
    )
    return remove_hashes_and_asterisks(response.choices[0].message.content)

def save_evaluation_to_csv(prompts, correct_answers, all_model_responses, fine_tuned_responses, df, file_path="eval-res.csv"):
    if isinstance(all_model_responses, dict):
        responses_df = pd.DataFrame(all_model_responses)
    else:
        responses_df = pd.DataFrame({
            "Fine-Tuned Response": fine_tuned_responses,
            "Standard Response": all_model_responses
        })

    evaluation_data = {
        "Prompt": prompts,
        "Correct Answer": correct_answers,
    }
    evaluation_df = pd.DataFrame(evaluation_data)

    combined_df = pd.concat([evaluation_df, responses_df, df.reset_index(drop=True)], axis=1)

    combined_df.to_csv(file_path, index=False, encoding='utf-8')
    print(f"Evaluation results saved to {file_path}")

def plot_and_save_results(df, output_folder="charts"):
    import os
    
    os.makedirs(output_folder, exist_ok=True)
    metrics = ["ROUGE-1", "ROUGE-2", "ROUGE-L", "BERTScore"]
    
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        for model in df['Model'].unique():
            model_data = df[df["Model"] == model]
            plt.bar(model_data['Prompt'], model_data[metric], label=model, alpha=0.6)
        
        plt.title(f"Comparison of {metric} between Fine-Tuned and Standard Models")
        plt.xlabel("Prompts")
        plt.ylabel(metric)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt_path = f"{output_folder}/{metric}_comparison.png"
        plt.savefig(plt_path)
        print(f"Saved {metric} chart to {plt_path}")
        plt.close()

def plot_average_scores(df):
    avg_scores = df.groupby("Model")[["ROUGE-1", "ROUGE-2", "ROUGE-L", "BERTScore"]].mean()
    
    print("Average scores (raw data):")
    print(avg_scores)

    avg_scores_transposed = avg_scores.transpose()
    print("Average scores (transposed for plotting):")
    print(avg_scores_transposed)

    if avg_scores_transposed.isnull().values.any():
        print("Data contains NaN values! Check your input data.")
        return

    plt.figure(figsize=(12, 8))
    avg_scores_transposed.plot(kind="bar", figsize=(12, 8), alpha=0.75)

    plt.title("Average Scores by Metric for Each Model")
    plt.xlabel("Metrics")
    plt.ylabel("Scores")
    plt.xticks(rotation=45)
    plt.legend(title="Models")
    plt.tight_layout()

    plt_path = "charts/average_scores_comparison.png"
    plt.savefig(plt_path)
    print(f"Saved average scores chart to {plt_path}")

    plt.draw()
    plt.pause(0.001)
    plt_path = "charts/average_scores_comparison.png"
    plt.savefig(plt_path)
    print(f"Saved average scores chart to {plt_path}")  
    plt.show()

def evaluate_all_models(prompts, correct_answers, model_agent_ids):
    results = {
        "Model": [],
        "Prompt": [],
        "ROUGE-1": [],
        "ROUGE-2": [],
        "ROUGE-L": [],
        "BERTScore": []
    }

    model_responses = {} 
    for model_name, agent_id in model_agent_ids.items():
        print(f"Evaluating {model_name}...")
        responses = [summarize_log(prompt, agent_id) for prompt in prompts]
        model_responses[model_name] = responses
        
        for idx in range(len(prompts)):
            rouge_scores = calculate_rouge(correct_answers[idx], responses[idx])
            bertscore = calculate_bertscore([correct_answers[idx]], [responses[idx]])

            results["Model"].append(model_name)
            results["Prompt"].append(f"Prompt {idx + 1}")
            results["ROUGE-1"].append(rouge_scores['ROUGE-1'])
            results["ROUGE-2"].append(rouge_scores['ROUGE-2'])
            results["ROUGE-L"].append(rouge_scores['ROUGE-L'])
            results["BERTScore"].append(bertscore)

    return pd.DataFrame(results), model_responses


def plot_all_models_results(df):
    metrics = ["ROUGE-1", "ROUGE-2", "ROUGE-L", "BERTScore"]
    
    for metric in metrics:
        plt.figure(figsize=(15, 8))
        prompts = df["Prompt"].unique()
        
        for prompt in prompts:
            prompt_data = df[df["Prompt"] == prompt]
            plt.bar(prompt_data["Model"], prompt_data[metric], label=prompt, alpha=0.6)
        
        plt.title(f"Comparison of {metric} for All Models")
        plt.xlabel("Models")
        plt.ylabel(metric)
        plt.xticks(rotation=45)
        plt.legend(title="Prompts")
        plt.tight_layout()
        plt.show()


def plot_metric_per_prompt(df, metric, output_folder="charts_per_prompt"):
    import os
    os.makedirs(output_folder, exist_ok=True)

    prompts = df["Prompt"].unique()
    models = df["Model"].unique()


    metric_data = []
    for prompt in prompts:
        prompt_data = df[df["Prompt"] == prompt]
        values = [prompt_data[prompt_data["Model"] == model][metric].values[0] for model in models]
        metric_data.append(values)

    metric_data = np.array(metric_data).T


    plt.figure(figsize=(15, 8))
    bar_width = 0.1 
    x = np.arange(len(prompts))

    for i, model in enumerate(models):
        plt.bar(x + i * bar_width, metric_data[i], bar_width, label=model)

    plt.title(f"{metric} Comparison Across Prompts")
    plt.xlabel("Prompts")
    plt.ylabel(metric)
    plt.xticks(x + bar_width * (len(models) / 2), prompts, rotation=45)
    plt.legend(title="Models")
    plt.tight_layout()

    chart_path = f"{output_folder}/{metric}_per_prompt.png"
    plt.savefig(chart_path)
    print(f"Saved {metric} per prompt chart to {chart_path}")
    plt.close()

def plot_metric_averages(df, metric, output_folder="charts_averages"):
    import os
    os.makedirs(output_folder, exist_ok=True)
    models = df["Model"].unique()
    avg_values = [df[df["Model"] == model][metric].mean() for model in models]


    plt.figure(figsize=(10, 6))
    bar_width = 0.5
    x = np.arange(len(models)) 
    plt.bar(x, avg_values, bar_width, color='skyblue', alpha=0.7)
    plt.title(f"Average {metric} Across All Prompts")
    plt.xlabel("Models")
    plt.ylabel(f"Average {metric}")
    plt.xticks(x, models, rotation=45)
    plt.tight_layout()
    chart_path = f"{output_folder}/{metric}_averages.png"
    plt.savefig(chart_path)
    print(f"Saved {metric} averages chart to {chart_path}")
    plt.close()


jsonl_path = "dataset-metrics-fin.jsonl"
prompts, correct_answers = load_prompts_and_answers(jsonl_path)


model_agent_ids = {
    "Standard_Mistral_7B": "ag:3cf886ba:20241013:untitled-agent:35b91216",
    "Finetuned_Mistral_7B": "ag:3cf886ba:20241029:finetune-2:f6f65dcb",
    "Mistral_8x7B": "ag:3cf886ba:20241126:untitled-agent:ead8382b",
    "Mistral_8x22B": "ag:3cf886ba:20241126:untitled-agent:3d24d69e",
    "Mistral_Small_24_02": "ag:3cf886ba:20241126:untitled-agent:f5ade087",
    "Mistral_Medium": "ag:3cf886ba:20241126:untitled-agent:6980fc20",
}

# Generate responses using both models
fine_tuned_responses = [summarize_log(prompt,model_agent_ids['Finetuned_Mistral_7B']) for prompt in prompts]
standard_model_responses = [summarize_log(prompt,model_agent_ids['Standard_Mistral_7B']) for prompt in prompts]



comparison_df = evaluate_models(prompts, correct_answers, fine_tuned_responses, standard_model_responses)


print_average_scores(comparison_df)


save_evaluation_to_csv(prompts, correct_answers, fine_tuned_responses, standard_model_responses, comparison_df)


plot_and_save_results(comparison_df)
plot_average_scores(comparison_df)

comparison_df, all_model_responses = evaluate_all_models(prompts, correct_answers, model_agent_ids)


save_evaluation_to_csv(prompts, correct_answers, all_model_responses, None, comparison_df, file_path="all_models_eval_res.csv")


metrics = ["ROUGE-1", "ROUGE-2", "ROUGE-L", "BERTScore"]
for metric in metrics:
    plot_metric_per_prompt(comparison_df, metric)
    plot_metric_averages(comparison_df, metric)